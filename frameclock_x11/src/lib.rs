// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! X11 timing adapters for [`frameclock`], built on the Present extension.
//!
//! The crate assumes a server with the Present extension and has no
//! pre-Present fallbacks. It reads the Present unadjusted-system-time clock as
//! [`HostTime`], converts `PresentCompleteNotify` completions into
//! [`FrameTick`] values via [`PresentTicker`], and carries per-pixmap Present
//! feedback as [`PresentEvent`] values.
//!
//! It intentionally does not own X11 windows, connections, event queues, or
//! protocol dispatch. Protocol I/O belongs to hosts and backend crates; this
//! crate owns the timing bookkeeping those hosts feed and poll, and references
//! no `x11rb` type.
//!
//! # Core flow
//!
//! ```text
//! PresentNotifyMSC request  -> PresentTicker (claims the in-flight slot)
//! PresentCompleteNotify      -> PresentTicker -> FrameTick
//! PresentCompleteNotify(px)  -> PresentEvent  -> PresentEventQueue (backend path)
//! ```
//!
//! A host's frame loop has this shape:
//!
//! ```rust,ignore
//! use frameclock::OutputId;
//! use frameclock_x11::{Clock, PresentTicker, frame_opportunity};
//!
//! let mut ticker = PresentTicker::new();
//!
//! // Arm the next vblank notify before sending a PresentNotifyMSC request:
//! if ticker.mark_notify_requested() {
//!     // present_notify_msc(window, serial, 0, 1, 0)  -- notify at the next msc
//! }
//!
//! // When the matching PresentCompleteNotify (kind NotifyMSC) arrives:
//! ticker.on_complete_notify(Clock::Monotonic, event.ust, event.msc);
//!
//! // After dispatch, drain the queued ticks and build opportunities:
//! while let Some(tick) = ticker.poll_tick() {
//!     let opportunity = frame_opportunity(tick, fallback_refresh_interval);
//!     _ = opportunity;
//! }
//! ```
//!
//! # The `ust` clock domain
//!
//! Present reports `ust` (unadjusted system time) in microseconds. On Linux with
//! DRI3/Present that is `CLOCK_MONOTONIC`, which [`Clock::Monotonic`] reads in
//! nanoseconds, so `ust`-derived timestamps and clock reads share one domain —
//! the same domain a host should stamp input events in. All [`HostTime`] values
//! are nanosecond ticks.
//!
//! # Pacing while a swapchain presents
//!
//! `PresentNotifyMSC` reads the window's vblank clock without presenting
//! anything, so the tick stream stays vsync-aligned even when a swapchain
//! owner such as wgpu/Vulkan performs the actual scan-out. Per-submission
//! feedback — which vblank a frame landed on, flip vs. copy, buffer idleness —
//! only reaches a presenter that itself submits `PresentPixmap`; the
//! [`PresentEvent`] queue models that feedback for such a presenter.

#![no_std]

extern crate alloc;

mod presentation;
mod queue;
mod tick;
mod time;

pub use presentation::{PresentEvent, PresentEventQueue, SubmissionId};
pub use tick::PresentTicker;
pub use time::{Clock, now, timebase, ust_to_host_time};

use frameclock::{DisplayTiming, Duration, FrameOpportunity, FrameTick, HostTime, PresentHints};

/// Returns the default commit lead for a refresh interval.
///
/// A predicted present time describes a vblank slot, not a promise that app work
/// can be committed at the last possible tick. Use a small platform-side lead so
/// [`PresentHints::latest_commit`] remains a commit boundary, while `frameclock`
/// still owns learned app build margins.
#[must_use]
pub const fn default_commit_lead(refresh_interval: Duration) -> Duration {
    refresh_interval.div_u64(4)
}

fn refresh_interval_for_tick(tick: &FrameTick, fallback_refresh_interval: Duration) -> Duration {
    tick.refresh_interval
        .filter(|ticks| *ticks > 0)
        .map(Duration)
        .unwrap_or(fallback_refresh_interval)
}

fn commit_boundary(target: HostTime, lead: Duration, floor: HostTime) -> HostTime {
    target.checked_sub(lead).unwrap_or(floor).max(floor)
}

/// Computes [`PresentHints`] from an X11 [`FrameTick`] using the default commit
/// lead.
///
/// Use [`present_hints_with_commit_lead`] when a host has a platform-specific
/// commit lead estimate.
#[must_use]
pub fn present_hints(tick: &FrameTick, fallback_refresh_interval: Duration) -> PresentHints {
    let refresh_interval = refresh_interval_for_tick(tick, fallback_refresh_interval);
    present_hints_with_commit_lead(
        tick,
        fallback_refresh_interval,
        default_commit_lead(refresh_interval),
    )
}

/// Computes [`PresentHints`] from an X11 [`FrameTick`].
///
/// A predicted present time derived from Present `ust`/`msc` (see
/// [`PresentTicker`]) is a client-side extrapolation of the vblank grid, so it
/// is reported as estimated timing (`PresentationTiming::Estimated`) rather than
/// predictive. If the prediction is missing or stale, the hint falls back to
/// pacing-only timing with a one-refresh commit boundary. The scheduler applies
/// its own learned build margin later.
#[must_use]
pub fn present_hints_with_commit_lead(
    tick: &FrameTick,
    fallback_refresh_interval: Duration,
    commit_lead: Duration,
) -> PresentHints {
    let refresh_interval = refresh_interval_for_tick(tick, fallback_refresh_interval);
    if let Some(predicted_present) = tick
        .predicted_present
        .filter(|predicted_present| *predicted_present >= tick.now)
    {
        return PresentHints::estimated(
            predicted_present,
            commit_boundary(predicted_present, commit_lead, tick.now),
        );
    }

    let pacing_target = tick
        .now
        .checked_add(refresh_interval)
        .unwrap_or(HostTime(u64::MAX));
    PresentHints::pacing_only(commit_boundary(pacing_target, commit_lead, tick.now))
}

/// Returns display timing for an X11 [`FrameTick`].
///
/// Prefers [`FrameTick::refresh_interval`] when present (the cadence measured
/// from consecutive `msc` deltas), falling back to the predicted-present delta
/// and finally to `fallback_interval`. The Present extension does not expose a
/// variable-refresh range here, so this always produces fixed-rate timing.
#[must_use]
pub fn display_timing(tick: &FrameTick, fallback_interval: Duration) -> DisplayTiming {
    DisplayTiming::from_tick(tick, fallback_interval)
}

/// Builds a [`FrameOpportunity`] from an X11 Present tick.
///
/// This pairs the estimated/pacing present hints with display timing derived
/// from the same tick, falling back to `fallback_interval` until a second
/// completion has established a cadence.
#[must_use]
pub fn frame_opportunity(tick: FrameTick, fallback_interval: Duration) -> FrameOpportunity {
    let refresh_interval = refresh_interval_for_tick(&tick, fallback_interval);
    frame_opportunity_with_commit_lead(
        tick,
        fallback_interval,
        default_commit_lead(refresh_interval),
    )
}

/// Builds a [`FrameOpportunity`] with an explicit platform commit lead.
#[must_use]
pub fn frame_opportunity_with_commit_lead(
    tick: FrameTick,
    fallback_interval: Duration,
    commit_lead: Duration,
) -> FrameOpportunity {
    let hints = present_hints_with_commit_lead(&tick, fallback_interval, commit_lead);
    let display_timing = display_timing(&tick, fallback_interval);
    FrameOpportunity::new(tick, hints, display_timing)
}

#[cfg(test)]
mod tests {
    use super::{
        default_commit_lead, display_timing, frame_opportunity, present_hints,
        present_hints_with_commit_lead,
    };
    use frameclock::OutputId;
    use frameclock::timing::PresentationTiming;
    use frameclock::{DisplayTiming, Duration, FrameTick, HostTime};

    fn tick(predicted_present: Option<HostTime>) -> FrameTick {
        FrameTick {
            now: HostTime(1_000_000),
            predicted_present,
            refresh_interval: Some(16_666_667),
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    #[test]
    fn present_hints_with_prediction_are_estimated() {
        let hints = present_hints(&tick(Some(HostTime(20_000_000))), Duration(16_666_667));

        assert_eq!(hints.presentation_timing(), PresentationTiming::Estimated);
        assert_eq!(hints.desired_present(), Some(HostTime(20_000_000)));
        assert_eq!(hints.latest_commit(), HostTime(15_833_334));
    }

    #[test]
    fn present_hints_respect_explicit_commit_lead() {
        let hints = present_hints_with_commit_lead(
            &tick(Some(HostTime(20_000_000))),
            Duration(16_666_667),
            Duration(2_000_000),
        );

        assert_eq!(hints.presentation_timing(), PresentationTiming::Estimated);
        assert_eq!(hints.desired_present(), Some(HostTime(20_000_000)));
        assert_eq!(hints.latest_commit(), HostTime(18_000_000));
    }

    #[test]
    fn present_hints_without_prediction_are_pacing_only() {
        let hints = present_hints(&tick(None), Duration(16_666_667));

        assert_eq!(hints.presentation_timing(), PresentationTiming::PacingOnly);
        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(13_500_001));
    }

    #[test]
    fn present_hints_ignore_stale_prediction() {
        let stale_tick = FrameTick {
            now: HostTime(2_000_000),
            predicted_present: Some(HostTime(1_900_000)),
            refresh_interval: Some(16_666_667),
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = present_hints(&stale_tick, Duration(16_666_667));

        assert_eq!(hints.presentation_timing(), PresentationTiming::PacingOnly);
        assert_eq!(hints.desired_present(), None);
    }

    #[test]
    fn default_commit_lead_is_quarter_refresh() {
        assert_eq!(
            default_commit_lead(Duration(16_666_667)),
            Duration(4_166_666)
        );
    }

    #[test]
    fn display_timing_prefers_reported_refresh_interval() {
        assert_eq!(
            display_timing(&tick(Some(HostTime(2_000_000))), Duration(8_333_333)),
            DisplayTiming::fixed(Duration(16_666_667))
        );
    }

    #[test]
    fn frame_opportunity_pairs_tick_hints_and_display_timing() {
        let tick = tick(Some(HostTime(20_000_000)));
        let opportunity = frame_opportunity(tick, Duration(8_333_333));

        assert_eq!(opportunity.tick, tick);
        assert_eq!(
            opportunity.hints.presentation_timing(),
            PresentationTiming::Estimated
        );
        assert_eq!(
            opportunity.hints.desired_present(),
            Some(HostTime(20_000_000))
        );
        assert_eq!(
            opportunity.display_timing,
            DisplayTiming::fixed(Duration(16_666_667))
        );
    }
}
