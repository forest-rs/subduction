// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Apple backend for subduction.
//!
//! This crate provides composable building blocks for driving a subduction
//! layer tree on Apple platforms (macOS, iOS, tvOS, visionOS):
//!
//! - [`LayerRoot`]: root `CALayer` container for a scene
//! - [`DisplayLink`]: Tick source (`CADisplayLink` or legacy `CVDisplayLink`)
//! - [`LayerPresenter`]: `CALayer` tree presenter
//! - [`MetalLayerPresenter`]: `CAMetalLayer` presenter
//! - `TickForwarder` / `TickSender`: Tick forwarding for `CVDisplayLink`
//!   (cross-thread dispatch, requires `cv-display-link` feature)
//!
//! `DisplayLink` is intentionally only a timing source. Host code should own
//! frame lifecycle state such as a [`frameclock::FrameDriver`] per render
//! surface/output, feed display-link ticks into it on that owner's thread, and
//! feed submission/presentation feedback back after rendering. `CADisplayLink`
//! callbacks already run on the main run loop; `CVDisplayLink` callbacks run
//! on a `CoreVideo` background thread and should be forwarded with
//! `TickForwarder` before touching non-thread-safe frame state.
//!
//! Apple-specific adapters that drive `ProMotion` should treat
//! [`frameclock::FramePlan::frame_interval`] as the scheduler's cadence
//! decision and translate it into the platform's preferred frame-rate API.

#![no_std]
#![expect(
    unsafe_code,
    reason = "Apple backend requires extensive Objective-C FFI"
)]

extern crate alloc;

mod calayer;
mod cametal;
mod mach_time;

#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
mod cv_display_link;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
mod threading;

#[cfg(feature = "ca-display-link")]
mod ca_display_link;

pub use calayer::{LayerPresenter, LayerRoot};
pub use cametal::MetalLayerPresenter;
pub use subduction_core::backend::Presenter;

// Re-export from whichever display link is enabled.
// ca-display-link takes precedence if both are enabled.
#[cfg(feature = "ca-display-link")]
pub use ca_display_link::DisplayLink;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
pub use cv_display_link::DisplayLink;

#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
pub use cv_display_link::DisplayLinkError;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
pub use threading::{TickForwarder, TickSender};

use frameclock::{Duration, FrameTick, HostTime, PresentHints};

/// Computes [`PresentHints`] from a [`FrameTick`] and a safety margin.
///
/// This is the standard Apple hint computation: the desired present time is
/// the tick's predicted present, and the latest commit is the predicted
/// present minus the safety margin. If a display-link callback arrives late
/// with a predicted present time that is already behind `tick.now`, the
/// prediction is ignored and the latest commit falls back to `tick.now`.
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, safety_margin: Duration) -> PresentHints {
    let desired_present = tick
        .predicted_present
        .filter(|predicted_present| *predicted_present >= tick.now);
    let latest_commit = desired_present
        .and_then(|pp| pp.checked_sub(safety_margin))
        .unwrap_or(tick.now)
        .max(tick.now);

    PresentHints::new(
        frameclock::PresentationTiming::Predictive,
        desired_present,
        latest_commit,
    )
}

/// Returns the current host time using Mach absolute time.
#[must_use]
pub fn now() -> HostTime {
    mach_time::now()
}

/// Returns the Mach absolute time [`Timebase`](frameclock::Timebase).
#[must_use]
pub fn timebase() -> frameclock::Timebase {
    mach_time::timebase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameclock::{
        OutputId, PendingFeedback, PresentFeedback, PresentationTiming, Scheduler, SchedulerConfig,
    };

    #[test]
    fn compute_present_hints_with_prediction() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(2_000_000)),
            refresh_interval: Some(16_666_666),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(500_000));

        assert_eq!(hints.presentation_timing(), PresentationTiming::Predictive);
        assert_eq!(hints.desired_present(), Some(HostTime(2_000_000)));
        assert_eq!(hints.latest_commit(), HostTime(1_500_000));
    }

    #[test]
    fn compute_present_hints_without_prediction() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: None,
            refresh_interval: None,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(500_000));

        assert_eq!(hints.presentation_timing(), PresentationTiming::Predictive);
        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(1_000_000));
    }

    #[test]
    fn compute_present_hints_ignores_stale_prediction() {
        let tick = FrameTick {
            now: HostTime(2_000_000),
            predicted_present: Some(HostTime(1_900_000)),
            refresh_interval: Some(16_666_666),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(500_000));

        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(2_000_000));
    }

    #[test]
    fn compute_present_hints_clamps_due_deadline_to_now() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(1_200_000)),
            refresh_interval: Some(16_666_666),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(500_000));

        assert_eq!(hints.desired_present(), Some(HostTime(1_200_000)));
        assert_eq!(hints.latest_commit(), HostTime(1_000_000));
    }

    #[test]
    fn scheduler_observe_uses_commit_only_feedback_as_soft_signal() {
        let hints = PresentHints::predictive(HostTime(2_000_000), HostTime(1_800_000));
        let mut scheduler = Scheduler::new(SchedulerConfig::predictive());

        let feedback = PresentFeedback::new(&hints, HostTime(1_700_000), HostTime(1_900_000), None);
        for _ in 0..5 {
            scheduler.observe(&feedback);
        }

        assert_eq!(feedback.missed_deadline, None);
        assert_eq!(feedback.pacing_overrun, Some(true));
        assert_eq!(scheduler.pipeline_depth(), 1);

        scheduler.observe(&feedback);
        assert_eq!(scheduler.pipeline_depth(), 2);
    }

    #[test]
    fn scheduler_observe_with_actual_present_uses_tier1() {
        let hints = PresentHints::predictive(HostTime(2_000_000), HostTime(1_800_000));
        let mut scheduler = Scheduler::new(SchedulerConfig::predictive());

        // Simulate deferred feedback: commit was on time, but actual present
        // was late (GPU stall / compositor delay).
        let pending = PendingFeedback {
            hints,
            build_start: HostTime(1_700_000),
            submitted_at: HostTime(1_750_000), // well before latest_commit
        };
        let feedback = pending.resolve(Some(HostTime(2_100_000))); // actual > desired

        // Tier-1 should detect the miss even though commit was on time.
        assert_eq!(feedback.missed_deadline, Some(true));
        assert_eq!(feedback.actual_present, Some(HostTime(2_100_000)));

        scheduler.observe(&feedback);
        scheduler.observe(&feedback);
        scheduler.observe(&feedback);

        assert_eq!(scheduler.pipeline_depth(), 2);
    }
}
