// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Apple display-link timing adapters for [`frameclock`].
//!
//! This crate owns Apple-specific timing adaptation. It converts
//! `CADisplayLink` and `CVDisplayLink` callbacks into [`FrameTick`] values,
//! exposes Mach absolute time as [`HostTime`], and provides
//! [`AppleFrameClock`] as a retained wrapper around [`FrameDriver`].
//!
//! It intentionally does not own `CALayer` trees, `CAMetalLayer` presentation,
//! renderers, windows, or app event-loop policy.
//!
//! This crate keeps its own implementation `no_std`, but the selected
//! Objective-C framework bindings currently require `std`. It is intended to be
//! validated on supported Apple targets, not on generic no-std targets such as
//! `x86_64-unknown-none`.

#![no_std]
#![expect(
    unsafe_code,
    reason = "Apple display-link adapters require Objective-C/CoreVideo FFI"
)]

extern crate alloc;

mod mach_time;

#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
mod cv_display_link;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
mod threading;

#[cfg(feature = "ca-display-link")]
mod ca_display_link;

#[cfg(feature = "ca-display-link")]
pub use ca_display_link::DisplayLink;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
pub use cv_display_link::DisplayLink;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
pub use cv_display_link::DisplayLinkError;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
pub use threading::{TickForwarder, TickSender};

use frameclock::time::Timebase;
use frameclock::{
    ActiveFrame, DisplayTiming, Duration, FrameBegin, FrameDemand, FrameDriver, FrameOpportunity,
    FrameSubmission, FrameSubmitResult, FrameTick, FrameTimingSummary, HostTime, PresentHints,
    SchedulerConfig,
};

/// Returns the current host time using Mach absolute time.
#[must_use]
pub fn now() -> HostTime {
    mach_time::now()
}

/// Returns the Mach absolute time [`Timebase`].
#[must_use]
pub fn timebase() -> Timebase {
    mach_time::timebase()
}

/// Computes [`PresentHints`] from an Apple display-link tick.
///
/// Fresh `CADisplayLink.targetTimestamp` / `CVDisplayLink` output times are
/// treated as predictive present targets. If the prediction is missing or
/// stale, the hint falls back to pacing-only timing with a one-refresh commit
/// boundary. The scheduler applies its own learned build margin later when it
/// turns these platform facts into a [`frameclock::timing::FramePlan`].
#[must_use]
pub fn present_hints(tick: &FrameTick, fallback_refresh_interval: Duration) -> PresentHints {
    if let Some(predicted_present) = tick
        .predicted_present
        .filter(|predicted_present| *predicted_present >= tick.now)
    {
        return PresentHints::predictive(predicted_present, predicted_present);
    }

    let refresh_interval = tick
        .refresh_interval
        .filter(|ticks| *ticks > 0)
        .map(Duration)
        .unwrap_or(fallback_refresh_interval);
    PresentHints::pacing_only(
        tick.now
            .checked_add(refresh_interval)
            .unwrap_or(HostTime(u64::MAX)),
    )
}

/// Compatibility helper matching existing backend naming.
///
/// Prefer [`AppleFrameClock`] for retained host integration.
#[must_use]
pub fn compute_present_hints(
    tick: &FrameTick,
    fallback_refresh_interval: Duration,
) -> PresentHints {
    present_hints(tick, fallback_refresh_interval)
}

/// Returns display timing for an Apple display-link tick and target output.
///
/// Pass a variable [`DisplayTiming`] when the current output is known to be a
/// ProMotion/VRR display. The tick's current interval remains available as
/// [`FrameTick::refresh_interval`], but the scheduler needs the broader
/// per-output range to choose cadence. Fixed fallback timing is refined from
/// the tick when the display link reports an explicit refresh interval.
#[must_use]
pub fn display_timing(tick: &FrameTick, fallback_timing: DisplayTiming) -> DisplayTiming {
    if fallback_timing.is_variable() {
        fallback_timing
    } else {
        DisplayTiming::from_tick(tick, fallback_timing.min_interval())
    }
}

/// Preferred Core Animation frame-rate range.
///
/// This is a platform-neutral mirror of `CAFrameRateRange` so code can compute
/// and test cadence requests without depending on Objective-C bindings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreferredFrameRateRange {
    /// Minimum acceptable frames per second.
    pub minimum: f32,
    /// Maximum acceptable frames per second.
    pub maximum: f32,
    /// Preferred frames per second.
    pub preferred: f32,
}

/// Computes a Core Animation-style frame-rate range for a planned interval.
///
/// `frame_interval` is usually
/// [`FramePlan::frame_interval`](frameclock::timing::FramePlan::frame_interval). The
/// display timing should describe the current target output. For variable
/// displays with unknown direct granularity, the preferred rate may be a stable
/// divisor below the display's slowest direct interval; in that case the
/// returned minimum is widened down to the preferred rate so Core Animation can
/// accept the request.
#[must_use]
pub fn preferred_frame_rate_range(
    frame_interval: Duration,
    display_timing: DisplayTiming,
    timebase: Timebase,
) -> Option<PreferredFrameRateRange> {
    let preferred = fps_for_interval(frame_interval, timebase)?;
    let fastest = fps_for_interval(display_timing.min_interval(), timebase)?;
    let slowest = fps_for_interval(display_timing.max_interval(), timebase)?;
    let maximum = fastest.max(preferred);
    let minimum = preferred.min(slowest).min(maximum);
    Some(PreferredFrameRateRange {
        minimum,
        maximum,
        preferred: preferred.clamp(minimum, maximum),
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "valid display rates are finite positive f32-sized values"
)]
fn fps_for_interval(interval: Duration, timebase: Timebase) -> Option<f32> {
    let nanos = timebase.ticks_to_nanos(interval.ticks());
    if nanos == 0 {
        return None;
    }
    let fps = 1_000_000_000.0 / nanos as f64;
    if !fps.is_finite() || fps <= 0.0 || fps > f64::from(f32::MAX) {
        return None;
    }
    Some(fps as f32)
}

/// Retained Apple frame lifecycle adapter.
///
/// `AppleFrameClock` owns a [`FrameDriver`] and turns display-link
/// [`FrameTick`] values into predictive [`FrameOpportunity`] values. Hosts still
/// own redraw demand, application update, rendering, surface acquisition, and
/// native presentation.
#[derive(Debug)]
pub struct AppleFrameClock {
    driver: FrameDriver,
    display_timing: DisplayTiming,
}

impl AppleFrameClock {
    /// Creates an Apple frame clock using `config` and target-output timing.
    ///
    /// Update [`Self::display_timing`] with [`Self::set_display_timing`] when
    /// a window or layer moves to another display or the platform reports a
    /// changed display mode.
    #[must_use]
    pub fn new(config: SchedulerConfig, display_timing: DisplayTiming) -> Self {
        Self::from_driver(FrameDriver::new(config), display_timing)
    }

    /// Creates an Apple frame clock around an existing [`FrameDriver`].
    #[must_use]
    pub const fn from_driver(driver: FrameDriver, display_timing: DisplayTiming) -> Self {
        Self {
            driver,
            display_timing,
        }
    }

    /// Returns the underlying frame driver.
    #[must_use]
    pub const fn driver(&self) -> &FrameDriver {
        &self.driver
    }

    /// Returns the current target-output display timing.
    #[must_use]
    pub const fn display_timing(&self) -> DisplayTiming {
        self.display_timing
    }

    /// Updates the current target-output display timing.
    pub fn set_display_timing(&mut self, display_timing: DisplayTiming) {
        self.display_timing = display_timing;
    }

    /// Adds host frame demand.
    pub fn request(&mut self, demand: FrameDemand) {
        self.driver.request(demand);
    }

    /// Returns whether demand is waiting for another planning turn.
    #[must_use]
    pub const fn has_pending_demand(&self) -> bool {
        self.driver.has_pending_demand()
    }

    /// Returns the frame-start time for the queued plan, if any.
    #[must_use]
    pub const fn next_frame_start(&self) -> Option<HostTime> {
        self.driver.next_frame_start()
    }

    /// Builds the frame opportunity that this adapter will pass to the driver.
    #[must_use]
    pub fn opportunity(&self, tick: FrameTick) -> FrameOpportunity {
        FrameOpportunity::new(
            tick,
            present_hints(&tick, self.display_timing.min_interval()),
            display_timing(&tick, self.display_timing),
        )
    }

    /// Begins frame work from a display-link tick.
    #[must_use]
    pub fn begin_frame(&mut self, tick: FrameTick) -> FrameBegin {
        let opportunity = self.opportunity(tick);
        self.driver.begin_frame(opportunity)
    }

    /// Reports that a ready frame was submitted.
    #[must_use]
    pub fn submit_frame(
        &mut self,
        frame: ActiveFrame,
        submission: FrameSubmission,
    ) -> FrameSubmitResult {
        self.driver.submit_frame(frame, submission)
    }

    /// Reports a submitted frame at the current Mach host time.
    ///
    /// This uses [`FrameSubmission::deferred`] because Apple display-link ticks
    /// normally report actual presentation for the previous frame on the next
    /// callback.
    #[must_use]
    pub fn submit_frame_now(&mut self, frame: ActiveFrame) -> FrameSubmitResult {
        self.submit_frame(frame, FrameSubmission::deferred(now()))
    }

    /// Drops a ready frame without feeding scheduler feedback.
    #[must_use]
    pub fn discard_frame(&mut self, frame: ActiveFrame) -> FrameTimingSummary {
        self.driver.discard_frame(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameclock::OutputId;
    use frameclock::timing::PresentationTiming;

    fn tick(predicted_present: Option<HostTime>) -> FrameTick {
        FrameTick {
            now: HostTime(1_000_000),
            predicted_present,
            refresh_interval: Some(16_666_667),
            frame_index: 7,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    #[test]
    fn present_hints_with_prediction() {
        let hints = present_hints(&tick(Some(HostTime(2_000_000))), Duration(16_666_667));

        assert_eq!(hints.presentation_timing(), PresentationTiming::Predictive);
        assert_eq!(hints.desired_present(), Some(HostTime(2_000_000)));
        assert_eq!(hints.latest_commit(), HostTime(2_000_000));
    }

    #[test]
    fn present_hints_without_prediction() {
        let hints = present_hints(&tick(None), Duration(16_666_667));

        assert_eq!(hints.presentation_timing(), PresentationTiming::PacingOnly);
        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(17_666_667));
    }

    #[test]
    fn present_hints_ignore_stale_prediction() {
        let stale_tick = FrameTick {
            now: HostTime(2_000_000),
            predicted_present: Some(HostTime(1_900_000)),
            refresh_interval: Some(16_666_667),
            frame_index: 7,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = present_hints(&stale_tick, Duration(16_666_667));

        assert_eq!(hints.presentation_timing(), PresentationTiming::PacingOnly);
        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(18_666_667));
    }

    #[test]
    fn display_timing_keeps_variable_output_range() {
        let output_timing =
            DisplayTiming::variable(Duration(8_333_333), Duration(16_666_667), None);

        assert_eq!(
            display_timing(&tick(Some(HostTime(2_000_000))), output_timing),
            output_timing
        );
    }

    #[test]
    fn display_timing_refines_fixed_fallback_from_tick() {
        assert_eq!(
            display_timing(
                &tick(Some(HostTime(2_000_000))),
                DisplayTiming::fixed(Duration(8_333_333)),
            ),
            DisplayTiming::fixed(Duration(16_666_667))
        );
    }

    #[test]
    fn preferred_frame_rate_range_uses_display_bounds() {
        let range = preferred_frame_rate_range(
            Duration(16_666_667),
            DisplayTiming::variable(Duration(8_333_333), Duration(16_666_667), None),
            Timebase::NANOS,
        )
        .expect("range should be representable");

        assert!((range.minimum - 60.0).abs() < 0.01);
        assert!((range.maximum - 120.0).abs() < 0.01);
        assert!((range.preferred - 60.0).abs() < 0.01);
    }

    #[test]
    fn preferred_frame_rate_range_can_request_stable_divisor_below_direct_range() {
        let range = preferred_frame_rate_range(
            Duration(33_333_333),
            DisplayTiming::variable(Duration(8_333_333), Duration(16_666_667), None),
            Timebase::NANOS,
        )
        .expect("range should be representable");

        assert!((range.minimum - 30.0).abs() < 0.01);
        assert!((range.maximum - 120.0).abs() < 0.01);
        assert!((range.preferred - 30.0).abs() < 0.01);
    }
}
