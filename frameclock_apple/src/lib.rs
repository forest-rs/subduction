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
use frameclock::timing::PresentationTiming;
use frameclock::{
    ActiveFrame, DisplayTiming, Duration, FrameBeginResult, FrameDemand, FrameDriver,
    FrameOpportunity, FrameSubmission, FrameTick, FrameTimingSummary, HostTime, PresentHints,
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

/// Computes predictive [`PresentHints`] from an Apple display-link tick.
///
/// The desired present time is the tick's predicted present. The latest commit
/// is the predicted present minus the scheduler safety margin, clamped to
/// `tick.now`. If the display-link callback arrives with a stale prediction,
/// the prediction is ignored and the latest commit falls back to `tick.now`.
#[must_use]
pub fn present_hints(tick: &FrameTick, safety_margin: Duration) -> PresentHints {
    let desired_present = tick
        .predicted_present
        .filter(|predicted_present| *predicted_present >= tick.now);
    let latest_commit = desired_present
        .and_then(|present| present.checked_sub(safety_margin))
        .unwrap_or(tick.now)
        .max(tick.now);

    PresentHints::new(
        PresentationTiming::Predictive,
        desired_present,
        latest_commit,
    )
}

/// Compatibility helper matching existing backend naming.
///
/// Prefer [`AppleFrameClock`] for retained host integration.
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, safety_margin: Duration) -> PresentHints {
    present_hints(tick, safety_margin)
}

/// Returns display timing for an Apple display-link tick.
#[must_use]
pub fn display_timing(tick: &FrameTick, fallback_interval: Duration) -> DisplayTiming {
    DisplayTiming::from_tick(tick, fallback_interval)
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
    fallback_refresh_interval: Duration,
}

impl AppleFrameClock {
    /// Creates an Apple frame clock using `config` and a fallback interval.
    #[must_use]
    pub fn new(config: SchedulerConfig, fallback_refresh_interval: Duration) -> Self {
        Self::from_driver(FrameDriver::new(config), fallback_refresh_interval)
    }

    /// Creates an Apple frame clock around an existing [`FrameDriver`].
    #[must_use]
    pub const fn from_driver(driver: FrameDriver, fallback_refresh_interval: Duration) -> Self {
        Self {
            driver,
            fallback_refresh_interval,
        }
    }

    /// Returns the underlying frame driver.
    #[must_use]
    pub const fn driver(&self) -> &FrameDriver {
        &self.driver
    }

    /// Returns the fallback refresh interval used when a tick has no interval.
    #[must_use]
    pub const fn fallback_refresh_interval(&self) -> Duration {
        self.fallback_refresh_interval
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
        let safety_margin = Duration(self.driver.scheduler().safety_margin_ticks());
        FrameOpportunity::new(
            tick,
            present_hints(&tick, safety_margin),
            display_timing(&tick, self.fallback_refresh_interval),
        )
    }

    /// Begins frame work from a display-link tick.
    #[must_use]
    pub fn begin_frame(&mut self, tick: FrameTick) -> FrameBeginResult {
        let opportunity = self.opportunity(tick);
        self.driver.begin_frame(opportunity)
    }

    /// Reports that a ready frame was submitted.
    ///
    /// Use [`FrameSubmission::actual_present`] when a backend has actual
    /// presentation feedback for this frame. Apple display-link sources often
    /// report actual present on a later tick, so callers may pass `None` and use
    /// the returned pacing/target-present summary until a richer deferred
    /// feedback API is needed.
    #[must_use]
    pub fn submit_frame(
        &mut self,
        frame: ActiveFrame,
        submission: FrameSubmission,
    ) -> FrameTimingSummary {
        self.driver.submit_frame(frame, submission)
    }

    /// Reports a submitted frame at the current Mach host time.
    #[must_use]
    pub fn submit_frame_now(&mut self, frame: ActiveFrame) -> FrameTimingSummary {
        self.submit_frame(frame, FrameSubmission::new(now(), None))
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
        let hints = present_hints(&tick(Some(HostTime(2_000_000))), Duration(500_000));

        assert_eq!(hints.presentation_timing(), PresentationTiming::Predictive);
        assert_eq!(hints.desired_present(), Some(HostTime(2_000_000)));
        assert_eq!(hints.latest_commit(), HostTime(1_500_000));
    }

    #[test]
    fn present_hints_without_prediction() {
        let hints = present_hints(&tick(None), Duration(500_000));

        assert_eq!(hints.presentation_timing(), PresentationTiming::Predictive);
        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(1_000_000));
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
        let hints = present_hints(&stale_tick, Duration(500_000));

        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(2_000_000));
    }
}
