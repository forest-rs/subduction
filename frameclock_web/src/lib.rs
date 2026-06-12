// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Browser timing adapters for [`frameclock`].
//!
//! This crate owns browser-specific timing adaptation. It converts
//! `requestAnimationFrame` callbacks into [`FrameTick`] values, exposes
//! `performance.now()` as [`HostTime`], and provides [`WebFrameClock`] as a
//! retained wrapper around [`FrameDriver`].
//!
//! It intentionally does not own DOM presentation, WebGL, WebGPU, application
//! state, or renderer submission.

#![no_std]

extern crate alloc;

mod raf;

pub use raf::RafLoop;

use frameclock::time::Timebase;
use frameclock::{
    ActiveFrame, DisplayTiming, Duration, FrameBeginResult, FrameDemand, FrameDriver,
    FrameOpportunity, FrameSubmission, FrameTick, FrameTimingSummary, HostTime, PresentHints,
    SchedulerConfig,
};

/// Browser host-time conversion: 1 tick = 1 microsecond = 1000 nanoseconds.
pub const TIMEBASE: Timebase = Timebase::new(1000, 1);

/// Fallback display interval for browser RAF ticks without an interval.
///
/// The value is a 60 Hz interval in microsecond ticks. Browsers do not expose a
/// portable refresh interval through `requestAnimationFrame`, so callers should
/// treat this only as a conservative pacing fallback.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration(16_667);

/// Returns the current host time from `performance.now()`.
///
/// The returned [`HostTime`] is in microsecond ticks.
#[must_use]
pub fn now() -> HostTime {
    let ms = raf::performance_now();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "performance.now() returns small positive f64 values; microseconds fit in u64"
    )]
    let us = (ms * 1000.0) as u64;
    HostTime(us)
}

/// Returns the browser [`Timebase`].
///
/// `Timebase { numer: 1000, denom: 1 }` means `nanoseconds = ticks * 1000`.
#[must_use]
pub const fn timebase() -> Timebase {
    TIMEBASE
}

/// Computes pacing-only [`PresentHints`] from a browser [`FrameTick`].
///
/// Browsers do not expose a portable predicted present time or commit deadline
/// through `requestAnimationFrame`, so `desired_present` is `None` and
/// `latest_commit` is one refresh interval after the tick's `now`.
#[must_use]
pub fn present_hints(tick: &FrameTick, fallback_refresh_interval: Duration) -> PresentHints {
    let refresh_interval = match tick.refresh_interval {
        Some(ticks) => Duration(ticks),
        None => fallback_refresh_interval,
    };
    PresentHints::pacing_only(
        tick.now
            .checked_add(refresh_interval)
            .unwrap_or(HostTime(u64::MAX)),
    )
}

/// Compatibility helper matching other backend hint functions.
///
/// Prefer [`WebFrameClock`] for retained host integration. The safety margin is
/// intentionally unused because RAF exposes no commit deadline.
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, _safety_margin: Duration) -> PresentHints {
    present_hints(tick, DEFAULT_REFRESH_INTERVAL)
}

/// Returns display timing for a browser RAF tick.
///
/// If the tick carries a predicted present or refresh interval, this delegates
/// to [`DisplayTiming::from_tick`]. Ordinary browser RAF ticks usually do not,
/// so callers should pass a conservative fallback such as
/// [`DEFAULT_REFRESH_INTERVAL`].
#[must_use]
pub fn display_timing(tick: &FrameTick, fallback_interval: Duration) -> DisplayTiming {
    DisplayTiming::from_tick(tick, fallback_interval)
}

/// Retained browser frame lifecycle adapter.
///
/// `WebFrameClock` owns a [`FrameDriver`] and turns RAF [`FrameTick`] values
/// into pacing-only [`FrameOpportunity`] values. Hosts still own redraw demand,
/// application update, rendering, and surface/backend submission.
#[derive(Debug)]
pub struct WebFrameClock {
    driver: FrameDriver,
    fallback_refresh_interval: Duration,
}

impl WebFrameClock {
    /// Creates a browser frame clock using `config` and a fallback interval.
    #[must_use]
    pub fn new(config: SchedulerConfig, fallback_refresh_interval: Duration) -> Self {
        Self::from_driver(FrameDriver::new(config), fallback_refresh_interval)
    }

    /// Creates a browser frame clock around an existing [`FrameDriver`].
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

    /// Returns the fallback refresh interval used for RAF ticks.
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
        FrameOpportunity::new(
            tick,
            present_hints(&tick, self.fallback_refresh_interval),
            display_timing(&tick, self.fallback_refresh_interval),
        )
    }

    /// Begins frame work from a RAF tick.
    #[must_use]
    pub fn begin_frame(&mut self, tick: FrameTick) -> FrameBeginResult {
        let opportunity = self.opportunity(tick);
        self.driver.begin_frame(opportunity)
    }

    /// Reports that a ready frame was submitted.
    #[must_use]
    pub fn submit_frame(
        &mut self,
        frame: ActiveFrame,
        submission: FrameSubmission,
    ) -> FrameTimingSummary {
        self.driver.submit_frame(frame, submission)
    }

    /// Reports a submitted frame at the current browser host time.
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

    fn test_tick() -> FrameTick {
        FrameTick {
            now: HostTime(16_000),
            predicted_present: None,
            refresh_interval: None,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    #[test]
    fn timebase_is_microsecond() {
        let tb = timebase();
        assert_eq!(tb.ticks_to_nanos(1), 1000);
        assert_eq!(tb.ticks_to_nanos(1_000_000), 1_000_000_000);
    }

    #[test]
    fn present_hints_are_pacing_only() {
        let tick = test_tick();
        let hints = present_hints(&tick, DEFAULT_REFRESH_INTERVAL);

        assert_eq!(hints.desired_present(), None);
        assert_eq!(
            hints.latest_commit(),
            HostTime(16_000 + DEFAULT_REFRESH_INTERVAL.ticks())
        );
    }

    #[test]
    fn opportunity_uses_default_display_fallback() {
        let tick = test_tick();
        let clock = WebFrameClock::new(SchedulerConfig::pacing_only(), DEFAULT_REFRESH_INTERVAL);
        let opportunity = clock.opportunity(tick);

        assert_eq!(opportunity.tick, tick);
        assert_eq!(
            opportunity.hints,
            present_hints(&tick, DEFAULT_REFRESH_INTERVAL)
        );
        assert_eq!(
            opportunity.display_timing,
            DisplayTiming::fixed(DEFAULT_REFRESH_INTERVAL)
        );
    }

    #[test]
    fn driver_returns_ready_frame_for_due_demand() {
        let tick = test_tick();
        let mut config = SchedulerConfig::pacing_only();
        config.initial_depth = 1;
        config.minimum_frame_start_margin = Duration::ZERO;
        let mut clock = WebFrameClock::new(config, DEFAULT_REFRESH_INTERVAL);

        clock.request(FrameDemand::INPUT);

        assert!(matches!(
            clock.begin_frame(tick),
            FrameBeginResult::Ready(_)
        ));
    }
}
