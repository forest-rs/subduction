// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Browser timing adapters for [`frameclock`].
//!
//! This crate owns browser-specific timing adaptation. It converts
//! `requestAnimationFrame` callbacks into [`FrameTick`] values, exposes
//! `performance.now()` as [`HostTime`], and converts RAF ticks into
//! [`FrameOpportunity`] values for callers that own a
//! [`frameclock::FrameDriver`].
//!
//! It intentionally does not own DOM presentation, WebGL, WebGPU, application
//! state, or renderer submission.

#![no_std]

extern crate alloc;

mod raf;

pub use raf::RafLoop;

use frameclock::time::Timebase;
use frameclock::{DisplayTiming, Duration, FrameOpportunity, FrameTick, HostTime, PresentHints};

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

/// Builds a [`FrameOpportunity`] from a browser RAF tick.
///
/// Browser RAF exposes no portable predicted present time or commit deadline,
/// so the opportunity is pacing-only and uses `fallback_interval` for both
/// display timing and the commit boundary.
#[must_use]
pub fn frame_opportunity(tick: FrameTick, fallback_interval: Duration) -> FrameOpportunity {
    let hints = present_hints(&tick, fallback_interval);
    let display_timing = display_timing(&tick, fallback_interval);
    FrameOpportunity::new(tick, hints, display_timing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameclock::{FrameBeginResult, FrameDemand, FrameDriver, OutputId, SchedulerConfig};

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
        let opportunity = frame_opportunity(tick, DEFAULT_REFRESH_INTERVAL);

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
        let mut driver = FrameDriver::new(config);

        driver.request(FrameDemand::INPUT);
        let opportunity = frame_opportunity(tick, DEFAULT_REFRESH_INTERVAL);

        assert!(matches!(
            driver.begin_frame(opportunity).result,
            FrameBeginResult::Ready(_)
        ));
    }
}
