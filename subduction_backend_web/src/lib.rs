// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Web backend for subduction.
//!
//! This crate provides integration with browser APIs:
//!
//! - [`RafLoop`]: `requestAnimationFrame` tick source (pacing-only timing)
//! - [`DomPresenter`]: DOM element management

#![no_std]

extern crate alloc;

mod presenter;
mod raf;

pub use presenter::DomPresenter;
pub use raf::RafLoop;
pub use subduction_core::backend::Presenter;

use subduction_core::time::{Duration, HostTime, Timebase};
use subduction_core::timing::{FrameTick, PresentHints};

/// Returns the current host time from `performance.now()`.
///
/// The returned [`HostTime`] is in microsecond ticks. Use [`timebase`] to
/// convert to nanoseconds.
#[must_use]
pub fn now() -> HostTime {
    let ms = raf::performance_now();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "performance.now() returns small positive f64; µs fits in u64"
    )]
    let us = (ms * 1000.0) as u64;
    HostTime(us)
}

/// Returns the web [`Timebase`]: 1 tick = 1 µs = 1000 ns.
///
/// `Timebase { numer: 1000, denom: 1 }` means `nanoseconds = ticks × 1000`.
#[must_use]
pub fn timebase() -> Timebase {
    Timebase::new(1000, 1)
}

/// Computes [`PresentHints`] from a [`FrameTick`] and a safety margin.
///
/// The web provides no predicted present time, so `desired_present` is always
/// `None` and `latest_commit` is simply the tick's `now`. The safety margin is
/// accepted for API compatibility but unused.
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, _safety_margin: Duration) -> PresentHints {
    PresentHints {
        desired_present: None,
        latest_commit: tick.now,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subduction_core::output::OutputId;
    use subduction_core::timing::TimingConfidence;

    #[test]
    fn timebase_is_microsecond() {
        let tb = timebase();
        // 1 tick = 1 µs = 1000 ns
        assert_eq!(tb.ticks_to_nanos(1), 1000);
        assert_eq!(tb.ticks_to_nanos(1_000_000), 1_000_000_000);
    }

    #[test]
    fn compute_present_hints_returns_pacing_only() {
        let tick = FrameTick {
            now: HostTime(16_000),
            predicted_present: None,
            refresh_interval: None,
            confidence: TimingConfidence::PacingOnly,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(1_000));

        assert_eq!(hints.desired_present, None);
        assert_eq!(hints.latest_commit, HostTime(16_000));
    }
}
