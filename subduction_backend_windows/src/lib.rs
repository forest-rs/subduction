// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Windows backend for subduction.
//!
//! This crate provides composable building blocks for driving a subduction
//! layer tree on Windows via `DirectComposition`:
//!
//! - [`DCompPresenter`]: `DirectComposition` visual tree presenter
//! - [`CompositionManager`]: Low-level `DirectComposition` visual tree manager
//! - [`TickSource`] / [`FrameEventTickSource`]: `VSync`-paced tick sources
//! - [`now`] / [`timebase`]: QPC-based timing
//!
//! Windows clock reads and tick construction live in `frameclock_windows`.
//! Use that crate directly for [`now`](frameclock_windows::now),
//! [`timebase`](frameclock_windows::timebase), and
//! [`make_tick`](frameclock_windows::make_tick).
//!
//! # Frame loop
//!
//! ```text
//! TickSource posts WM_APP_TICK to the window on each VSync
//!   → wnd_proc calls handler
//!   → make_tick() → FrameTick
//!   → Scheduler::plan() → FramePlan
//!   → LayerStore::evaluate() → FrameChanges
//!   → DCompPresenter::apply() → update visual tree
//!   → (app renders content into layers via visual_for + SetContent)
//!   → scheduler.observe(feedback)
//! ```
//!
//! # Content rendering
//!
//! This backend manages **compositing only** (transforms, opacity, clips,
//! visibility, topology). Visuals are property-only — they carry no backing
//! surface or swapchain. Applications attach GPU content by obtaining the
//! visual via [`DCompPresenter::visual_for`] and calling `SetContent`.

#![expect(
    unsafe_code,
    reason = "Windows backend requires DirectComposition and Win32 FFI"
)]

pub mod composition;
pub mod presenter;
pub mod surface;
pub mod tick;

pub use composition::{AnimationProperty, CompositionManager, LayerId, PendingAnimation};
pub use presenter::DCompPresenter;
pub use subduction_core::backend::Presenter;
pub use surface::DCompSurfacePresenter;
pub use tick::{FrameEventTickSource, TickSource, WM_APP_TICK, compute_hints, make_tick};
pub use windows::Win32::Graphics::DirectComposition::DCOMPOSITION_FRAME_STATISTICS;

use frameclock::time::Timebase;
use frameclock::{FrameTick, HostTime, PresentHints};

/// Returns the current host time using QPC (`QueryPerformanceCounter`).
#[must_use]
pub fn now() -> HostTime {
    frameclock_windows::now()
}

/// Returns the QPC [`Timebase`].
///
/// `nanos = ticks * numer / denom` where `numer = 1_000_000_000` and
/// `denom = QPC frequency`.
#[must_use]
pub fn timebase() -> Timebase {
    frameclock_windows::timebase()
}

/// Computes [`PresentHints`] from a [`FrameTick`] and a safety margin (nanoseconds).
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, safety_margin_ns: u64) -> PresentHints {
    compute_hints(tick, safety_margin_ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameclock::OutputId;
    use frameclock::timing::PresentationTiming;

    #[test]
    fn timebase_is_valid() {
        let tb = timebase();
        assert_eq!(tb.numer, 1_000_000_000);
        assert!(tb.denom > 0, "QPC frequency must be positive");
    }

    #[test]
    fn now_is_monotonic() {
        let a = now();
        let b = now();
        assert!(b.ticks() >= a.ticks(), "QPC must be monotonic");
    }

    #[test]
    fn compute_hints_with_prediction() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(2_000_000)),
            refresh_interval: Some(16_666_667),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, 2_000_000);

        assert_eq!(hints.presentation_timing(), PresentationTiming::Estimated);
        assert_eq!(hints.desired_present(), Some(HostTime(2_000_000)));
        // latest_commit should be before desired_present
        assert!(hints.latest_commit().ticks() < 2_000_000);
    }

    #[test]
    fn make_tick_with_refresh_and_prev() {
        let prev = HostTime(1_000_000);
        let tick = make_tick(16_666_667, 5, Some(prev));
        assert_eq!(tick.frame_index, 5);
        assert_eq!(tick.prev_actual_present, Some(prev));
        assert!(tick.predicted_present.is_some());
        assert!(tick.refresh_interval.is_some());
    }

    #[test]
    fn make_tick_zero_refresh() {
        let tick = make_tick(0, 1, None);
        assert_eq!(tick.predicted_present, None);
        assert_eq!(tick.refresh_interval, None);
        assert_eq!(tick.prev_actual_present, None);
    }

    #[test]
    fn make_tick_first_frame_predicts_from_now() {
        let tick = make_tick(16_666_667, 0, None);
        // First frame with no prev: predicted_present = now + interval
        let predicted = tick.predicted_present.unwrap();
        assert!(predicted.ticks() > tick.now.ticks());
    }

    #[test]
    fn compute_hints_without_prediction() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: None,
            refresh_interval: None,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, 2_000_000);

        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(1_000_000));
    }
}
