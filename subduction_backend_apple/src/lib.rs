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
//! - [`TickForwarder`] / [`TickSender`]: Tick forwarding for `CVDisplayLink`
//!   (cross-thread dispatch, requires `cv-display-link` feature)

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

use subduction_core::time::{Duration, HostTime};
use subduction_core::timing::{FrameTick, PresentHints};

/// Computes [`PresentHints`] from a [`FrameTick`] and a safety margin.
///
/// This is the standard Apple hint computation: the desired present time is
/// the tick's predicted present, and the latest commit is the predicted
/// present minus the safety margin.
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, safety_margin: Duration) -> PresentHints {
    let desired_present = tick.predicted_present;
    let latest_commit = desired_present
        .and_then(|pp| pp.checked_sub(safety_margin))
        .unwrap_or(tick.now);

    PresentHints {
        desired_present,
        latest_commit,
    }
}

/// Returns the current host time using Mach absolute time.
#[must_use]
pub fn now() -> HostTime {
    mach_time::now()
}

/// Returns the Mach absolute time [`Timebase`](subduction_core::time::Timebase).
#[must_use]
pub fn timebase() -> subduction_core::time::Timebase {
    mach_time::timebase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use subduction_core::output::OutputId;
    use subduction_core::scheduler::{Scheduler, SchedulerConfig};
    use subduction_core::timing::{PendingFeedback, PresentFeedback, TimingConfidence};

    #[test]
    fn compute_present_hints_with_prediction() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(2_000_000)),
            refresh_interval: Some(16_666_666),
            confidence: TimingConfidence::Predictive,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(500_000));

        assert_eq!(hints.desired_present, Some(HostTime(2_000_000)));
        assert_eq!(hints.latest_commit, HostTime(1_500_000));
    }

    #[test]
    fn compute_present_hints_without_prediction() {
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: None,
            refresh_interval: None,
            confidence: TimingConfidence::Estimated,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = compute_present_hints(&tick, Duration(500_000));

        assert_eq!(hints.desired_present, None);
        assert_eq!(hints.latest_commit, HostTime(1_000_000));
    }

    #[test]
    fn scheduler_observe_can_be_driven_by_apple_feedback() {
        let hints = PresentHints {
            desired_present: Some(HostTime(2_000_000)),
            latest_commit: HostTime(1_800_000),
        };
        let mut scheduler = Scheduler::new(SchedulerConfig::macos());

        let feedback = PresentFeedback::new(&hints, HostTime(1_700_000), HostTime(1_900_000), None);
        scheduler.observe(&feedback);
        scheduler.observe(&feedback);
        scheduler.observe(&feedback);

        // Three misses should increase depth by one.
        assert_eq!(scheduler.pipeline_depth(), 3);
    }

    #[test]
    fn scheduler_observe_with_actual_present_uses_tier1() {
        let hints = PresentHints {
            desired_present: Some(HostTime(2_000_000)),
            latest_commit: HostTime(1_800_000),
        };
        let mut scheduler = Scheduler::new(SchedulerConfig::macos());

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

        assert_eq!(scheduler.pipeline_depth(), 3);
    }
}
