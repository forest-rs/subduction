// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Present-hint computation for Wayland ticks.

use subduction_core::time::Duration;
use subduction_core::timing::{FrameTick, PresentHints};

/// Computes [`PresentHints`] from a [`FrameTick`] and a safety margin.
///
/// This matches the Apple backend policy:
///
/// - `desired_present` is copied from `tick.predicted_present`
/// - `latest_commit` is `desired_present - safety_margin` when possible
/// - otherwise `latest_commit` falls back to `tick.now`
#[must_use]
pub fn compute_present_hints(tick: &FrameTick, safety_margin: Duration) -> PresentHints {
    let desired_present = tick.predicted_present;
    let latest_commit = desired_present
        .and_then(|present| present.checked_sub(safety_margin))
        .unwrap_or(tick.now);

    PresentHints {
        desired_present,
        latest_commit,
    }
}

#[cfg(test)]
mod tests {
    use super::compute_present_hints;
    use subduction_core::output::OutputId;
    use subduction_core::time::{Duration, HostTime};
    use subduction_core::timing::{FrameTick, TimingConfidence};

    fn tick(predicted_present: Option<HostTime>) -> FrameTick {
        FrameTick {
            now: HostTime(1_000_000),
            predicted_present,
            refresh_interval: Some(16_666_667),
            confidence: TimingConfidence::Estimated,
            frame_index: 7,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    #[test]
    fn compute_present_hints_with_prediction() {
        let hints = compute_present_hints(&tick(Some(HostTime(2_000_000))), Duration(500_000));

        assert_eq!(hints.desired_present, Some(HostTime(2_000_000)));
        assert_eq!(hints.latest_commit, HostTime(1_500_000));
    }

    #[test]
    fn compute_present_hints_without_prediction() {
        let hints = compute_present_hints(&tick(None), Duration(500_000));

        assert_eq!(hints.desired_present, None);
        assert_eq!(hints.latest_commit, HostTime(1_000_000));
    }

    #[test]
    fn compute_present_hints_with_underflowing_margin_falls_back_to_now() {
        let hints = compute_present_hints(&tick(Some(HostTime(200_000))), Duration(500_000));

        assert_eq!(hints.desired_present, Some(HostTime(200_000)));
        assert_eq!(hints.latest_commit, HostTime(1_000_000));
    }
}
