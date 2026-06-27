// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `FrameTick` construction from QPC reads.

use frameclock::{FrameTick, HostTime, OutputId};

/// Build a [`FrameTick`] from QPC. Call inside a `VSync`-paced tick handler.
#[must_use]
pub fn make_tick(
    refresh_interval_ns: u64,
    frame_index: u64,
    prev_present_time: Option<HostTime>,
) -> FrameTick {
    let timebase = crate::time::timebase();
    let interval_ticks = if refresh_interval_ns > 0 {
        refresh_interval_ns * u64::from(timebase.denom) / u64::from(timebase.numer)
    } else {
        0
    };

    let now = crate::time::now();

    let predicted_present = if interval_ticks > 0 {
        if let Some(prev) = prev_present_time {
            Some(HostTime(prev.ticks() + interval_ticks))
        } else {
            Some(HostTime(now.ticks() + interval_ticks))
        }
    } else {
        None
    };

    FrameTick {
        now,
        predicted_present,
        refresh_interval: if refresh_interval_ns > 0 {
            Some(refresh_interval_ns)
        } else {
            None
        },
        frame_index,
        output: OutputId(0),
        prev_actual_present: prev_present_time,
    }
}

#[cfg(test)]
mod tests {
    use super::make_tick;
    use frameclock::HostTime;

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
}
