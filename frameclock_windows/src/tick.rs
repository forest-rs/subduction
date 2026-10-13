// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `FrameTick` construction from QPC reads.

use frameclock::time::Timebase;
use frameclock::{FrameTick, HostTime, OutputId};

/// Build a [`FrameTick`] from QPC. Call inside a `VSync`-paced tick handler.
///
/// `refresh_interval_ns` is the display's refresh interval in nanoseconds,
/// or `0` when unknown. The tick reports it, like every time it carries, in
/// QPC ticks.
#[must_use]
pub fn make_tick(refresh_interval_ns: u64, prev_present_time: Option<HostTime>) -> FrameTick {
    tick_at(
        crate::time::now(),
        crate::time::timebase(),
        refresh_interval_ns,
        prev_present_time,
    )
}

/// Builds the tick for `now`, converting the refresh interval from
/// nanoseconds to ticks of `timebase`.
fn tick_at(
    now: HostTime,
    timebase: Timebase,
    refresh_interval_ns: u64,
    prev_present_time: Option<HostTime>,
) -> FrameTick {
    let interval_ticks = timebase.nanos_to_ticks(refresh_interval_ns);
    let predicted_present = (interval_ticks > 0)
        .then(|| HostTime(prev_present_time.unwrap_or(now).ticks() + interval_ticks));

    FrameTick {
        now,
        predicted_present,
        refresh_interval: (interval_ticks > 0).then_some(interval_ticks),
        output: OutputId(0),
        prev_actual_present: prev_present_time,
    }
}

#[cfg(test)]
mod tests {
    use super::{make_tick, tick_at};
    use frameclock::HostTime;
    use frameclock::time::Timebase;

    /// A 10 MHz QPC, the usual frequency: 100 ns per tick.
    const QPC_10_MHZ: Timebase = Timebase::new(1_000_000_000, 10_000_000);

    #[test]
    fn refresh_interval_is_in_qpc_ticks() {
        let tick = tick_at(HostTime(5_000_000), QPC_10_MHZ, 16_666_667, None);
        // 16.67 ms is 166,666 ticks of 100 ns, not 16,666,667 ticks (1.67 s).
        assert_eq!(tick.refresh_interval, Some(166_666));
        assert_eq!(tick.predicted_present, Some(HostTime(5_166_666)));
    }

    #[test]
    fn prediction_follows_the_previous_present() {
        let prev = HostTime(4_000_000);
        let tick = tick_at(HostTime(5_000_000), QPC_10_MHZ, 16_666_667, Some(prev));
        assert_eq!(tick.predicted_present, Some(HostTime(4_166_666)));
        assert_eq!(tick.prev_actual_present, Some(prev));
    }

    #[test]
    fn make_tick_with_refresh_and_prev() {
        let prev = HostTime(1_000_000);
        let tick = make_tick(16_666_667, Some(prev));
        assert_eq!(tick.prev_actual_present, Some(prev));
        assert!(tick.predicted_present.is_some());
        assert!(tick.refresh_interval.is_some());
    }

    #[test]
    fn make_tick_zero_refresh() {
        let tick = make_tick(0, None);
        assert_eq!(tick.predicted_present, None);
        assert_eq!(tick.refresh_interval, None);
        assert_eq!(tick.prev_actual_present, None);
    }

    #[test]
    fn make_tick_first_frame_predicts_from_now() {
        let tick = make_tick(16_666_667, None);
        // First frame with no prev: predicted_present = now + interval
        let predicted = tick.predicted_present.unwrap();
        assert!(predicted.ticks() > tick.now.ticks());
    }
}
