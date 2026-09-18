// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! X11 host clock reads and Present timestamp conversion.

use frameclock::HostTime;
use frameclock::time::Timebase;
use rustix::time::{ClockId as PosixClockId, Timespec, clock_gettime};

const NANOS_PER_SECOND: u128 = 1_000_000_000;
const NANOS_PER_MICRO: u64 = 1_000;

/// Clock source used for X11 timing facts.
///
/// The Present extension reports `ust` (unadjusted system time) in the server's
/// monotonic clock domain. On Linux with DRI3/Present that is `CLOCK_MONOTONIC`
/// in microseconds, the same clock this reads in nanoseconds, so
/// [`ust_to_host_time`] and [`Clock::now`] produce comparable [`HostTime`]
/// values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Clock {
    /// `CLOCK_MONOTONIC`, the domain Present `ust` timestamps are reported in.
    #[default]
    Monotonic,
}

impl Clock {
    /// Returns the current host time read from this clock in nanoseconds.
    #[must_use]
    pub fn now(self) -> HostTime {
        let timespec = clock_gettime(self.posix_clock_id());
        timespec_to_host_time(timespec)
    }

    #[must_use]
    const fn posix_clock_id(self) -> PosixClockId {
        match self {
            Self::Monotonic => PosixClockId::Monotonic,
        }
    }
}

/// Converts a Present `ust` value (microseconds) to [`HostTime`] nanoseconds.
///
/// The Present extension reports `ust` in microseconds; host time is
/// nanoseconds. Arithmetic saturates so an out-of-range server value can never
/// panic.
#[must_use]
pub const fn ust_to_host_time(ust_micros: u64) -> HostTime {
    HostTime(ust_micros.saturating_mul(NANOS_PER_MICRO))
}

/// Returns the X11 [`Timebase`]: host ticks are nanoseconds.
#[must_use]
pub const fn timebase() -> Timebase {
    Timebase::NANOS
}

/// Returns the current monotonic host time in nanoseconds.
#[must_use]
pub fn now() -> HostTime {
    Clock::Monotonic.now()
}

fn timespec_to_host_time(timespec: Timespec) -> HostTime {
    let seconds = u64::try_from(timespec.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(timespec.tv_nsec)
        .unwrap_or(0)
        .min(999_999_999);

    let ticks_u128 = u128::from(seconds)
        .saturating_mul(NANOS_PER_SECOND)
        .saturating_add(u128::from(nanos));
    let ticks = u64::try_from(ticks_u128).unwrap_or(u64::MAX);
    HostTime(ticks)
}

#[cfg(test)]
mod tests {
    use super::{Clock, now, timebase, timespec_to_host_time, ust_to_host_time};
    use frameclock::HostTime;
    use frameclock::time::Timebase;
    use rustix::time::Timespec;

    #[test]
    fn timebase_is_nanos_identity() {
        assert_eq!(timebase(), Timebase::NANOS);
    }

    #[test]
    fn now_is_monotonic_non_decreasing() {
        let first = now();
        let second = now();
        assert!(second >= first, "monotonic clock should not go backwards");
    }

    #[test]
    fn monotonic_clock_is_readable() {
        assert!(
            Clock::Monotonic.now().ticks() > 0,
            "clock_gettime(monotonic) should be positive"
        );
    }

    #[test]
    fn ust_micros_scale_to_nanoseconds() {
        assert_eq!(ust_to_host_time(0), HostTime(0));
        assert_eq!(ust_to_host_time(1), HostTime(1_000));
        assert_eq!(ust_to_host_time(1_500_000), HostTime(1_500_000_000));
    }

    #[test]
    fn ust_conversion_saturates_on_overflow() {
        assert_eq!(ust_to_host_time(u64::MAX), HostTime(u64::MAX));
    }

    #[test]
    fn timespec_conversion_builds_nanosecond_ticks() {
        let input = Timespec {
            tv_sec: 12,
            tv_nsec: 345_678_901,
        };
        let expected = HostTime(12 * 1_000_000_000 + 345_678_901);
        assert_eq!(timespec_to_host_time(input), expected);
    }

    #[test]
    fn timespec_conversion_saturates_on_large_values() {
        let input = Timespec {
            tv_sec: i64::MAX,
            tv_nsec: 999_999_999,
        };
        assert_eq!(timespec_to_host_time(input), HostTime(u64::MAX));
    }
}
