// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland backend host clock selection and reads.

use rustix::time::{ClockId as PosixClockId, Timespec, clock_gettime};
use subduction_core::time::{HostTime, Timebase};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Clock source used for backend timestamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Clock {
    /// `CLOCK_MONOTONIC` fallback clock.
    #[default]
    Monotonic,
    /// Clock selected from `wp_presentation.clock_id`.
    Presentation(PosixClockId),
}

impl Clock {
    #[must_use]
    const fn posix_clock_id(self) -> PosixClockId {
        match self {
            Self::Monotonic => PosixClockId::Monotonic,
            Self::Presentation(clock_id) => clock_id,
        }
    }
}

/// Returns the Wayland backend [`Timebase`]: host ticks are nanoseconds.
#[must_use]
pub const fn timebase() -> Timebase {
    Timebase::NANOS
}

/// Returns the current monotonic host time in nanoseconds.
#[must_use]
pub fn now() -> HostTime {
    now_for_clock(Clock::Monotonic)
}

pub(crate) fn now_for_clock(clock: Clock) -> HostTime {
    let timespec = clock_gettime(clock.posix_clock_id());
    timespec_to_host_time(timespec)
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
    use super::{Clock, now, now_for_clock, timebase, timespec_to_host_time};
    use rustix::time::{ClockId as PosixClockId, Timespec};
    use subduction_core::time::{HostTime, Timebase};

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
    fn presentation_clock_variant_is_usable() {
        let tick = now_for_clock(Clock::Presentation(PosixClockId::Monotonic));
        assert!(
            tick.ticks() > 0,
            "clock_gettime(monotonic) should be positive"
        );
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
