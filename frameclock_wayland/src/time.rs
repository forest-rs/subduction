// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland host clock selection and reads.

use frameclock::HostTime;
use frameclock::time::Timebase;
use rustix::time::{ClockId as PosixClockId, Timespec, clock_gettime};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Clock source used for Wayland timing facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Clock {
    /// `CLOCK_MONOTONIC` fallback clock.
    #[default]
    Monotonic,
    /// Clock selected from `wp_presentation.clock_id`.
    Presentation(PosixClockId),
}

impl Clock {
    /// Attempts to map a `wp_presentation.clock_id` value to a [`Clock`].
    ///
    /// Returns `Some(Clock::Presentation(...))` if the raw id maps to a POSIX
    /// clock recognized by the platform. Returns `None` for unknown or
    /// out-of-range values, in which case the caller should keep the current
    /// clock and degrade gracefully.
    #[must_use]
    pub fn from_presentation_clock_id(clk_id: u32) -> Option<Self> {
        // `PosixClockId` is `u32` on Apple platforms, `i32` elsewhere
        #[cfg(not(target_vendor = "apple"))]
        let posix_id = {
            let raw = i32::try_from(clk_id).ok()?;
            PosixClockId::try_from(raw).ok()?
        };
        #[cfg(target_vendor = "apple")]
        let posix_id = PosixClockId::try_from(clk_id).ok()?;
        Some(Self::Presentation(posix_id))
    }

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
            Self::Presentation(clock_id) => clock_id,
        }
    }
}

/// Returns the Wayland [`Timebase`]: host ticks are nanoseconds.
#[must_use]
pub const fn timebase() -> Timebase {
    Timebase::NANOS
}

/// Returns the current monotonic host time in nanoseconds.
///
/// Hosts tracking a `wp_presentation` clock domain should prefer
/// [`Clock::now`] on the selected clock so all timing facts stay comparable.
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
    use super::{Clock, now, timebase, timespec_to_host_time};
    use frameclock::HostTime;
    use frameclock::time::Timebase;
    use rustix::time::{ClockId as PosixClockId, Timespec};

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
        let tick = Clock::Presentation(PosixClockId::Monotonic).now();
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

    #[test]
    fn clock_from_known_monotonic_id() {
        let clk_id = PosixClockId::Monotonic as u32;
        let clock = Clock::from_presentation_clock_id(clk_id).unwrap();
        assert_eq!(clock, Clock::Presentation(PosixClockId::Monotonic));
        // The returned clock must be readable.
        assert!(clock.now().ticks() > 0);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn clock_from_known_monotonic_raw_id() {
        let clk_id = PosixClockId::MonotonicRaw as u32;
        let clock = Clock::from_presentation_clock_id(clk_id).unwrap();
        assert_eq!(clock, Clock::Presentation(PosixClockId::MonotonicRaw));
        assert!(clock.now().ticks() > 0);
    }

    #[test]
    fn clock_from_unknown_in_range_id() {
        // A value that fits in i32 but is not a recognized POSIX clock.
        assert!(Clock::from_presentation_clock_id(12345).is_none());
    }

    #[test]
    fn clock_from_overflow_id() {
        // u32::MAX overflows i32, should be rejected.
        assert!(Clock::from_presentation_clock_id(u32::MAX).is_none());
    }
}
