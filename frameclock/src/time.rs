// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic host time and timebase conversion.
//!
//! [`HostTime`] represents a point in time as platform-native monotonic ticks
//! (e.g. `mach_absolute_time` on macOS, `QueryPerformanceCounter` on Windows).
//!
//! [`Timebase`] carries the rational conversion factor from ticks to
//! nanoseconds, matching the pattern used by `mach_timebase_info` (numer/denom
//! converts ticks → nanoseconds).
//!
//! [`Duration`] represents a duration in the same tick units as [`HostTime`].
//! Conversion arithmetic uses `u128` intermediates to avoid overflow. Time and
//! duration operators saturate instead of panicking or wrapping.

use core::fmt;
use core::ops::{Add, Sub};

/// A point in time expressed as platform-native monotonic ticks.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HostTime(pub u64);

impl HostTime {
    /// Returns the raw tick value.
    #[inline]
    #[must_use]
    pub const fn ticks(self) -> u64 {
        self.0
    }

    /// Converts this host time to nanoseconds using the given timebase.
    ///
    /// Uses `u128` intermediate arithmetic to avoid overflow.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u128 intermediate avoids overflow; truncation back to u64 is intentional"
    )]
    pub const fn to_nanos(self, timebase: Timebase) -> u64 {
        let wide = self.0 as u128 * timebase.numer as u128 / timebase.denom as u128;
        wide as u64
    }

    /// Creates a [`HostTime`] from a nanosecond value and timebase.
    ///
    /// This is the inverse of [`to_nanos`](Self::to_nanos).
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u128 intermediate avoids overflow; truncation back to u64 is intentional"
    )]
    pub const fn from_nanos(nanos: u64, timebase: Timebase) -> Self {
        let wide = nanos as u128 * timebase.denom as u128 / timebase.numer as u128;
        Self(wide as u64)
    }

    /// Returns the duration between `self` and an earlier time, or zero if
    /// `earlier` is after `self`.
    #[inline]
    #[must_use]
    pub const fn saturating_duration_since(self, earlier: Self) -> Duration {
        Duration(self.0.saturating_sub(earlier.0))
    }

    /// Checked addition of a duration.
    #[inline]
    #[must_use]
    pub const fn checked_add(self, duration: Duration) -> Option<Self> {
        match self.0.checked_add(duration.0) {
            Some(t) => Some(Self(t)),
            None => None,
        }
    }

    /// Checked subtraction of a duration.
    #[inline]
    #[must_use]
    pub const fn checked_sub(self, duration: Duration) -> Option<Self> {
        match self.0.checked_sub(duration.0) {
            Some(t) => Some(Self(t)),
            None => None,
        }
    }
}

impl Add<Duration> for HostTime {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Duration) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub<Duration> for HostTime {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Duration) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Sub for HostTime {
    type Output = Duration;

    #[inline]
    fn sub(self, rhs: Self) -> Duration {
        Duration(self.0.saturating_sub(rhs.0))
    }
}

impl fmt::Debug for HostTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HostTime({})", self.0)
    }
}

/// Rational conversion factor from ticks to nanoseconds.
///
/// `nanoseconds = ticks * numer / denom`
///
/// This matches the `mach_timebase_info` pattern on macOS. The correct
/// instance for a given platform is provided by the platform adapter's
/// `timebase()` function.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timebase {
    /// Numerator of the ticks-to-nanoseconds ratio.
    pub numer: u32,
    /// Denominator of the ticks-to-nanoseconds ratio.
    pub denom: u32,
}

impl Timebase {
    /// A timebase where ticks are already nanoseconds (1:1).
    pub const NANOS: Self = Self { numer: 1, denom: 1 };

    /// Creates a new timebase with the given numerator and denominator.
    ///
    /// # Panics
    ///
    /// Panics if `denom` is zero.
    #[inline]
    #[must_use]
    pub const fn new(numer: u32, denom: u32) -> Self {
        assert!(denom != 0, "timebase denominator must not be zero");
        Self { numer, denom }
    }

    /// Converts a tick count to nanoseconds.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u128 intermediate avoids overflow; truncation back to u64 is intentional"
    )]
    pub const fn ticks_to_nanos(self, ticks: u64) -> u64 {
        let wide = ticks as u128 * self.numer as u128 / self.denom as u128;
        wide as u64
    }

    /// Converts nanoseconds to a tick count.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u128 intermediate avoids overflow; truncation back to u64 is intentional"
    )]
    pub const fn nanos_to_ticks(self, nanos: u64) -> u64 {
        let wide = nanos as u128 * self.denom as u128 / self.numer as u128;
        wide as u64
    }
}

impl fmt::Debug for Timebase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timebase({}/{})", self.numer, self.denom)
    }
}

/// A duration in platform-native ticks.
///
/// Arithmetic uses the same tick units as [`HostTime`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Duration(pub u64);

impl Duration {
    /// A zero-length duration.
    pub const ZERO: Self = Self(0);

    /// Returns the raw tick value.
    #[inline]
    #[must_use]
    pub const fn ticks(self) -> u64 {
        self.0
    }

    /// Returns whether this duration is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Converts this duration to nanoseconds using the given timebase.
    #[inline]
    #[must_use]
    pub const fn to_nanos(self, timebase: Timebase) -> u64 {
        timebase.ticks_to_nanos(self.0)
    }

    /// Creates a duration from a nanosecond value and timebase.
    #[inline]
    #[must_use]
    pub const fn from_nanos(nanos: u64, timebase: Timebase) -> Self {
        Self(timebase.nanos_to_ticks(nanos))
    }

    /// Saturating addition.
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Saturating integer multiplication.
    #[inline]
    #[must_use]
    pub const fn saturating_mul(self, rhs: u64) -> Self {
        Self(self.0.saturating_mul(rhs))
    }

    /// Integer division, returning zero when `rhs` is zero.
    #[inline]
    #[must_use]
    pub const fn div_u64(self, rhs: u64) -> Self {
        match self.0.checked_div(rhs) {
            Some(value) => Self(value),
            None => Self::ZERO,
        }
    }
}

impl Add for Duration {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}

impl Sub for Duration {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}

impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duration({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nanos_round_trip_identity_timebase() {
        let tb = Timebase::NANOS;
        let t = HostTime(1_000_000_000);
        assert_eq!(t.to_nanos(tb), 1_000_000_000, "identity timebase");
        assert_eq!(HostTime::from_nanos(1_000_000_000, tb), t);
    }

    #[test]
    fn nanos_round_trip_macos_style() {
        // Typical ARM Mac: 125/3 (ticks run at 24 MHz)
        let tb = Timebase::new(125, 3);
        let ticks = 24_000_000_u64; // 1 second worth of ticks
        let nanos = HostTime(ticks).to_nanos(tb);
        assert_eq!(nanos, 1_000_000_000, "24 MHz → 1s");

        let back = HostTime::from_nanos(nanos, tb);
        assert_eq!(back.ticks(), ticks);
    }

    #[test]
    fn overflow_safe_conversion() {
        // Large tick value that would overflow u64 if multiplied naively
        let tb = Timebase::new(125, 3);
        let t = HostTime(u64::MAX / 2);
        // Should not panic; result is approximate but deterministic
        let _nanos = t.to_nanos(tb);
    }

    #[test]
    fn duration_arithmetic() {
        let a = Duration(100);
        let b = Duration(30);
        assert_eq!((a + b).ticks(), 130);
        assert_eq!((a - b).ticks(), 70);
        assert_eq!(a.saturating_sub(Duration(200)), Duration::ZERO);
        assert_eq!((Duration(u64::MAX) + Duration(1)).ticks(), u64::MAX);
        assert_eq!((Duration(0) - Duration(1)).ticks(), 0);
    }

    #[test]
    fn host_time_duration_ops() {
        let t = HostTime(1000);
        let d = Duration(200);
        assert_eq!((t + d).ticks(), 1200);
        assert_eq!((t - d).ticks(), 800);
        assert_eq!((HostTime(u64::MAX) + Duration(1)).ticks(), u64::MAX);
        assert_eq!((HostTime(0) - Duration(1)).ticks(), 0);
        assert_eq!((HostTime(0) - HostTime(1)).ticks(), 0);
        assert_eq!(t.saturating_duration_since(HostTime(1500)), Duration::ZERO);
        assert_eq!(t.saturating_duration_since(HostTime(400)), Duration(600));
    }
}
