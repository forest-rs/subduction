// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mach absolute time utilities shared across display link implementations.

use frameclock::HostTime;
use frameclock::time::Timebase;

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

// SAFETY: These are stable macOS/iOS kernel ABI functions.
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

/// Returns the Mach absolute time timebase (numer/denom → nanoseconds).
pub(crate) fn timebase() -> Timebase {
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    // SAFETY: passing a valid pointer to a local struct.
    unsafe { mach_timebase_info(&mut info) };
    Timebase::new(info.numer, info.denom)
}

/// Returns the current Mach absolute time as a [`HostTime`].
pub(crate) fn now() -> HostTime {
    // SAFETY: mach_absolute_time is always safe to call.
    HostTime(unsafe { mach_absolute_time() })
}

/// Converts `CFTimeInterval` (seconds since boot) to Mach absolute ticks.
#[cfg_attr(
    not(feature = "ca-display-link"),
    allow(dead_code, reason = "used only by ca_display_link module")
)]
///
/// `CACurrentMediaTime()` and `CADisplayLink` timestamps are `CFTimeInterval`
/// (f64 seconds) derived from Mach absolute time. Converting back introduces
/// sub-nanosecond rounding error (f64 has ~15 significant digits), which is
/// negligible for frame timing.
pub(crate) fn seconds_to_ticks(seconds: f64, tb: Timebase) -> u64 {
    let nanos = seconds * 1_000_000_000.0;
    // ticks = nanos * denom / numer
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64→u64 truncation is intentional; values are always positive and in range"
    )]
    {
        (nanos * (tb.denom as f64) / (tb.numer as f64)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timebase_is_valid() {
        let tb = timebase();
        assert!(tb.numer > 0, "timebase numer must be non-zero");
        assert!(tb.denom > 0, "timebase denom must be non-zero");
    }

    #[test]
    fn now_returns_nonzero() {
        let t = now();
        assert!(t.ticks() > 0, "mach_absolute_time should be non-zero");
    }

    #[test]
    fn seconds_to_ticks_roundtrip() {
        let tb = timebase();
        // 1 second in ticks
        let one_sec_ticks = seconds_to_ticks(1.0, tb);
        // Convert back to nanos for sanity check
        let nanos = tb.ticks_to_nanos(one_sec_ticks);
        // Should be within 1 microsecond of 1 billion nanos
        let error = (nanos as i64 - 1_000_000_000_i64).unsigned_abs();
        assert!(
            error < 1_000,
            "1 second roundtrip error too large: {error} ns"
        );
    }
}
