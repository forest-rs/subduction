// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! QPC-based timing functions for [`HostTime`] / [`Timebase`].

use std::sync::OnceLock;

use frameclock::HostTime;
use frameclock::time::Timebase;
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

/// Cached QPC frequency — constant for the lifetime of the system.
static QPC_FREQ: OnceLock<i64> = OnceLock::new();

fn cached_frequency() -> i64 {
    *QPC_FREQ.get_or_init(|| {
        let mut freq = 0_i64;
        unsafe { QueryPerformanceFrequency(&mut freq).unwrap() };
        freq
    })
}

/// Current monotonic time as a [`HostTime`] (raw QPC ticks).
#[must_use]
#[expect(clippy::cast_sign_loss, reason = "QPC values are always non-negative")]
pub fn now() -> HostTime {
    let mut count = 0_i64;
    unsafe { QueryPerformanceCounter(&mut count).unwrap() };
    HostTime(count as u64)
}

/// Conversion factor from QPC ticks to nanoseconds.
///
/// `nanos = ticks * numer / denom`
/// → `numer = 1_000_000_000`, `denom = QPC frequency`
#[must_use]
pub fn timebase() -> Timebase {
    let freq = cached_frequency();
    debug_assert!(
        freq > 0 && freq <= i64::from(u32::MAX),
        "QPC frequency {freq} out of u32 range"
    );
    Timebase {
        numer: 1_000_000_000,
        #[expect(
            clippy::cast_possible_truncation,
            reason = "QPC frequency fits in u32 on all known hardware (typically 10 MHz)"
        )]
        denom: freq as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::{now, timebase};

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
}
