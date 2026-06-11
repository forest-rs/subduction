// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mach absolute time utilities shared across display-link implementations.

use frameclock::HostTime;
use frameclock::time::Timebase;

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

pub(crate) fn timebase() -> Timebase {
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    unsafe { mach_timebase_info(&mut info) };
    Timebase::new(info.numer, info.denom)
}

pub(crate) fn now() -> HostTime {
    HostTime(unsafe { mach_absolute_time() })
}

#[cfg_attr(
    not(feature = "ca-display-link"),
    allow(dead_code, reason = "used only by ca_display_link module")
)]
pub(crate) fn seconds_to_ticks(seconds: f64, tb: Timebase) -> u64 {
    let nanos = seconds * 1_000_000_000.0;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 to u64 truncation is intentional for positive host-time values"
    )]
    {
        (nanos * (tb.denom as f64) / (tb.numer as f64)) as u64
    }
}
