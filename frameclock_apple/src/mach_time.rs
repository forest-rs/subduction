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
        (nanos * f64::from(tb.denom) / f64::from(tb.numer)) as u64
    }
}

/// Projects a Core Animation media timestamp into the Mach host-time domain.
///
/// The display-link timestamp and `CACurrentMediaTime()` are sampled in Core
/// Animation media seconds. This preserves their relative delta and applies it
/// to a Mach host-time sample taken in the same callback.
/// This avoids assuming Core Animation media seconds and Mach ticks share an
/// absolute epoch.
#[cfg(any(feature = "ca-display-link", test))]
pub(crate) fn media_time_to_host_time(
    media_time_seconds: f64,
    host_now: HostTime,
    ca_now_seconds: f64,
    tb: Timebase,
) -> Option<HostTime> {
    if !media_time_seconds.is_finite() || !ca_now_seconds.is_finite() {
        return None;
    }

    let delta_seconds = media_time_seconds - ca_now_seconds;
    if !delta_seconds.is_finite() {
        return None;
    }
    let delta = frameclock::Duration(seconds_to_ticks(delta_seconds.abs(), tb));

    Some(if delta_seconds >= 0.0 {
        host_now + delta
    } else {
        host_now - delta
    })
}

#[cfg(test)]
mod tests {
    use super::media_time_to_host_time;
    use frameclock::HostTime;
    use frameclock::time::Timebase;

    const NANOS: Timebase = Timebase::NANOS;
    const MACOS_STYLE: Timebase = Timebase::new(125, 3);

    #[test]
    fn relative_media_time_maps_present_future_and_past_timestamps() {
        let host_now = HostTime(1_000);

        assert_eq!(
            media_time_to_host_time(12.0, host_now, 12.0, NANOS),
            Some(host_now)
        );
        assert_eq!(
            media_time_to_host_time(12.25, host_now, 12.0, NANOS),
            Some(HostTime(250_001_000))
        );
        assert_eq!(
            media_time_to_host_time(
                11.75,
                host_now + frameclock::Duration(250_000_000),
                12.0,
                NANOS
            ),
            Some(host_now)
        );
    }

    #[test]
    fn relative_media_time_saturates_at_host_time_bounds() {
        let host_now = HostTime(1_000);

        assert_eq!(
            media_time_to_host_time(10.0, host_now, 12.0, NANOS),
            Some(HostTime(0))
        );
        assert_eq!(
            media_time_to_host_time(12.0, HostTime(u64::MAX - 10), 11.0, NANOS),
            Some(HostTime(u64::MAX))
        );
    }

    #[test]
    fn relative_media_time_rejects_non_finite_input() {
        let host_now = HostTime(1_000);

        assert_eq!(
            media_time_to_host_time(f64::NAN, host_now, 12.0, NANOS),
            None
        );
        assert_eq!(
            media_time_to_host_time(12.0, host_now, f64::INFINITY, NANOS),
            None
        );
        assert_eq!(
            media_time_to_host_time(f64::MAX, host_now, -f64::MAX, NANOS),
            None
        );
    }

    #[test]
    fn relative_media_time_uses_supplied_timebase() {
        let host_now = HostTime(24_000_000);

        let mapped = media_time_to_host_time(13.0, host_now, 12.0, MACOS_STYLE);

        assert_eq!(mapped, Some(HostTime(48_000_000)));
    }
}
