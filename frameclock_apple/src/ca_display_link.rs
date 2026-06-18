// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `CADisplayLink` integration for predictive frame timing.

use alloc::boxed::Box;
use core::cell::Cell;
use core::fmt;

use frameclock::time::Timebase;
use frameclock::{DisplayTiming, Duration, FrameTick, HostTime, OutputId};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_foundation::{NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSRunLoop};
use objc2_quartz_core::{CACurrentMediaTime, CADisplayLink as CADisplayLinkRaw, CAFrameRateRange};

use crate::{PreferredFrameRateRange, mach_time, preferred_frame_rate_range};

struct DisplayLinkTargetIvars {
    callback: Box<dyn Fn(FrameTick)>,
    frame_counter: Cell<u64>,
    output: OutputId,
    timebase: Timebase,
}

define_class! {
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "FrameclockDisplayLinkTarget"]
    #[ivars = DisplayLinkTargetIvars]
    struct DisplayLinkTarget;

    unsafe impl NSObjectProtocol for DisplayLinkTarget {}

    impl DisplayLinkTarget {
        #[unsafe(method(tick:))]
        fn tick(&self, sender: &AnyObject) {
            self.handle_tick(sender);
        }
    }
}

impl DisplayLinkTarget {
    fn new<F>(callback: F, output: OutputId, mtm: MainThreadMarker) -> Retained<Self>
    where
        F: Fn(FrameTick) + 'static,
    {
        let tb = mach_time::timebase();
        let this = mtm.alloc::<Self>().set_ivars(DisplayLinkTargetIvars {
            callback: Box::new(callback),
            frame_counter: Cell::new(0),
            output,
            timebase: tb,
        });
        unsafe { msg_send![super(this), init] }
    }

    fn handle_tick(&self, sender: &AnyObject) {
        let ivars = self.ivars();

        let target_ts: f64 = unsafe { msg_send![sender, targetTimestamp] };
        let duration: f64 = unsafe { msg_send![sender, duration] };
        let timestamp: f64 = unsafe { msg_send![sender, timestamp] };

        // These samples intentionally define one same-callback conversion pair
        // from Core Animation media time into the Mach host-time domain.
        let now = mach_time::now();
        let ca_now = CACurrentMediaTime();
        let predicted_present =
            mach_time::media_time_to_host_time(target_ts, now, ca_now, ivars.timebase);
        let refresh_interval = mach_time::seconds_to_ticks(duration, ivars.timebase);

        let frame_index = ivars.frame_counter.get();
        ivars.frame_counter.set(frame_index + 1);

        let prev_actual_present = if frame_index > 0 {
            mach_time::media_time_to_host_time(timestamp, now, ca_now, ivars.timebase)
        } else {
            None
        };

        let tick = FrameTick {
            now,
            predicted_present,
            refresh_interval: Some(refresh_interval),
            frame_index,
            output: ivars.output,
            prev_actual_present,
        };

        (ivars.callback)(tick);
    }
}

/// Safe wrapper around `CADisplayLink` that produces [`FrameTick`] events on
/// the main thread.
pub struct DisplayLink {
    raw: Retained<CADisplayLinkRaw>,
    _target: Retained<DisplayLinkTarget>,
}

impl fmt::Debug for DisplayLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisplayLink")
            .field("paused", &self.raw.isPaused())
            .finish_non_exhaustive()
    }
}

impl DisplayLink {
    /// Returns the Mach absolute time timebase.
    #[must_use]
    pub fn timebase() -> Timebase {
        mach_time::timebase()
    }

    /// Returns the current Mach absolute time as a [`HostTime`].
    #[must_use]
    pub fn now() -> HostTime {
        mach_time::now()
    }

    /// Creates a new display link with the given callback.
    ///
    /// The link is created but not started. Call [`start`](Self::start) to
    /// begin receiving ticks on the main run loop.
    pub fn new<F>(callback: F, output: OutputId, mtm: MainThreadMarker) -> Self
    where
        F: Fn(FrameTick) + 'static,
    {
        let target = DisplayLinkTarget::new(callback, output, mtm);

        let raw = unsafe {
            CADisplayLinkRaw::displayLinkWithTarget_selector(
                &*((&*target) as *const DisplayLinkTarget as *const AnyObject),
                sel!(tick:),
            )
        };

        Self {
            raw,
            _target: target,
        }
    }

    /// Starts the display link by adding it to the main run loop.
    pub fn start(&self) {
        let run_loop = NSRunLoop::mainRunLoop();
        unsafe {
            self.raw
                .addToRunLoop_forMode(&run_loop, NSDefaultRunLoopMode);
        }
    }

    /// Stops the display link by removing it from the main run loop.
    pub fn stop(&self) {
        let run_loop = NSRunLoop::mainRunLoop();
        unsafe {
            self.raw
                .removeFromRunLoop_forMode(&run_loop, NSDefaultRunLoopMode);
        }
    }

    /// Returns whether the display link is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.raw.isPaused()
    }

    /// Pauses or resumes the display link without removing it from the run loop.
    ///
    /// Hosts can pause when no frame demand is pending and resume when input,
    /// animation, or other visible work arrives.
    pub fn set_paused(&self, paused: bool) {
        self.raw.setPaused(paused);
    }

    /// Sets the Core Animation preferred frame-rate range.
    ///
    /// This is the native `ProMotion` writeback seam. Hosts typically compute the
    /// range from a ready frame's selected interval with
    /// [`preferred_frame_rate_range`](crate::preferred_frame_rate_range), apply
    /// it to the display link, and then submit/render normally.
    pub fn set_preferred_frame_rate_range(&self, range: PreferredFrameRateRange) {
        self.raw.setPreferredFrameRateRange(CAFrameRateRange::new(
            range.minimum,
            range.maximum,
            range.preferred,
        ));
    }

    /// Computes and applies a preferred frame-rate range for `frame_interval`.
    ///
    /// Returns `None` when the interval cannot be represented as a finite
    /// frames-per-second value.
    pub fn set_preferred_frame_interval(
        &self,
        frame_interval: Duration,
        display_timing: DisplayTiming,
    ) -> Option<PreferredFrameRateRange> {
        let range =
            preferred_frame_rate_range(frame_interval, display_timing, mach_time::timebase())?;
        self.set_preferred_frame_rate_range(range);
        Some(range)
    }
}

impl Drop for DisplayLink {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mach_timebase_available() {
        let tb = mach_time::timebase();
        assert!(tb.numer > 0, "timebase numerator must be non-zero");
        assert!(tb.denom > 0, "timebase denominator must be non-zero");
    }
}
