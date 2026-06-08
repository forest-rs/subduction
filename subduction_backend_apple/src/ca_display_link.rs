// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `CADisplayLink` integration for predictive frame timing.
//!
//! Available on macOS 14+ and iOS 15+. This is the modern replacement for
//! `CVDisplayLink`, running callbacks on the main thread's run loop without
//! requiring cross-thread dispatch.
//!
//! # Frame loop
//!
//! ```text
//! CADisplayLink callback (main thread run loop)
//!   → user's callback(FrameTick)
//!     → scheduler.plan() → animate → store.evaluate()
//!       → presenter.apply()
//! ```

use alloc::boxed::Box;
use core::cell::Cell;
use core::fmt;

use frameclock::{FrameTick, HostTime, OutputId, Timebase, TimingConfidence};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_foundation::{NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSRunLoop};
use objc2_quartz_core::CADisplayLink as CADisplayLinkRaw;

use crate::mach_time;

struct DisplayLinkTargetIvars {
    callback: Box<dyn Fn(FrameTick)>,
    frame_counter: Cell<u64>,
    output: OutputId,
    timebase: Timebase,
}

define_class! {
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SubductionDisplayLinkTarget"]
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
        // SAFETY: NSObject's init is always safe.
        unsafe { msg_send![super(this), init] }
    }

    fn handle_tick(&self, sender: &AnyObject) {
        let ivars = self.ivars();

        // Read timing from the CADisplayLink.
        // SAFETY: `sender` is the CADisplayLink that invoked this selector.
        let target_ts: f64 = unsafe { msg_send![sender, targetTimestamp] };
        let duration: f64 = unsafe { msg_send![sender, duration] };
        let timestamp: f64 = unsafe { msg_send![sender, timestamp] };

        let now = mach_time::now();
        let predicted_present = HostTime(mach_time::seconds_to_ticks(target_ts, ivars.timebase));
        let refresh_interval = mach_time::seconds_to_ticks(duration, ivars.timebase);

        let frame_index = ivars.frame_counter.get();
        ivars.frame_counter.set(frame_index + 1);

        // `timestamp` is the actual display time of the previous frame.
        // On the first callback it has no meaningful previous-frame data.
        let prev_actual_present = if frame_index > 0 {
            Some(HostTime(mach_time::seconds_to_ticks(
                timestamp,
                ivars.timebase,
            )))
        } else {
            None
        };

        let tick = FrameTick {
            now,
            predicted_present: Some(predicted_present),
            refresh_interval: Some(refresh_interval),
            confidence: TimingConfidence::Predictive,
            frame_index,
            output: ivars.output,
            prev_actual_present,
        };

        (ivars.callback)(tick);
    }
}

/// Safe wrapper around `CADisplayLink` that produces [`FrameTick`] events on
/// the main thread.
///
/// Unlike the `CVDisplayLink`-based implementation, `CADisplayLink` fires
/// directly on the main thread's run loop. No cross-thread dispatch is needed,
/// and the callback can be a plain `Fn` (not `Send + Sync`).
///
/// Requires [`MainThreadMarker`] at construction time.
///
/// # Example
///
/// ```ignore
/// let link = DisplayLink::new(|tick| { /* handle tick */ }, OutputId(0), mtm);
/// link.start();
/// ```
pub struct DisplayLink {
    raw: Retained<CADisplayLinkRaw>,
    // Prevent the target from being deallocated while the link is alive.
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
    /// Returns the Mach absolute time timebase (numer/denom → nanoseconds).
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
    /// The link is created but **not started**. Call [`start`](Self::start) to
    /// begin receiving ticks on the main run loop.
    ///
    /// The callback is invoked on the main thread for each display refresh.
    pub fn new<F>(callback: F, output: OutputId, mtm: MainThreadMarker) -> Self
    where
        F: Fn(FrameTick) + 'static,
    {
        let target = DisplayLinkTarget::new(callback, output, mtm);

        // SAFETY: `target` is a valid NSObject and `sel!(tick:)` matches the
        // method defined in define_class! above. The target is kept alive by
        // `_target` for the lifetime of this DisplayLink.
        let raw = unsafe {
            CADisplayLinkRaw::displayLinkWithTarget_selector(
                // Cast to &AnyObject for the target parameter.
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
        // SAFETY: We're adding to the main run loop which is always valid.
        // `NSDefaultRunLoopMode` is the standard mode for event processing.
        unsafe {
            self.raw
                .addToRunLoop_forMode(&run_loop, NSDefaultRunLoopMode);
        }
    }

    /// Stops the display link by removing it from the main run loop.
    pub fn stop(&self) {
        let run_loop = NSRunLoop::mainRunLoop();
        // SAFETY: Removing from the main run loop is safe.
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

    /// Pauses or resumes the display link.
    pub fn set_paused(&self, paused: bool) {
        self.raw.setPaused(paused);
    }
}

impl Drop for DisplayLink {
    fn drop(&mut self) {
        self.raw.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use crate::mach_time;

    #[test]
    fn seconds_to_ticks_accuracy() {
        let tb = mach_time::timebase();
        // 1/60th of a second (16.667ms)
        let ticks = mach_time::seconds_to_ticks(1.0 / 60.0, tb);
        let nanos = tb.ticks_to_nanos(ticks);
        // Should be approximately 16_666_666 ns
        let expected = 16_666_666_u64;
        let error = (nanos as i64 - expected as i64).unsigned_abs();
        assert!(
            error < 100,
            "1/60s conversion error too large: {error} ns (got {nanos}, expected ~{expected})"
        );
    }
}
