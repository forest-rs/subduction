// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `CVDisplayLink` integration for predictive frame timing.
//!
//! Wraps `CVDisplayLink` in a safe Rust API that produces [`FrameTick`] events
//! with [`TimingConfidence::Predictive`].
//!
//! [`FrameTick`]: frameclock::FrameTick
//! [`TimingConfidence::Predictive`]: frameclock::TimingConfidence::Predictive

use alloc::boxed::Box;
use core::ffi::c_void;
use core::fmt;
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};

use frameclock::{FrameTick, HostTime, OutputId, Timebase, TimingConfidence};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{CVDisplayLink as CVDisplayLinkRaw, CVTimeStamp, kCVReturnSuccess};

use crate::threading::TickSender;

/// Errors from [`DisplayLink`] operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayLinkError {
    /// `CVDisplayLinkCreateWithActiveCGDisplays` failed.
    CreateFailed(i32),
    /// `CVDisplayLinkSetOutputCallback` failed.
    CallbackFailed(i32),
    /// `CVDisplayLinkStart` failed.
    StartFailed(i32),
    /// `CVDisplayLinkStop` failed.
    StopFailed(i32),
}

impl fmt::Display for DisplayLinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateFailed(code) => write!(f, "CVDisplayLink creation failed ({code})"),
            Self::CallbackFailed(code) => {
                write!(f, "CVDisplayLink set callback failed ({code})")
            }
            Self::StartFailed(code) => write!(f, "CVDisplayLink start failed ({code})"),
            Self::StopFailed(code) => write!(f, "CVDisplayLink stop failed ({code})"),
        }
    }
}

impl core::error::Error for DisplayLinkError {}

struct CallbackState {
    sender: TickSender,
    frame_counter: AtomicU64,
    output: OutputId,
}

/// Safe wrapper around `CVDisplayLink` that produces [`FrameTick`] events.
///
/// `DisplayLink` is `!Send` because the underlying `CVDisplayLink` is not
/// thread-safe for mutation. The callback itself runs on a `CoreVideo`
/// background thread by design — it uses only atomics and the `Send + Sync`
/// [`TickSender`].
///
/// # Example
///
/// ```ignore
/// let forwarder = TickForwarder::new(|tick| { /* handle on main thread */ });
/// let link = DisplayLink::new(forwarder.sender(), OutputId(0))?;
/// link.start()?;
/// ```
pub struct DisplayLink {
    /// Retained reference to the underlying `CVDisplayLink`.
    /// `CFRetained` handles release on drop.
    raw: CFRetained<CVDisplayLinkRaw>,
    /// Pinned callback state whose pointer is shared with the C callback.
    /// Must outlive `raw` (declared after, dropped after).
    _state: Pin<Box<CallbackState>>,
}

impl fmt::Debug for DisplayLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisplayLink")
            .field("output", &self._state.output)
            .finish_non_exhaustive()
    }
}

impl DisplayLink {
    /// Returns the Mach absolute time timebase (numer/denom → nanoseconds).
    #[must_use]
    pub fn timebase() -> Timebase {
        crate::mach_time::timebase()
    }

    /// Returns the current Mach absolute time as a [`HostTime`].
    #[must_use]
    pub fn now() -> HostTime {
        crate::mach_time::now()
    }

    /// Creates a new display link targeting all active displays.
    ///
    /// The link is created but **not started**. Call [`start`](Self::start) to
    /// begin receiving ticks.
    ///
    /// # Errors
    ///
    /// Returns [`DisplayLinkError`] if the underlying `CoreVideo` calls fail.
    #[expect(
        deprecated,
        reason = "CVDisplayLink API is deprecated by Apple but still functional"
    )]
    pub fn new(sender: TickSender, output: OutputId) -> Result<Self, DisplayLinkError> {
        let state = Box::pin(CallbackState {
            sender,
            frame_counter: AtomicU64::new(0),
            output,
        });

        // Create the CVDisplayLink.
        let mut link_ptr: *mut CVDisplayLinkRaw = core::ptr::null_mut();
        // SAFETY: link_ptr is a valid out-pointer.
        let ret = unsafe {
            CVDisplayLinkRaw::create_with_active_cg_displays(NonNull::new_unchecked(&mut link_ptr))
        };
        if ret != kCVReturnSuccess {
            return Err(DisplayLinkError::CreateFailed(ret));
        }
        let raw_nn = NonNull::new(link_ptr).ok_or(DisplayLinkError::CreateFailed(ret))?;
        // SAFETY: create_with_active_cg_displays follows the Create Rule,
        // returning a +1 retained reference.
        let raw = unsafe { CFRetained::from_raw(raw_nn) };

        // Set the output callback, passing a pointer to the pinned state.
        let state_ptr: *const CallbackState = &*state;
        // SAFETY: display_link_callback matches the expected C signature,
        // and state_ptr will remain valid as long as this DisplayLink exists.
        let ret = unsafe {
            raw.set_output_callback(Some(display_link_callback), state_ptr as *mut c_void)
        };
        if ret != kCVReturnSuccess {
            return Err(DisplayLinkError::CallbackFailed(ret));
        }

        Ok(Self { raw, _state: state })
    }

    /// Starts the display link. Ticks will begin arriving on the main thread
    /// via the [`TickSender`].
    ///
    /// # Errors
    ///
    /// Returns [`DisplayLinkError::StartFailed`] if already running or if
    /// `CoreVideo` reports an error.
    #[expect(
        deprecated,
        reason = "CVDisplayLink API is deprecated by Apple but still functional"
    )]
    pub fn start(&self) -> Result<(), DisplayLinkError> {
        let ret = self.raw.start();
        if ret != kCVReturnSuccess {
            return Err(DisplayLinkError::StartFailed(ret));
        }
        Ok(())
    }

    /// Stops the display link.
    ///
    /// # Errors
    ///
    /// Returns [`DisplayLinkError::StopFailed`] if not running or if
    /// `CoreVideo` reports an error.
    #[expect(
        deprecated,
        reason = "CVDisplayLink API is deprecated by Apple but still functional"
    )]
    pub fn stop(&self) -> Result<(), DisplayLinkError> {
        let ret = self.raw.stop();
        if ret != kCVReturnSuccess {
            return Err(DisplayLinkError::StopFailed(ret));
        }
        Ok(())
    }
}

impl Drop for DisplayLink {
    #[expect(
        deprecated,
        reason = "CVDisplayLink API is deprecated by Apple but still functional"
    )]
    fn drop(&mut self) {
        // Stop if running (ignore errors during drop).
        // CFRetained handles the release automatically.
        if self.raw.is_running() {
            let _ = self.raw.stop();
        }
    }
}

/// The C callback invoked by `CoreVideo` on its background thread.
///
/// # Safety
///
/// - `user_info` must point to a valid, pinned `CallbackState`.
/// - `in_now` and `in_output_time` must be valid `CVTimeStamp` pointers
///   (guaranteed by `CoreVideo`).
unsafe extern "C-unwind" fn display_link_callback(
    _display_link: NonNull<CVDisplayLinkRaw>,
    in_now: NonNull<CVTimeStamp>,
    in_output_time: NonNull<CVTimeStamp>,
    _flags_in: u64,
    _flags_out: NonNull<u64>,
    user_info: *mut c_void,
) -> i32 {
    // SAFETY: user_info is the pinned CallbackState pointer we set in `new`.
    let state = unsafe { &*(user_info.cast::<CallbackState>()) };

    let now_ts = unsafe { in_now.as_ref() };
    let out_ts = unsafe { in_output_time.as_ref() };

    let now = HostTime(now_ts.hostTime);
    let predicted_present = HostTime(out_ts.hostTime);

    // Compute refresh interval from the difference between output and now
    // timestamps' hostTime values.
    let refresh_interval = if out_ts.hostTime > now_ts.hostTime {
        Some(out_ts.hostTime - now_ts.hostTime)
    } else {
        None
    };

    let frame_index = state.frame_counter.fetch_add(1, Ordering::Relaxed);

    let tick = FrameTick {
        now,
        predicted_present: Some(predicted_present),
        refresh_interval,
        confidence: TimingConfidence::Predictive,
        frame_index,
        output: state.output,
        prev_actual_present: None,
    };

    state.sender.send(tick);

    kCVReturnSuccess
}
