// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `CVDisplayLink` integration for predictive frame timing.

use alloc::boxed::Box;
use core::ffi::c_void;
use core::fmt;
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};

use frameclock::time::Timebase;
use frameclock::{FrameTick, HostTime, OutputId};
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
pub struct DisplayLink {
    raw: CFRetained<CVDisplayLinkRaw>,
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
    /// Returns the Mach absolute time timebase.
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

        let mut link_ptr: *mut CVDisplayLinkRaw = core::ptr::null_mut();
        let ret = unsafe {
            CVDisplayLinkRaw::create_with_active_cg_displays(NonNull::new_unchecked(&mut link_ptr))
        };
        if ret != kCVReturnSuccess {
            return Err(DisplayLinkError::CreateFailed(ret));
        }
        let raw_nn = NonNull::new(link_ptr).ok_or(DisplayLinkError::CreateFailed(ret))?;
        let raw = unsafe { CFRetained::from_raw(raw_nn) };

        let state_ptr: *const CallbackState = &*state;
        let ret = unsafe {
            raw.set_output_callback(Some(display_link_callback), state_ptr as *mut c_void)
        };
        if ret != kCVReturnSuccess {
            return Err(DisplayLinkError::CallbackFailed(ret));
        }

        Ok(Self { raw, _state: state })
    }

    /// Starts the display link.
    ///
    /// # Errors
    ///
    /// Returns [`DisplayLinkError::StartFailed`] if `CoreVideo` reports an error.
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
    /// Returns [`DisplayLinkError::StopFailed`] if `CoreVideo` reports an error.
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
        if self.raw.is_running() {
            let _ = self.raw.stop();
        }
    }
}

unsafe extern "C-unwind" fn display_link_callback(
    _display_link: NonNull<CVDisplayLinkRaw>,
    in_now: NonNull<CVTimeStamp>,
    in_output_time: NonNull<CVTimeStamp>,
    _flags_in: u64,
    _flags_out: NonNull<u64>,
    user_info: *mut c_void,
) -> i32 {
    let state = unsafe { &*(user_info.cast::<CallbackState>()) };

    let now_ts = unsafe { in_now.as_ref() };
    let out_ts = unsafe { in_output_time.as_ref() };

    let now = HostTime(now_ts.hostTime);
    let predicted_present = HostTime(out_ts.hostTime);
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
        frame_index,
        output: state.output,
        prev_actual_present: None,
    };

    state.sender.send(tick);

    kCVReturnSuccess
}
