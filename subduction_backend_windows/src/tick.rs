// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `VSync`-paced tick sources that drive frames via `PostMessageW`.
//!
//! The tick thread calls `DwmFlush()` (or waits on a frame latency
//! waitable) to pace itself to the DWM compositor `VSync`, then posts
//! [`WM_APP_TICK`] to the window's message queue. Because the message
//! is *posted* (not sent), Windows' internal modal loops for move and
//! resize dispatch it — animations continue uninterrupted during window
//! drag.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use subduction_core::output::OutputId;
use subduction_core::time::HostTime;
use subduction_core::timing::{FrameTick, PresentHints, TimingConfidence};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Dwm::DwmFlush;
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::Win32::UI::WindowsAndMessaging::{PostMessageW, WM_APP};

/// Custom window message posted by the tick thread on each `VSync`.
///
/// Handle this in your `wnd_proc` to drive one frame per `VSync`.
/// Call [`make_tick`] inside the handler to construct the [`FrameTick`].
pub const WM_APP_TICK: u32 = WM_APP + 1;

/// Tick source paced by `DwmFlush()`.
///
/// Spawns a thread that blocks on `DwmFlush()` and posts [`WM_APP_TICK`]
/// to the provided HWND. Dropping this struct stops the thread.
pub struct TickSource {
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for TickSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TickSource")
            .field("running", &self.running.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl TickSource {
    /// Start the tick source thread for `hwnd`.
    pub fn start(hwnd: HWND) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        // Cast the pointer to usize so it is `Send + Copy` across the
        // thread boundary. `PostMessageW` is safe from non-GUI threads.
        let hwnd_bits = hwnd.0 as usize;

        let handle = std::thread::Builder::new()
            .name("SubductionTick".into())
            .spawn(move || {
                let hwnd = HWND(hwnd_bits as *mut core::ffi::c_void);
                Self::thread_main(hwnd, &running_clone);
            })
            .expect("failed to spawn tick source thread");

        Self {
            running,
            handle: Some(handle),
        }
    }

    fn thread_main(hwnd: HWND, running: &AtomicBool) {
        while running.load(Ordering::Relaxed) {
            if unsafe { DwmFlush() }.is_err() {
                std::thread::sleep(Duration::from_millis(16));
            }
            unsafe {
                // Failure means the hwnd is gone or the queue is full — either
                // way the tick is harmlessly skipped; the next VSync retries.
                let _ = PostMessageW(Some(hwnd), WM_APP_TICK, WPARAM(0), LPARAM(0));
            }
        }
    }

    /// Stop the tick source and join the thread.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // Thread panic during shutdown — nothing to recover; propagating
            // from Drop would abort.
            let _ = h.join();
        }
    }
}

impl Drop for TickSource {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Tick source paced by a swapchain frame latency waitable.
///
/// Uses `GetFrameLatencyWaitableObject` (from the application's
/// swapchain) instead of `DwmFlush()`. This provides per-swapchain
/// pacing rather than global DWM sync, enabling tighter frame timing.
///
/// Falls back to `DwmFlush()` if the waitable times out (e.g. swapchain
/// stalled or destroyed).
pub struct FrameEventTickSource {
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for FrameEventTickSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameEventTickSource")
            .field("running", &self.running.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl FrameEventTickSource {
    /// Start the tick source thread for `hwnd`, paced by `frame_event`
    /// (`IDXGISwapChain2::GetFrameLatencyWaitableObject`).
    pub fn start(hwnd: HWND, frame_event: HANDLE) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let hwnd_bits = hwnd.0 as usize;
        let event_bits = frame_event.0 as usize;

        let handle = std::thread::Builder::new()
            .name("SubductionTick".into())
            .spawn(move || {
                let hwnd = HWND(hwnd_bits as *mut core::ffi::c_void);
                let event = HANDLE(event_bits as *mut core::ffi::c_void);
                Self::thread_main(hwnd, event, &running_clone);
            })
            .expect("failed to spawn tick source thread");

        Self {
            running,
            handle: Some(handle),
        }
    }

    fn thread_main(hwnd: HWND, frame_event: HANDLE, running: &AtomicBool) {
        while running.load(Ordering::Relaxed) {
            let result = unsafe { WaitForSingleObject(frame_event, 32) };
            if result != WAIT_OBJECT_0 && unsafe { DwmFlush() }.is_err() {
                std::thread::sleep(Duration::from_millis(16));
            }
            unsafe {
                // Failure means the hwnd is gone or the queue is full — either
                // way the tick is harmlessly skipped; the next VSync retries.
                let _ = PostMessageW(Some(hwnd), WM_APP_TICK, WPARAM(0), LPARAM(0));
            }
        }
    }

    /// Stop the tick source and join the thread.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // Thread panic during shutdown — nothing to recover; propagating
            // from Drop would abort.
            let _ = h.join();
        }
    }
}

impl Drop for FrameEventTickSource {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Build a [`FrameTick`] from QPC. Call inside the `WM_APP_TICK` handler.
#[must_use]
pub fn make_tick(
    refresh_interval_ns: u64,
    frame_index: u64,
    prev_present_time: Option<HostTime>,
) -> FrameTick {
    let timebase = crate::timing::timebase();
    let interval_ticks = if refresh_interval_ns > 0 {
        refresh_interval_ns * u64::from(timebase.denom) / u64::from(timebase.numer)
    } else {
        0
    };

    let now = crate::timing::now();

    let predicted_present = if interval_ticks > 0 {
        if let Some(prev) = prev_present_time {
            Some(HostTime(prev.ticks() + interval_ticks))
        } else {
            Some(HostTime(now.ticks() + interval_ticks))
        }
    } else {
        None
    };

    FrameTick {
        now,
        predicted_present,
        refresh_interval: if refresh_interval_ns > 0 {
            Some(refresh_interval_ns)
        } else {
            None
        },
        confidence: TimingConfidence::Estimated,
        frame_index,
        output: OutputId(0),
        prev_actual_present: prev_present_time,
    }
}

/// Compute presentation hints from a tick and safety margin (nanoseconds).
#[must_use]
pub fn compute_hints(tick: &FrameTick, safety_margin_ns: u64) -> PresentHints {
    let timebase = crate::timing::timebase();
    let margin_ticks = safety_margin_ns * u64::from(timebase.denom) / u64::from(timebase.numer);

    PresentHints {
        desired_present: tick.predicted_present,
        latest_commit: HostTime(
            tick.predicted_present
                .map(|p| p.ticks().saturating_sub(margin_ticks))
                .unwrap_or(tick.now.ticks()),
        ),
    }
}
