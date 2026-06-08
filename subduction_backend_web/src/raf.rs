// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `requestAnimationFrame` tick source.
//!
//! [`RafLoop`] drives a [`FrameTick`]-based animation loop using the browser's
//! `requestAnimationFrame` API. Each callback receives a
//! [`DOMHighResTimeStamp`][mdn] (milliseconds from `performance.now()`),
//! which is converted to microsecond [`HostTime`] ticks.
//!
//! Timing confidence is [`PacingOnly`] — the browser provides frame pacing but
//! no predicted present time.
//!
//! [mdn]: https://developer.mozilla.org/en-US/docs/Web/API/DOMHighResTimeStamp
//! [`FrameTick`]: frameclock::FrameTick
//! [`HostTime`]: frameclock::HostTime
//! [`PacingOnly`]: frameclock::TimingConfidence::PacingOnly

use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::{Cell, RefCell};

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

use frameclock::{FrameTick, HostTime, OutputId, TimingConfidence};

// Direct global bindings instead of `web_sys::Window` methods — avoids
// fetching (and unwrapping) the Window/Performance objects on every frame.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = "now")]
    pub(crate) fn performance_now() -> f64;

    #[wasm_bindgen(js_name = "requestAnimationFrame")]
    fn request_animation_frame(callback: &JsValue) -> i32;

    #[wasm_bindgen(js_name = "cancelAnimationFrame")]
    fn cancel_animation_frame(id: i32);
}

/// A `requestAnimationFrame` animation loop that emits [`FrameTick`] events.
///
/// Create with [`RafLoop::new`], then call [`start`](Self::start) to begin
/// receiving callbacks. The loop re-registers itself each frame until
/// [`stop`](Self::stop) is called or the `RafLoop` is dropped.
///
/// [`FrameTick`]: frameclock::FrameTick
pub struct RafLoop {
    inner: Rc<RafInner>,
}

type RafClosure = Closure<dyn FnMut(f64)>;

struct RafInner {
    /// The JS closure registered with `requestAnimationFrame`.
    ///
    /// Stored in its own `RefCell` so we can set it once in `start()` and
    /// reference it from inside itself without conflicting with `callback`.
    closure: RefCell<Option<RafClosure>>,

    /// The user-supplied callback that receives [`FrameTick`] events.
    callback: RefCell<Box<dyn FnMut(FrameTick)>>,

    /// Monotonically increasing frame counter (becomes `FrameTick::frame_index`).
    frame_counter: Cell<u64>,

    /// The output identifier passed through to each [`FrameTick`].
    output: OutputId,

    /// Whether the loop is currently running.
    running: Cell<bool>,

    /// The ID returned by the most recent `requestAnimationFrame` call,
    /// used by [`cancel_animation_frame`] when stopping.
    raf_id: Cell<i32>,
}

impl RafLoop {
    /// Creates a new `RafLoop` that is **not yet running**.
    ///
    /// `callback` will receive a [`FrameTick`] on each animation frame once
    /// [`start`](Self::start) is called. `output` identifies the display
    /// surface for the ticks.
    ///
    /// [`FrameTick`]: frameclock::FrameTick
    pub fn new(callback: impl FnMut(FrameTick) + 'static, output: OutputId) -> Self {
        Self {
            inner: Rc::new(RafInner {
                closure: RefCell::new(None),
                callback: RefCell::new(Box::new(callback)),
                frame_counter: Cell::new(0),
                output,
                running: Cell::new(false),
                raf_id: Cell::new(0),
            }),
        }
    }

    /// Starts the animation loop.
    ///
    /// If already running, this is a no-op.
    pub fn start(&self) {
        if self.inner.running.get() {
            return;
        }
        self.inner.running.set(true);

        let inner = Rc::clone(&self.inner);
        let closure = Closure::wrap(Box::new(move |timestamp_ms: f64| {
            if !inner.running.get() {
                return;
            }

            // Convert DOMHighResTimeStamp (ms) → µs ticks.
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "rAF timestamp is a small positive f64; µs fits in u64"
            )]
            let now = HostTime((timestamp_ms * 1000.0) as u64);

            let frame_index = inner.frame_counter.get();
            inner.frame_counter.set(frame_index + 1);

            let tick = FrameTick {
                now,
                predicted_present: None,
                refresh_interval: None,
                confidence: TimingConfidence::PacingOnly,
                frame_index,
                output: inner.output,
                prev_actual_present: None,
            };

            // Invoke user callback. The borrow is scoped so it doesn't
            // overlap with the `closure` RefCell.
            inner.callback.borrow_mut()(tick);

            // Re-register for the next frame if still running.
            if inner.running.get()
                && let Some(ref closure) = *inner.closure.borrow()
            {
                let id = request_animation_frame(closure.as_ref().unchecked_ref());
                inner.raf_id.set(id);
            }
        }) as Box<dyn FnMut(f64)>);

        // Register the first frame.
        let id = request_animation_frame(closure.as_ref().unchecked_ref());
        self.inner.raf_id.set(id);
        *self.inner.closure.borrow_mut() = Some(closure);
    }

    /// Stops the animation loop.
    ///
    /// The pending `requestAnimationFrame` callback is cancelled. Can be
    /// restarted by calling [`start`](Self::start) again.
    pub fn stop(&self) {
        if !self.inner.running.get() {
            return;
        }
        self.inner.running.set(false);
        cancel_animation_frame(self.inner.raf_id.get());
    }

    /// Returns `true` if the loop is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.inner.running.get()
    }
}

impl Drop for RafLoop {
    fn drop(&mut self) {
        self.stop();
        // Drop the JS closure so it doesn't leak.
        self.inner.closure.borrow_mut().take();
    }
}

impl core::fmt::Debug for RafLoop {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RafLoop")
            .field("running", &self.inner.running.get())
            .field("frame_counter", &self.inner.frame_counter.get())
            .field("output", &self.inner.output)
            .finish()
    }
}
