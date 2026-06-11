// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `requestAnimationFrame` tick source.
//!
//! [`RafLoop`] drives a [`FrameTick`]-based animation loop using the browser's
//! `requestAnimationFrame` API. Each callback receives a
//! [`DOMHighResTimeStamp`][mdn] in milliseconds from `performance.now()`, which
//! is converted to microsecond [`HostTime`] ticks.
//!
//! [mdn]: https://developer.mozilla.org/en-US/docs/Web/API/DOMHighResTimeStamp
//! [`FrameTick`]: frameclock::FrameTick
//! [`HostTime`]: frameclock::HostTime

use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::{Cell, RefCell};

use frameclock::{FrameTick, HostTime, OutputId};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = "now")]
    pub(crate) fn performance_now() -> f64;

    #[wasm_bindgen(js_name = "requestAnimationFrame")]
    fn request_animation_frame(callback: &JsValue) -> i32;

    #[wasm_bindgen(js_name = "cancelAnimationFrame")]
    fn cancel_animation_frame(id: i32);
}

/// A `requestAnimationFrame` loop that emits [`FrameTick`] events.
///
/// Create with [`RafLoop::new`], then call [`start`](Self::start) to begin
/// receiving callbacks. The loop re-registers itself each frame until
/// [`stop`](Self::stop) is called or the loop is dropped.
pub struct RafLoop {
    inner: Rc<RafInner>,
}

type RafClosure = Closure<dyn FnMut(f64)>;

struct RafInner {
    closure: RefCell<Option<RafClosure>>,
    callback: RefCell<Box<dyn FnMut(FrameTick)>>,
    frame_counter: Cell<u64>,
    output: OutputId,
    running: Cell<bool>,
    raf_id: Cell<i32>,
}

impl RafLoop {
    /// Creates a new loop that is not yet running.
    ///
    /// `callback` receives a [`FrameTick`] on each animation frame once
    /// [`start`](Self::start) is called. `output` identifies the display
    /// surface for emitted ticks.
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
    /// Calling this while the loop is already running is a no-op.
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

            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "RAF timestamp is a small positive f64; microseconds fit in u64"
            )]
            let now = HostTime((timestamp_ms * 1000.0) as u64);

            let frame_index = inner.frame_counter.get();
            inner.frame_counter.set(frame_index + 1);

            let tick = FrameTick {
                now,
                predicted_present: None,
                refresh_interval: None,
                frame_index,
                output: inner.output,
                prev_actual_present: None,
            };

            inner.callback.borrow_mut()(tick);

            if inner.running.get()
                && let Some(ref closure) = *inner.closure.borrow()
            {
                let id = request_animation_frame(closure.as_ref().unchecked_ref());
                inner.raf_id.set(id);
            }
        }) as Box<dyn FnMut(f64)>);

        let id = request_animation_frame(closure.as_ref().unchecked_ref());
        self.inner.raf_id.set(id);
        *self.inner.closure.borrow_mut() = Some(closure);
    }

    /// Stops the animation loop.
    ///
    /// The pending `requestAnimationFrame` callback is cancelled. The loop can
    /// be restarted by calling [`start`](Self::start) again.
    pub fn stop(&self) {
        if !self.inner.running.get() {
            return;
        }
        self.inner.running.set(false);
        cancel_animation_frame(self.inner.raf_id.get());
    }

    /// Returns `true` when the loop is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.inner.running.get()
    }
}

impl Drop for RafLoop {
    fn drop(&mut self) {
        self.stop();
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
