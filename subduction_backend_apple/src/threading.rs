// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tick forwarding from the `CVDisplayLink` callback thread to the main thread.
//!
//! `CVDisplayLink` callbacks run on a high-priority `CoreVideo` background thread.
//! This module provides [`TickForwarder`] and [`TickSender`] to dispatch
//! [`FrameTick`] events to the main thread's run loop with minimal latency.
//!
//! [`FrameTick`]: frameclock::FrameTick

use alloc::sync::Arc;
use core::fmt;

use dispatch2::DispatchQueue;
use frameclock::FrameTick;

/// Owns the tick callback and produces [`TickSender`] handles.
///
/// Created on the main thread with a callback that will be invoked (also on the
/// main thread) for each [`FrameTick`] forwarded by a [`TickSender`].
pub struct TickForwarder {
    inner: Arc<dyn Fn(FrameTick) + Send + Sync>,
}

impl fmt::Debug for TickForwarder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TickForwarder").finish_non_exhaustive()
    }
}

impl TickForwarder {
    /// Creates a new forwarder with the given callback.
    ///
    /// The callback will be invoked on the main thread for each tick
    /// dispatched by a [`TickSender`].
    pub fn new<F: Fn(FrameTick) + Send + Sync + 'static>(callback: F) -> Self {
        Self {
            inner: Arc::new(callback),
        }
    }

    /// Returns a [`TickSender`] that can forward ticks to this forwarder's
    /// callback on the main thread.
    #[must_use]
    pub fn sender(&self) -> TickSender {
        TickSender {
            callback: Arc::clone(&self.inner),
        }
    }
}

/// A `Send + Sync` handle that dispatches [`FrameTick`] events to the main
/// thread.
///
/// Obtained from [`TickForwarder::sender`]. Cloning is cheap (Arc bump).
#[derive(Clone)]
pub struct TickSender {
    callback: Arc<dyn Fn(FrameTick) + Send + Sync>,
}

impl fmt::Debug for TickSender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TickSender").finish_non_exhaustive()
    }
}

impl TickSender {
    /// Dispatches `tick` to the main thread asynchronously.
    ///
    /// The [`TickForwarder`]'s callback will be invoked on the main queue.
    /// This method is safe to call from any thread (including the
    /// `CVDisplayLink` callback thread).
    pub(crate) fn send(&self, tick: FrameTick) {
        let cb = Arc::clone(&self.callback);
        DispatchQueue::main().exec_async(move || cb(tick));
    }
}
