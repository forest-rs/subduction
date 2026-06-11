// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tick forwarding from the `CVDisplayLink` callback thread to the main thread.

use alloc::sync::Arc;
use core::fmt;

use dispatch2::DispatchQueue;
use frameclock::FrameTick;

/// Owns the tick callback and produces [`TickSender`] handles.
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
    pub fn new<F: Fn(FrameTick) + Send + Sync + 'static>(callback: F) -> Self {
        Self {
            inner: Arc::new(callback),
        }
    }

    /// Returns a [`TickSender`] that forwards ticks to this forwarder.
    #[must_use]
    pub fn sender(&self) -> TickSender {
        TickSender {
            callback: Arc::clone(&self.inner),
        }
    }
}

/// A `Send + Sync` handle that dispatches [`FrameTick`] events to the main
/// thread.
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
    pub(crate) fn send(&self, tick: FrameTick) {
        let cb = Arc::clone(&self.callback);
        DispatchQueue::main().exec_async(move || cb(tick));
    }
}
