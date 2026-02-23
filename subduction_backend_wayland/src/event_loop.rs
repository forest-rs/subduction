// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Event-loop ownership contracts for the Wayland backend.
//!
//! This module encodes the two integration modes used by the backend:
//!
//! - [`OwnedQueueMode`]: backend-owned `EventQueue<WaylandState>`
//! - [`EmbeddedStateMode`]: host-owned `EventQueue<HostState>` with backend
//!   dispatch logic delegated from host state
//!
//! # Queue ownership wiring diagram
//!
//! ```text
//! Owned queue mode
//! ----------------
//! backend owns:
//!   EventQueue<WaylandState> + WaylandState
//!     -> QueueHandle<WaylandState>
//! host/toolkit creates wl_surface with QueueHandle<WaylandState>
//! backend objects bind/create with QueueHandle<WaylandState>
//! backend dispatches via OwnedQueueMode::dispatch_pending() or
//! OwnedQueueMode::blocking_dispatch()
//!
//! Embedded-state mode
//! -------------------
//! host owns:
//!   EventQueue<HostState> + HostState { wayland: WaylandState, ... }
//!     -> QueueHandle<HostState>
//! backend wraps:
//!   EmbeddedStateMode<HostState> (contains QueueHandle<HostState>)
//! host/toolkit creates wl_surface with QueueHandle<HostState>
//! backend objects bind/create with QueueHandle<HostState>
//! host dispatches via host EventQueue::dispatch_pending(&mut host_state)
//! and delegates backend Dispatch impls from HostState.
//! ```
//!
//! # `QueueHandle` object-creation contract
//!
//! Every object participating in backend event delivery must be created with
//! the queue handle for the selected mode.
//!
//! | Object | Owned queue mode | Embedded-state mode |
//! |---|---|---|
//! | `wl_surface` | Host/toolkit creates with [`OwnedQueueMode::queue_handle`]. | Host/toolkit creates with host queue handle (`QueueHandle<HostState>`). |
//! | `wl_registry` | Backend binds with `QueueHandle<WaylandState>`. | Backend binds with `QueueHandle<HostState>`. |
//! | `wl_output` | Backend binds with `QueueHandle<WaylandState>`. | Backend binds with `QueueHandle<HostState>`. |
//! | `wp_presentation` | Backend binds with `QueueHandle<WaylandState>`. | Backend binds with `QueueHandle<HostState>`. |
//! | `wl_callback` / `wp_presentation_feedback` | Backend creates with `QueueHandle<WaylandState>`. | Backend creates with `QueueHandle<HostState>`. |
//!
//! Single-surface v1 contract: one backend instance manages one `wl_surface`.
//! Multi-surface routing is intentionally deferred.
//!
//! Using the wrong queue handle causes silent non-delivery of events.

use wayland_client::{
    Connection, DispatchError, EventQueue, QueueHandle,
    backend::{ReadEventsGuard, WaylandError},
};

/// Backend-owned state for Wayland protocol handling.
///
/// In embedded mode, host application state should contain one of these and
/// delegate backend dispatch handling to it.
#[derive(Debug, Default)]
pub struct WaylandState {
    _private: (),
}

impl WaylandState {
    /// Creates a new empty backend state.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

/// Owned-queue integration mode.
///
/// This mode keeps queue ownership entirely inside the backend wrapper and
/// exposes explicit dispatch and queue-handle accessors.
#[derive(Debug)]
pub struct OwnedQueueMode {
    event_queue: EventQueue<WaylandState>,
    state: WaylandState,
}

impl OwnedQueueMode {
    /// Creates an owned-queue integration from an existing Wayland connection.
    #[must_use]
    pub fn new(connection: &Connection) -> Self {
        Self {
            event_queue: connection.new_event_queue(),
            state: WaylandState::new(),
        }
    }

    /// Returns the queue handle that must be used for all backend-relevant
    /// object creation in this mode.
    #[must_use]
    pub fn queue_handle(&self) -> QueueHandle<WaylandState> {
        self.event_queue.handle()
    }

    /// Dispatches already-queued events without blocking.
    ///
    /// This method only runs handlers for events that have already been read
    /// from the Wayland socket into this queue. It does **not** perform socket
    /// I/O by itself.
    ///
    /// In a non-blocking loop, pair this method with [`Self::flush`] and
    /// [`Self::prepare_read`] (or equivalent external connection I/O) to move
    /// protocol traffic before dispatching.
    pub fn dispatch_pending(&mut self) -> Result<usize, DispatchError> {
        self.event_queue.dispatch_pending(&mut self.state)
    }

    /// Flushes requests, blocks for new events when needed, and dispatches.
    ///
    /// This is the easiest complete pumping primitive for simple owned-mode
    /// loops, and wraps [`EventQueue::blocking_dispatch`].
    pub fn blocking_dispatch(&mut self) -> Result<usize, DispatchError> {
        self.event_queue.blocking_dispatch(&mut self.state)
    }

    /// Flushes pending outgoing requests to the Wayland socket.
    pub fn flush(&self) -> Result<(), WaylandError> {
        self.event_queue.flush()
    }

    /// Starts a synchronized socket read for poll-based loops.
    ///
    /// If this returns [`None`], dispatch queued events before trying again.
    #[must_use]
    pub fn prepare_read(&self) -> Option<ReadEventsGuard> {
        self.event_queue.prepare_read()
    }

    /// Returns an immutable reference to backend state.
    #[must_use]
    pub fn state(&self) -> &WaylandState {
        &self.state
    }

    /// Returns a mutable reference to backend state.
    pub fn state_mut(&mut self) -> &mut WaylandState {
        &mut self.state
    }
}

/// Embedded-state integration mode.
///
/// Host code owns the event queue and dispatch loop. The backend stores the
/// host queue handle and relies on delegation wiring from host state.
#[derive(Debug, Clone)]
pub struct EmbeddedStateMode<HostState> {
    queue_handle: QueueHandle<HostState>,
}

impl<HostState> EmbeddedStateMode<HostState> {
    /// Creates an embedded-state integration wrapper from a host-owned queue
    /// handle.
    #[must_use]
    pub fn new(queue_handle: QueueHandle<HostState>) -> Self {
        Self { queue_handle }
    }

    /// Returns the queue handle that must be used for all backend-relevant
    /// object creation in this mode.
    #[must_use]
    pub fn queue_handle(&self) -> QueueHandle<HostState> {
        self.queue_handle.clone()
    }
}
