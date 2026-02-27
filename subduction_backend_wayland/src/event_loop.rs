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
//!
//! # Owned queue mode
//!
//! ## Simple blocking loop
//!
//! [`OwnedQueueMode::blocking_dispatch`] flushes and blocks in one call,
//! which is the easiest way to pump events:
//!
//! ```rust,no_run
//! use wayland_client::Connection;
//! use subduction_backend_wayland::OwnedQueueMode;
//!
//! let connection = Connection::connect_to_env().unwrap();
//! let mut mode = OwnedQueueMode::new(&connection);
//! mode.bootstrap().unwrap();
//!
//! loop {
//!     mode.blocking_dispatch().unwrap();
//!     let _caps = mode.capabilities();
//!     // ... poll ticks, check capabilities ...
//! }
//! ```
//!
//! ## Non-blocking (poll-based) loop
//!
//! For integration with an external event loop, use the five-step pattern:
//!
//! 1. [`flush()`](OwnedQueueMode::flush) — send pending outgoing requests.
//! 2. [`dispatch_pending()`](OwnedQueueMode::dispatch_pending) — process
//!    any already-buffered events.
//! 3. [`prepare_read()`](OwnedQueueMode::prepare_read) — obtain a
//!    [`ReadEventsGuard`](wayland_client::backend::ReadEventsGuard). If
//!    this returns `None`, go back to step 2.
//! 4. Poll the fd from `guard.connection_fd()` for readability.
//! 5. `guard.read()` — read events from the socket, then go to step 2.
//!
//! `dispatch_pending` alone never reads from the socket — skipping the
//! `prepare_read` / `read` cycle will stall the loop.
//!
//! ```rust,no_run
//! use wayland_client::Connection;
//! use subduction_backend_wayland::OwnedQueueMode;
//!
//! let connection = Connection::connect_to_env().unwrap();
//! let mut mode = OwnedQueueMode::new(&connection);
//! mode.bootstrap().unwrap();
//!
//! loop {
//!     mode.flush().unwrap();
//!     mode.dispatch_pending().unwrap();
//!
//!     if let Some(guard) = mode.prepare_read() {
//!         let _fd = guard.connection_fd();
//!         // ... poll fd for readability ...
//!         guard.read().unwrap();
//!     }
//!     // dispatch again after reading
//!     mode.dispatch_pending().unwrap();
//! }
//! ```
//!
//! # Embedded-state mode
//!
//! When the host already owns the Wayland event queue, embed a
//! [`WaylandState`] inside the host state and wire delegation so that
//! backend protocol events are forwarded through [`WaylandProtocol`].
//!
//! The host must:
//!
//! - Contain a [`WaylandState`] field in its state struct.
//! - Implement `AsMut<WaylandState>` for the host state.
//! - Call [`delegate_dispatch!`](wayland_client::delegate_dispatch) for
//!   each protocol object the backend handles.
//! - Call [`WaylandState::set_registry`] with the host-created registry.
//! - Drive the roundtrip and dispatch loop itself.
//! - Flush the connection after emitting requests (the backend does not
//!   flush on the host's behalf in this mode).
//!
//! ```rust,no_run
//! use wayland_client::protocol::{wl_output, wl_registry};
//! use wayland_client::{Connection, EventQueue};
//! use wayland_protocols::wp::presentation_time::client::wp_presentation;
//! use subduction_backend_wayland::{
//!     EmbeddedStateMode, OutputGlobalData, WaylandProtocol, WaylandState,
//! };
//!
//! struct HostState {
//!     wayland: WaylandState,
//!     // ... other host fields ...
//! }
//!
//! impl AsMut<WaylandState> for HostState {
//!     fn as_mut(&mut self) -> &mut WaylandState {
//!         &mut self.wayland
//!     }
//! }
//!
//! wayland_client::delegate_dispatch!(HostState:
//!     [wl_registry::WlRegistry: ()] => WaylandProtocol);
//! wayland_client::delegate_dispatch!(HostState:
//!     [wl_output::WlOutput: OutputGlobalData] => WaylandProtocol);
//! wayland_client::delegate_dispatch!(HostState:
//!     [wp_presentation::WpPresentation: ()] => WaylandProtocol);
//!
//! let connection = Connection::connect_to_env().unwrap();
//! let mut event_queue: EventQueue<HostState> = connection.new_event_queue();
//! let qh = event_queue.handle();
//!
//! let display = connection.display();
//! let registry = display.get_registry(&qh, ());
//!
//! let mut state = HostState {
//!     wayland: WaylandState::new(),
//! };
//! state.wayland.set_registry(registry);
//!
//! // Initial roundtrip populates the output registry.
//! event_queue.roundtrip(&mut state).unwrap();
//!
//! loop {
//!     event_queue.blocking_dispatch(&mut state).unwrap();
//!     let _caps = state.wayland.capabilities();
//!     // ... host dispatch logic ...
//! }
//! ```

use crate::output_registry::OutputRegistry;
use crate::protocol::{Capabilities, OutputGlobalData, WaylandProtocol};
use crate::time::{Clock, now_for_clock};
use subduction_core::time::HostTime;
use wayland_client::protocol::{wl_output, wl_registry};
use wayland_client::{
    Connection, DispatchError, EventQueue, QueueHandle,
    backend::{ReadEventsGuard, WaylandError},
};
use wayland_protocols::wp::presentation_time::client::wp_presentation;

/// Backend-owned state for Wayland protocol handling.
///
/// In embedded mode, host application state should contain one of these and
/// delegate backend dispatch handling to it.
///
/// # Embedded-mode wiring
///
/// Hosts that own their own event queue must:
///
/// 1. Call `wl_display.get_registry()` on their queue handle.
/// 2. Call [`WaylandState::set_registry`] to store the registry proxy.
/// 3. Implement `AsMut<WaylandState>` on their host state.
/// 4. Wire [`delegate_dispatch!`](wayland_client::delegate_dispatch) for
///    `WlRegistry` and `WlOutput` via [`WaylandProtocol`].
/// 5. Drive dispatch and the initial roundtrip themselves.
///
/// Embedded-mode hosts are responsible for flushing the connection after
/// emitting requests. Future commit-sequencing APIs will handle flushing
/// internally.
#[derive(Debug)]
pub struct WaylandState {
    pub(crate) registry: Option<wl_registry::WlRegistry>,
    pub(crate) output_registry: OutputRegistry,
    pub(crate) capabilities: Capabilities,
    pub(crate) clock: Clock,
    pub(crate) presentation: Option<wp_presentation::WpPresentation>,
    pub(crate) bootstrapped: bool,
}

impl WaylandState {
    /// Creates a new empty backend state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registry: None,
            output_registry: OutputRegistry::new(),
            capabilities: Capabilities::new(),
            clock: Clock::Monotonic,
            presentation: None,
            bootstrapped: false,
        }
    }

    /// Returns the current protocol capabilities.
    #[must_use]
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    /// Stores a host-created registry proxy for embedded-mode integration.
    ///
    /// Call this after creating the registry on your own queue handle so that
    /// [`WaylandState`] knows global discovery is possible.
    pub fn set_registry(&mut self, registry: wl_registry::WlRegistry) {
        self.registry = Some(registry);
    }

    /// Returns current host time using the selected backend clock.
    ///
    /// After `wp_presentation.clock_id` has been received, this reads the
    /// compositor-aligned clock. Before that, it falls back to
    /// `CLOCK_MONOTONIC`.
    #[allow(
        dead_code,
        reason = "will be used by ticker/presenter in future implementation"
    )]
    #[must_use]
    pub(crate) fn now(&self) -> HostTime {
        now_for_clock(self.clock)
    }
}

impl Default for WaylandState {
    fn default() -> Self {
        Self::new()
    }
}

impl AsMut<Self> for WaylandState {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

wayland_client::delegate_dispatch!(WaylandState: [wl_registry::WlRegistry: ()] => WaylandProtocol);
wayland_client::delegate_dispatch!(WaylandState: [wl_output::WlOutput: OutputGlobalData] => WaylandProtocol);
wayland_client::delegate_dispatch!(WaylandState: [wp_presentation::WpPresentation: ()] => WaylandProtocol);

/// Owned-queue integration mode.
///
/// This mode keeps queue ownership entirely inside the backend wrapper and
/// exposes explicit dispatch and queue-handle accessors.
#[derive(Debug)]
pub struct OwnedQueueMode {
    connection: Connection,
    event_queue: EventQueue<WaylandState>,
    state: WaylandState,
}

impl OwnedQueueMode {
    /// Creates an owned-queue integration from an existing Wayland connection.
    ///
    /// The connection is cloned internally so that [`Self::bootstrap`] does
    /// not require the caller to pass it again.
    #[must_use]
    pub fn new(connection: &Connection) -> Self {
        Self {
            connection: connection.clone(),
            event_queue: connection.new_event_queue(),
            state: WaylandState::new(),
        }
    }

    /// Performs initial global discovery via a blocking roundtrip.
    ///
    /// This creates the `wl_registry` (once), performs a blocking roundtrip to
    /// populate the output registry, and marks the state as bootstrapped.
    ///
    /// Failure is idempotent: the registry proxy survives a failed roundtrip so
    /// a retry re-attempts without creating a duplicate.
    pub fn bootstrap(&mut self) -> Result<(), DispatchError> {
        if self.state.bootstrapped {
            return Ok(());
        }
        if self.state.registry.is_none() {
            let display = self.connection.display();
            let qh = self.event_queue.handle();
            self.state.registry = Some(display.get_registry(&qh, ()));
        }
        self.event_queue.roundtrip(&mut self.state)?;
        self.state.bootstrapped = true;
        Ok(())
    }

    /// Returns the current protocol capabilities.
    #[must_use]
    pub fn capabilities(&self) -> Capabilities {
        self.state.capabilities()
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
