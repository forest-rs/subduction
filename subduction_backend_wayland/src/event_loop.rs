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
//! which is the easiest way to pump events. After dispatch, drain queued
//! ticks via [`OwnedQueueMode::poll_tick`]:
//!
//! ```rust,no_run
//! use wayland_client::Connection;
//! use subduction_backend_wayland::OwnedQueueMode;
//!
//! let connection = Connection::connect_to_env().unwrap();
//! let mut mode = OwnedQueueMode::new(&connection);
//! mode.bootstrap().unwrap();
//!
//! // Register a surface (created by the host/toolkit).
//! // mode.state_mut().set_surface(surface).unwrap();
//!
//! loop {
//!     // 1. Dispatch — delivers wl_callback.done → enqueues ticks.
//!     mode.blocking_dispatch().unwrap();
//!
//!     // 2. Poll — drain all queued ticks.
//!     while let Some(_tick) = mode.poll_tick() {
//!         // 3. Process — compute hints, build frame ...
//!         // 4. attach buffer + damage ...
//!         // 5. Commit — requests next callback, feedback, commits, flushes.
//!         // let _id = mode.commit_frame().unwrap();
//!     }
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
//!    [`ReadEventsGuard`]. If
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
//! use wayland_client::protocol::{wl_callback, wl_output, wl_registry};
//! use wayland_client::{Connection, EventQueue};
//! use wayland_protocols::wp::presentation_time::client::{
//!     wp_presentation, wp_presentation_feedback,
//! };
//! use subduction_backend_wayland::{
//!     EmbeddedStateMode, FeedbackData, FrameCallbackData, OutputGlobalData,
//!     WaylandProtocol, WaylandState,
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
//! wayland_client::delegate_dispatch!(HostState:
//!     [wl_callback::WlCallback: FrameCallbackData] => WaylandProtocol);
//! wayland_client::delegate_dispatch!(HostState:
//!     [wp_presentation_feedback::WpPresentationFeedback: FeedbackData] => WaylandProtocol);
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
//!     // Dispatch — delivers protocol events including wl_callback.done.
//!     event_queue.blocking_dispatch(&mut state).unwrap();
//!
//!     // Poll ticks enqueued by the callback handler.
//!     while let Some(_tick) = state.wayland.poll_tick() {
//!         // Process tick, compute hints, build frame ...
//!         // attach buffer + damage ...
//!         // Commit — requests next callback, feedback, commits, flushes.
//!         // let _id = state.wayland.commit_frame(&qh, &connection).unwrap();
//!     }
//! }
//! ```

use crate::commit::{CommitFrameError, CommitState, FeedbackData};
use crate::output_registry::OutputRegistry;
use crate::presentation::{PendingFeedback, PresentEvent, PresentEventQueue, SubmissionId};
use crate::protocol::{Capabilities, FrameCallbackData, OutputGlobalData, WaylandProtocol};
use crate::tick::TickerState;
use crate::time::{Clock, now_for_clock};
use std::collections::HashMap;
use subduction_core::time::HostTime;
use subduction_core::timing::FrameTick;
use wayland_client::protocol::{wl_callback, wl_output, wl_registry, wl_surface};
use wayland_client::{
    Connection, Dispatch, DispatchError, EventQueue, QueueHandle,
    backend::{ReadEventsGuard, WaylandError},
};
use wayland_protocols::wp::presentation_time::client::{wp_presentation, wp_presentation_feedback};

/// Error returned by [`WaylandState::set_surface`] when a surface has already
/// been registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetSurfaceError {
    /// A surface has already been set; the single-surface contract allows only
    /// one.
    AlreadySet,
}

/// Error returned when requesting a frame callback is not possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestFrameError {
    /// No surface has been registered via [`WaylandState::set_surface`].
    NoSurface,
    /// A frame callback is already in flight.
    AlreadyInFlight,
}

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
///    `WlRegistry`, `WlOutput`, `WpPresentation`, and `WlCallback` via
///    [`WaylandProtocol`].
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
    pub(crate) ticker: TickerState,
    pub(crate) commit: CommitState,
    pub(crate) surface: Option<wl_surface::WlSurface>,
    pub(crate) present_events: PresentEventQueue,
    pub(crate) pending_feedback: HashMap<SubmissionId, PendingFeedback>,
}

impl WaylandState {
    /// Creates a new empty backend state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: None,
            output_registry: OutputRegistry::new(),
            capabilities: Capabilities::new(),
            clock: Clock::Monotonic,
            presentation: None,
            bootstrapped: false,
            ticker: TickerState::new(),
            commit: CommitState::new(),
            surface: None,
            present_events: PresentEventQueue::default(),
            pending_feedback: HashMap::new(),
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

    /// Registers the surface that the backend will pace with frame callbacks.
    ///
    /// This is a one-shot operation enforcing the single-surface v1 contract.
    /// Returns [`SetSurfaceError::AlreadySet`] if called more than once.
    pub fn set_surface(&mut self, surface: wl_surface::WlSurface) -> Result<(), SetSurfaceError> {
        if self.surface.is_some() {
            return Err(SetSurfaceError::AlreadySet);
        }
        self.surface = Some(surface);
        Ok(())
    }

    /// Pops the next queued [`FrameTick`], if any.
    #[must_use]
    pub fn poll_tick(&mut self) -> Option<FrameTick> {
        self.ticker.poll_tick()
    }

    /// Pops the next queued [`PresentEvent`], if any.
    #[must_use]
    pub fn poll_present_event(&mut self) -> Option<PresentEvent> {
        self.present_events.pop()
    }

    /// Requests the next frame callback from the compositor.
    ///
    /// Sends a `wl_surface.frame()` request with backend-specific userdata,
    /// marking one callback as in-flight. Returns
    /// [`RequestFrameError::NoSurface`] if no surface has been registered, or
    /// [`RequestFrameError::AlreadyInFlight`] if a callback is already pending.
    ///
    /// # Flush requirement
    ///
    /// This method emits a protocol request but does **not** flush the
    /// connection. In owned mode, the next
    /// [`blocking_dispatch`](OwnedQueueMode::blocking_dispatch) flushes
    /// automatically. In non-blocking or embedded mode, the caller must flush.
    pub fn request_frame<D>(&mut self, qh: &QueueHandle<D>) -> Result<(), RequestFrameError>
    where
        D: Dispatch<wl_callback::WlCallback, FrameCallbackData> + AsMut<Self> + 'static,
    {
        let surface = self.surface.as_ref().ok_or(RequestFrameError::NoSurface)?;
        if self.ticker.is_callback_in_flight() {
            return Err(RequestFrameError::AlreadyInFlight);
        }
        let _callback = surface.frame(qh, FrameCallbackData);
        self.ticker.mark_callback_requested();
        Ok(())
    }

    /// Sequences a frame callback request, presentation feedback request,
    /// surface commit, and connection flush in the correct protocol order.
    ///
    /// Returns the [`SubmissionId`] assigned to this commit, which can be
    /// correlated with future [`PresentEvent`]s.
    ///
    /// # Flush
    ///
    /// This method always flushes the connection after committing. If flush
    /// fails, the surface commit was buffered but may not have reached the
    /// compositor; the caller should treat this as a transport error.
    ///
    /// # Presentation feedback
    ///
    /// When `wp_presentation` is available and the pending feedback count is
    /// below the internal limit, `commit_frame` requests presentation feedback
    /// for this commit. If the limit is reached, feedback is silently skipped
    /// — the commit and frame callback request still proceed.
    pub fn commit_frame<D>(
        &mut self,
        qh: &QueueHandle<D>,
        conn: &Connection,
    ) -> Result<SubmissionId, CommitFrameError>
    where
        D: Dispatch<wl_callback::WlCallback, FrameCallbackData>
            + Dispatch<wp_presentation_feedback::WpPresentationFeedback, FeedbackData>
            + AsMut<Self>
            + 'static,
    {
        // Clone proxies to avoid borrow conflicts with self.ticker / self.commit.
        let surface = self.surface.clone().ok_or(CommitFrameError::NoSurface)?;
        let presentation = self.presentation.clone();

        // 1. Request next frame callback (best-effort; skip if already in flight).
        if !self.ticker.is_callback_in_flight() {
            let _cb = surface.frame(qh, FrameCallbackData);
            self.ticker.mark_callback_requested();
        }

        // 2. Allocate submission ID.
        let id = self.commit.allocate_id();

        // 3. Request presentation feedback if available and under limit.
        if let Some(pres) = presentation
            && !self.commit.is_at_limit()
        {
            let _fb = pres.feedback(&surface, qh, FeedbackData { submission_id: id });
            self.commit.increment_pending();
            self.pending_feedback
                .insert(id, PendingFeedback { sync_output: None });
        }

        // 4. Commit the surface.
        surface.commit();

        // 5. Flush.
        conn.flush().map_err(CommitFrameError::Flush)?;

        Ok(id)
    }

    /// Returns current host time using the selected backend clock.
    ///
    /// After `wp_presentation.clock_id` has been received, this reads the
    /// compositor-aligned clock. Before that, it falls back to
    /// `CLOCK_MONOTONIC`.
    #[allow(dead_code, reason = "called from future wl_callback dispatch handler")]
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
wayland_client::delegate_dispatch!(WaylandState: [wl_callback::WlCallback: FrameCallbackData] => WaylandProtocol);
wayland_client::delegate_dispatch!(WaylandState: [wp_presentation_feedback::WpPresentationFeedback: FeedbackData] => WaylandProtocol);

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

    /// Requests the next frame callback from the compositor.
    ///
    /// Convenience wrapper that calls [`WaylandState::request_frame`] with the
    /// owned queue handle. Does **not** flush — the next
    /// [`blocking_dispatch`](Self::blocking_dispatch) flushes automatically, or
    /// call [`flush`](Self::flush) explicitly.
    pub fn request_frame(&mut self) -> Result<(), RequestFrameError> {
        let qh = self.event_queue.handle();
        self.state.request_frame(&qh)
    }

    /// Sequences a frame callback request, presentation feedback request,
    /// surface commit, and connection flush.
    ///
    /// Convenience wrapper that calls [`WaylandState::commit_frame`] with
    /// the owned queue handle and stored connection.
    pub fn commit_frame(&mut self) -> Result<SubmissionId, CommitFrameError> {
        let qh = self.event_queue.handle();
        self.state.commit_frame(&qh, &self.connection)
    }

    /// Pops the next queued [`FrameTick`], if any.
    ///
    /// Convenience wrapper that calls [`WaylandState::poll_tick`].
    #[must_use]
    pub fn poll_tick(&mut self) -> Option<FrameTick> {
        self.state.poll_tick()
    }

    /// Pops the next queued [`PresentEvent`], if any.
    ///
    /// Convenience wrapper that calls [`WaylandState::poll_present_event`].
    #[must_use]
    pub fn poll_present_event(&mut self) -> Option<PresentEvent> {
        self.state.poll_present_event()
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

#[cfg(test)]
impl WaylandState {
    /// Test helper: simulates the `sync_output` event from the dispatch handler.
    ///
    /// Takes a pre-resolved `Option<OutputId>` (bypassing proxy lookup).
    fn test_on_sync_output(
        &mut self,
        id: SubmissionId,
        resolved: Option<subduction_core::output::OutputId>,
    ) {
        if let Some(pending) = self.pending_feedback.get_mut(&id)
            && (resolved.is_some() || pending.sync_output.is_none())
        {
            pending.sync_output = resolved;
        }
    }

    /// Test helper: simulates the `presented` terminal event from the dispatch handler.
    fn test_on_presented(
        &mut self,
        id: SubmissionId,
        actual_present: HostTime,
        refresh_interval: Option<u64>,
        flags: u32,
    ) {
        let output = self
            .pending_feedback
            .remove(&id)
            .and_then(|p| p.sync_output);
        self.present_events.push(PresentEvent::Presented {
            id,
            actual_present,
            refresh_interval,
            output,
            flags,
        });
        self.ticker.set_last_observed_actual_present(actual_present);
        self.commit.decrement_pending();
    }

    /// Test helper: simulates the `discarded` terminal event from the dispatch handler.
    fn test_on_discarded(&mut self, id: SubmissionId) {
        let _ = self.pending_feedback.remove(&id);
        self.present_events.push(PresentEvent::Discarded { id });
        self.commit.decrement_pending();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subduction_core::output::OutputId;
    use wayland_client::Proxy;
    use wayland_client::backend::ObjectId;

    fn test_queue_handle() -> (EventQueue<WaylandState>, QueueHandle<WaylandState>) {
        let (s1, _s2) = std::os::unix::net::UnixStream::pair().unwrap();
        let conn = Connection::from_socket(s1).unwrap();
        let eq: EventQueue<WaylandState> = conn.new_event_queue();
        let qh = eq.handle();
        (eq, qh)
    }

    fn inert_surface() -> wl_surface::WlSurface {
        let (s1, _s2) = std::os::unix::net::UnixStream::pair().unwrap();
        let conn = Connection::from_socket(s1).unwrap();
        wl_surface::WlSurface::from_id(&conn, ObjectId::null()).unwrap()
    }

    #[test]
    fn request_frame_without_surface_returns_error() {
        let (_eq, qh) = test_queue_handle();
        let mut state = WaylandState::new();
        assert_eq!(state.request_frame(&qh), Err(RequestFrameError::NoSurface),);
    }

    #[test]
    fn request_frame_when_in_flight_returns_already_in_flight() {
        let (_eq, qh) = test_queue_handle();
        let mut state = WaylandState::new();
        state.set_surface(inert_surface()).unwrap();
        state.ticker.mark_callback_requested();
        assert_eq!(
            state.request_frame(&qh),
            Err(RequestFrameError::AlreadyInFlight),
        );
    }

    fn test_connection_and_queue() -> (
        Connection,
        EventQueue<WaylandState>,
        QueueHandle<WaylandState>,
    ) {
        let (s1, _s2) = std::os::unix::net::UnixStream::pair().unwrap();
        let conn = Connection::from_socket(s1).unwrap();
        let eq: EventQueue<WaylandState> = conn.new_event_queue();
        let qh = eq.handle();
        (conn, eq, qh)
    }

    #[test]
    fn commit_frame_without_surface_returns_no_surface() {
        let (conn, _eq, qh) = test_connection_and_queue();
        let mut state = WaylandState::new();
        let result = state.commit_frame(&qh, &conn);
        assert!(matches!(result, Err(CommitFrameError::NoSurface)));
    }

    #[test]
    fn commit_frame_with_inert_surface_returns_submission_id() {
        let (conn, _eq, qh) = test_connection_and_queue();
        let mut state = WaylandState::new();
        state.set_surface(inert_surface()).unwrap();
        let id = state.commit_frame(&qh, &conn).unwrap();
        assert_eq!(id, SubmissionId(0));
    }

    #[test]
    fn successive_commit_frames_produce_monotonic_ids() {
        let (conn, _eq, qh) = test_connection_and_queue();
        let mut state = WaylandState::new();
        state.set_surface(inert_surface()).unwrap();

        let id0 = state.commit_frame(&qh, &conn).unwrap();
        // Clear in-flight so next commit_frame can request a new callback.
        let empty_reg = OutputRegistry::new();
        state.ticker.on_callback_done(state.clock, &empty_reg);

        let id1 = state.commit_frame(&qh, &conn).unwrap();
        assert!(id1 > id0);
    }

    #[test]
    fn commit_frame_marks_callback_in_flight() {
        let (conn, _eq, qh) = test_connection_and_queue();
        let mut state = WaylandState::new();
        state.set_surface(inert_surface()).unwrap();

        assert!(!state.ticker.is_callback_in_flight());
        let _id = state.commit_frame(&qh, &conn).unwrap();
        assert!(state.ticker.is_callback_in_flight());
    }

    #[test]
    fn commit_frame_skips_callback_when_already_in_flight() {
        let (conn, _eq, qh) = test_connection_and_queue();
        let mut state = WaylandState::new();
        state.set_surface(inert_surface()).unwrap();

        // First commit requests a callback.
        let _id0 = state.commit_frame(&qh, &conn).unwrap();
        assert!(state.ticker.is_callback_in_flight());

        // Second commit skips the callback request (no error).
        let _id1 = state.commit_frame(&qh, &conn).unwrap();
        assert!(state.ticker.is_callback_in_flight());
    }

    // --- Presentation feedback integration tests ---

    /// Helper: sets up a `WaylandState` with a pending feedback entry.
    fn state_with_pending(id: SubmissionId) -> WaylandState {
        let mut state = WaylandState::new();
        state
            .pending_feedback
            .insert(id, PendingFeedback { sync_output: None });
        state.commit.increment_pending();
        state
    }

    #[test]
    fn flags_extraction_wenum_value() {
        use wayland_protocols::wp::presentation_time::client::wp_presentation_feedback::Kind;
        // Kind is bitflags: Vsync = 0x1, HwClock = 0x2.
        let combined = Kind::Vsync | Kind::HwClock;
        assert_eq!(combined.bits(), 0x3);
    }

    #[test]
    fn flags_extraction_wenum_unknown() {
        use wayland_client::WEnum;
        use wayland_protocols::wp::presentation_time::client::wp_presentation_feedback::Kind;
        let raw: u32 = match WEnum::<Kind>::Unknown(0xFF) {
            WEnum::Value(k) => k.bits(),
            WEnum::Unknown(v) => v,
        };
        assert_eq!(raw, 0xFF);
    }

    #[test]
    fn discard_enqueues_event_and_decrements_pending() {
        let id = SubmissionId(0);
        let mut state = state_with_pending(id);
        assert_eq!(state.commit.pending_count(), 1);

        state.test_on_discarded(id);

        assert_eq!(
            state.poll_present_event(),
            Some(PresentEvent::Discarded { id })
        );
        assert_eq!(state.commit.pending_count(), 0);
        assert!(state.pending_feedback.is_empty());
    }

    #[test]
    fn out_of_order_delivery() {
        let id0 = SubmissionId(0);
        let id1 = SubmissionId(1);
        let mut state = WaylandState::new();
        // Insert both pending entries.
        state
            .pending_feedback
            .insert(id0, PendingFeedback { sync_output: None });
        state.commit.increment_pending();
        state
            .pending_feedback
            .insert(id1, PendingFeedback { sync_output: None });
        state.commit.increment_pending();

        // Feedback for id1 arrives first.
        state.test_on_presented(id1, HostTime(200), Some(16_666_667), 0);
        // Then id0.
        state.test_on_presented(id0, HostTime(100), Some(16_666_667), 0);

        let ev1 = state.poll_present_event().unwrap();
        let ev0 = state.poll_present_event().unwrap();
        assert!(matches!(ev1, PresentEvent::Presented { id, .. } if id == id1));
        assert!(matches!(ev0, PresentEvent::Presented { id, .. } if id == id0));
        assert_eq!(state.commit.pending_count(), 0);
    }

    #[test]
    fn sync_output_present_resolves_output() {
        let id = SubmissionId(0);
        let mut state = state_with_pending(id);
        let output = OutputId(7);

        state.test_on_sync_output(id, Some(output));
        state.test_on_presented(id, HostTime(1000), None, 0);

        let ev = state.poll_present_event().unwrap();
        assert!(matches!(
            ev,
            PresentEvent::Presented { output: Some(o), .. } if o == output
        ));
    }

    #[test]
    fn sync_output_unknown_produces_none() {
        let id = SubmissionId(0);
        let mut state = state_with_pending(id);

        // sync_output with a proxy not in the registry → None.
        state.test_on_sync_output(id, None);
        state.test_on_presented(id, HostTime(1000), None, 0);

        let ev = state.poll_present_event().unwrap();
        assert!(matches!(ev, PresentEvent::Presented { output: None, .. }));
    }

    #[test]
    fn sync_output_missing_entirely() {
        let id = SubmissionId(0);
        let mut state = state_with_pending(id);

        // No sync_output before presented → output: None.
        state.test_on_presented(id, HostTime(1000), None, 0);

        let ev = state.poll_present_event().unwrap();
        assert!(matches!(ev, PresentEvent::Presented { output: None, .. }));
    }

    #[test]
    fn known_beats_unknown_sync_output_policy() {
        let id = SubmissionId(0);
        let mut state = state_with_pending(id);
        let known = OutputId(3);

        // First: known output arrives.
        state.test_on_sync_output(id, Some(known));
        // Second: unknown output arrives — should NOT overwrite.
        state.test_on_sync_output(id, None);

        state.test_on_presented(id, HostTime(1000), None, 0);

        let ev = state.poll_present_event().unwrap();
        assert!(matches!(
            ev,
            PresentEvent::Presented { output: Some(o), .. } if o == known
        ));
    }

    #[test]
    fn last_observed_actual_present_updated() {
        let id = SubmissionId(0);
        let mut state = state_with_pending(id);

        state.test_on_presented(id, HostTime(42_000), None, 0);

        // The ticker should have the actual present time stored.
        // Verify by generating a tick and checking prev_actual_present.
        let reg = OutputRegistry::new();
        state.ticker.mark_callback_requested();
        state.ticker.on_callback_done(state.clock, &reg);
        let tick = state.ticker.poll_tick().unwrap();
        assert_eq!(tick.prev_actual_present, Some(HostTime(42_000)));
    }

    #[test]
    fn terminal_event_for_missing_submission_presented() {
        let mut state = WaylandState::new();
        let id = SubmissionId(99);
        // Simulate increment that happened at creation time.
        state.commit.increment_pending();

        // Presented for an id not in the pending map — no panic, event
        // still emitted, pending count still decrements.
        state.test_on_presented(id, HostTime(500), None, 0);

        let ev = state.poll_present_event().unwrap();
        assert!(matches!(
            ev,
            PresentEvent::Presented { id: sid, output: None, .. } if sid == id
        ));
        assert_eq!(state.commit.pending_count(), 0);
    }

    #[test]
    fn terminal_event_for_missing_submission_discarded() {
        let mut state = WaylandState::new();
        let id = SubmissionId(99);
        state.commit.increment_pending();

        state.test_on_discarded(id);

        let ev = state.poll_present_event().unwrap();
        assert_eq!(ev, PresentEvent::Discarded { id });
        assert_eq!(state.commit.pending_count(), 0);
    }

    #[test]
    fn pending_map_counter_coherence() {
        let mut state = WaylandState::new();

        // Normal flow: insert two pending entries.
        let id0 = SubmissionId(0);
        let id1 = SubmissionId(1);
        state
            .pending_feedback
            .insert(id0, PendingFeedback { sync_output: None });
        state.commit.increment_pending();
        state
            .pending_feedback
            .insert(id1, PendingFeedback { sync_output: None });
        state.commit.increment_pending();
        assert_eq!(
            state.pending_feedback.len(),
            state.commit.pending_count() as usize
        );

        // Discard one.
        state.test_on_discarded(id0);
        assert_eq!(
            state.pending_feedback.len(),
            state.commit.pending_count() as usize
        );

        // Present the other.
        state.test_on_presented(id1, HostTime(100), None, 0);
        assert_eq!(
            state.pending_feedback.len(),
            state.commit.pending_count() as usize
        );
        assert_eq!(state.pending_feedback.len(), 0);
        assert_eq!(state.commit.pending_count(), 0);
    }
}
