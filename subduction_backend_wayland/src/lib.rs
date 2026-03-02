// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland backend for subduction.
//!
//! This crate provides Wayland compositor integration for the subduction
//! timing framework, including:
//!
//! - Frame callback tick source (pull-based, pacing-only)
//! - Optional `wp_presentation` for actual present time feedback
//! - `wl_surface` commit presenter
//!
//! # Integration modes
//!
//! The backend supports two queue-ownership models so it can fit into both
//! self-contained applications and larger toolkits that already own an event
//! queue.
//!
//! - **[`OwnedQueueMode`]** — the backend owns the `EventQueue` and
//!   `WaylandState`. The host calls [`OwnedQueueMode::bootstrap`] and then
//!   drives dispatch via [`OwnedQueueMode::blocking_dispatch`] or
//!   [`OwnedQueueMode::dispatch_pending`]. Best for applications that do
//!   not already have a Wayland event loop.
//!
//! - **[`EmbeddedStateMode`]** — the host owns the `EventQueue` and
//!   embeds a [`WaylandState`] inside its own state struct. The host wires
//!   [`delegate_dispatch!`](wayland_client::delegate_dispatch) so that
//!   backend protocol events are forwarded through [`WaylandProtocol`].
//!   Best for toolkits that need to multiplex many protocol objects on a
//!   single queue.
//!
//! # Flush policy
//!
//! - [`OwnedQueueMode::bootstrap`] flushes internally (it performs a
//!   blocking roundtrip).
//! - [`OwnedQueueMode::blocking_dispatch`] flushes before blocking.
//! - [`OwnedQueueMode::dispatch_pending`] does **not** flush — the caller
//!   must call [`OwnedQueueMode::flush`] separately.
//! - [`WaylandState::request_frame`] emits a `wl_surface.frame` request but
//!   does **not** flush. In owned mode, the next
//!   [`blocking_dispatch`](OwnedQueueMode::blocking_dispatch) flushes
//!   automatically. In non-blocking or embedded mode, the caller must flush.
//! - Future commit-sequencing APIs will always flush after committing.
//!
//! # Frame callback lifecycle
//!
//! The backend converts Wayland `wl_surface.frame` callbacks into
//! [`FrameTick`](subduction_core::timing::FrameTick) values using a
//! pull-based API. The typical per-frame sequence is:
//!
//! 1. **Request** — call [`request_frame`](WaylandState::request_frame)
//!    (or [`OwnedQueueMode::request_frame`]) to send a `wl_surface.frame()`
//!    request. This emits a protocol request but does **not** flush (see
//!    [Flush policy](#flush-policy) above).
//! 2. **Dispatch** — drive the event loop
//!    ([`OwnedQueueMode::blocking_dispatch`],
//!    [`OwnedQueueMode::dispatch_pending`], or host dispatch in embedded
//!    mode) until the compositor delivers the `wl_callback.done` event.
//! 3. **Poll** — call [`poll_tick`](WaylandState::poll_tick) (or
//!    [`OwnedQueueMode::poll_tick`]) in a loop until it returns `None` to
//!    drain all queued ticks.
//! 4. **Process** — for each tick, compute present hints, build and
//!    evaluate the frame, then commit. The next callback is typically
//!    requested as part of the commit sequence; until that is in place,
//!    call `request_frame` again after consuming the tick.
//!
//! Always dispatch before polling: ticks are enqueued by dispatch handlers,
//! so `poll_tick` will not return anything new until dispatch has run.
//!
//! ## One callback in flight
//!
//! Only one frame callback may be in flight at a time. Calling
//! `request_frame` while a callback is pending returns
//! [`RequestFrameError::AlreadyInFlight`]. The in-flight flag is cleared
//! when the `wl_callback.done` event is dispatched, at which point the
//! host may request the next callback.
//!
//! ## Callback pause behaviour
//!
//! Compositors stop delivering frame callbacks when the surface is
//! occluded, minimised, or otherwise not visible. This is normal Wayland
//! behaviour — the tick stream will stall until the surface becomes visible
//! again. Hosts should handle tick starvation gracefully: idle, apply a
//! timeout, or fall back to a timer-based tick source.
//!
//! ## Callback ownership (`FrameCallbackData`)
//!
//! The backend attaches [`FrameCallbackData`] as user data to every
//! `wl_callback` it creates via `request_frame`. This marker type
//! distinguishes backend-issued frame callbacks from any host or toolkit
//! callbacks that share the same event queue.
//!
//! Embedded-mode hosts that create their own `wl_callback` objects must use
//! different user data (for example `()`) to avoid dispatch conflicts. The
//! backend's `Dispatch<WlCallback, FrameCallbackData, D>` impl will only
//! fire for callbacks carrying `FrameCallbackData`.

mod event_loop;
mod hints;
mod output_registry;
mod presentation;
mod protocol;
mod queue;
mod tick;
mod time;

pub use event_loop::{
    EmbeddedStateMode, OwnedQueueMode, RequestFrameError, SetSurfaceError, WaylandState,
};
pub use hints::compute_present_hints;
pub use presentation::{PresentEvent, PresentEventQueue, SubmissionId};
pub use protocol::{Capabilities, FrameCallbackData, OutputGlobalData, WaylandProtocol};
pub use subduction_core::backend::Presenter;
pub use time::{Clock, now, timebase};
