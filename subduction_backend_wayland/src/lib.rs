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
//! - Future commit-sequencing APIs will always flush after committing.

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
pub use protocol::{Capabilities, OutputGlobalData, WaylandProtocol};
pub use subduction_core::backend::Presenter;
pub use time::{Clock, now, timebase};
