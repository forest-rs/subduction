// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland backend for subduction.
//!
//! This crate will provide integration with Wayland compositing:
//!
//! - Frame callback tick source (pull-based, pacing-only)
//! - Optional `wp_presentation` for actual present time feedback
//! - `wl_surface` commit presenter

mod event_loop;
mod presentation;
mod queue;
mod tick;

pub use event_loop::{EmbeddedStateMode, OwnedQueueMode, WaylandState};
pub use presentation::{PresentEvent, PresentEventQueue, SubmissionId};
pub use subduction_core::backend::Presenter;
