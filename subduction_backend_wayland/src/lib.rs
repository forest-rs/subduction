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
mod hints;
mod presentation;
mod queue;
mod tick;
mod time;

pub use event_loop::{EmbeddedStateMode, OwnedQueueMode, WaylandState};
pub use hints::compute_present_hints;
pub use presentation::{PresentEvent, PresentEventQueue, SubmissionId};
pub use subduction_core::backend::Presenter;
pub use time::{Clock, now, timebase};
