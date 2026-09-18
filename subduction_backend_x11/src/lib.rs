// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! X11 backend for subduction.
//!
//! The backend assumes a server with the Present extension. It presents
//! directly to a top-level window and reads that window's vblank clock
//! through Present; a compositing manager is not required.
//!
//! - [`X11PresentSource`] turns a window's Present completions into
//!   `frameclock` ticks and per-submission feedback. It is pure bookkeeping
//!   over [`frameclock_x11`] and references no `x11rb` type, so any event loop
//!   can drive it.
//! - [`X11Compositor`] is the presenting [`Presenter`]: it owns the X11 window
//!   and a wgpu surface, and composites the subduction layer tree onto it with
//!   `subduction_backend_wgpu`. It is available on non-Apple Unix targets.
//!
//! The compositor presents through wgpu's own swapchain. Per-pixmap Present
//! feedback only flows to a presenter that submits `PresentPixmap` itself, so
//! in this configuration the [`X11PresentSource`] feedback queue stays empty.

mod present_source;

pub use frameclock_x11::{PresentEvent, SubmissionId};
pub use present_source::X11PresentSource;
pub use subduction_core::backend::Presenter;

#[cfg(all(unix, not(target_os = "macos")))]
mod compositor;

#[cfg(all(unix, not(target_os = "macos")))]
pub use compositor::{X11BackendError, X11Compositor, X11WindowConfig};
