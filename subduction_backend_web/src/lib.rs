// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Web backend for subduction.
//!
//! This crate provides integration with browser APIs:
//!
//! - [`LayerRoot`]: root DOM container for a scene
//! - [`DomPresenter`]: DOM element management
//!
//! Browser frame timing lives in `frameclock_web`. Use that crate for
//! `requestAnimationFrame` ticks and retained `frameclock` driver integration.

#![no_std]

extern crate alloc;

mod presenter;

pub use presenter::{DomPresenter, LayerRoot};
pub use subduction_core::backend::Presenter;
