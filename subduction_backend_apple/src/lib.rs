// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Apple backend for subduction.
//!
//! This crate provides composable building blocks for driving a subduction
//! layer tree on Apple platforms (macOS, iOS, tvOS, visionOS):
//!
//! - [`LayerRoot`]: root `CALayer` container for a scene
//! - [`LayerPresenter`]: `CALayer` tree presenter
//! - [`MetalLayerPresenter`]: `CAMetalLayer` presenter
//!
//! Apple display-link timing lives in `frameclock_apple`. Use that crate for
//! `CADisplayLink` / `CVDisplayLink` ticks, Mach host-time helpers, and
//! retained `frameclock` driver integration.

#![no_std]
#![expect(
    unsafe_code,
    reason = "Apple backend requires extensive Objective-C FFI"
)]

extern crate alloc;

mod calayer;
mod cametal;

pub use calayer::{LayerPresenter, LayerRoot};
pub use cametal::MetalLayerPresenter;
pub use subduction_core::backend::Presenter;
