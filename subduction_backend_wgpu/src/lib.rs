// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! wgpu fallback compositor backend for subduction.
//!
//! For platforms without a system compositor, this crate provides a GPU-based
//! fallback that allocates a wgpu texture per layer, lets the app render
//! content into each texture, and composites all visible layers (with
//! transforms, opacity, and scissor clips) into a final output surface.
//!
//! See [`WgpuPresenter`] for usage.

mod pipeline;
mod presenter;
mod shader;

pub use presenter::WgpuPresenter;
