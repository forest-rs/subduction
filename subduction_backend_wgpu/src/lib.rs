// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! wgpu fallback compositor backend for subduction.
//!
//! For platforms without a system compositor, this crate provides a GPU-based
//! fallback that allocates a wgpu texture per [`SurfaceId`], lets the app render
//! content into each surface texture, and composites all visible attached
//! surfaces through the layer tree (with transforms, opacity, and scissor clips)
//! into a final output surface.
//! The output surface format can differ from the format used for presenter-
//! owned surface textures.
//!
//! [`LayerRoot`] describes the final compositing target, while
//! [`WgpuPresenter`] owns per-surface textures and composites into that root.
//!
//! [`SurfaceId`]: subduction_core::layer::SurfaceId

mod pipeline;
mod presenter;
mod shader;

pub use presenter::{LayerRoot, WgpuPresenter, WgpuPresenterConfig, WgpuSurfaceTarget};
pub use subduction_core::backend::Presenter;
