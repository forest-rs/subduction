// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `CAMetalLayer` presenter for GPU-rendered content.
//!
//! Provides [`MetalLayerPresenter`], which manages a `CAMetalLayer` for use
//! with Metal-based rendering pipelines. Two usage modes are supported:
//!
//! - **Backend-owned**: create the layer via [`MetalLayerPresenter::new`] and
//!   attach it to a view hierarchy.
//! - **External renderer**: call [`as_raw`](MetalLayerPresenter::as_raw) to
//!   obtain a raw pointer for wgpu's `create_surface_from_layer()` or similar.

use core::ffi::c_void;
use core::fmt;

use objc2::rc::Retained;
use objc2_core_foundation::CGSize;
use objc2_core_graphics::CGColor;
use objc2_quartz_core::CAMetalLayer;
use subduction_core::output::{Backdrop, Color};

/// Manages a `CAMetalLayer` for GPU-rendered content.
///
/// This is the minimal presenter needed for integration with external
/// renderers like wgpu or Vello. It owns the `CAMetalLayer` and provides
/// access for configuration and raw pointer extraction.
///
/// # Example
///
/// ```ignore
/// let presenter = MetalLayerPresenter::new();
/// presenter.set_drawable_size(1920.0, 1080.0);
///
/// // For wgpu integration:
/// let raw = presenter.as_raw();
/// // unsafe { wgpu::Instance::create_surface_from_layer(raw) }
/// ```
pub struct MetalLayerPresenter {
    metal_layer: Retained<CAMetalLayer>,
}

impl fmt::Debug for MetalLayerPresenter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalLayerPresenter")
            .field("drawable_size", &self.metal_layer.drawableSize())
            .finish_non_exhaustive()
    }
}

impl Default for MetalLayerPresenter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalLayerPresenter {
    /// Creates a new presenter with a fresh `CAMetalLayer`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metal_layer: CAMetalLayer::new(),
        }
    }

    /// Returns a reference to the underlying `CAMetalLayer` for configuration.
    ///
    /// Use this to set properties like pixel format, device, colorspace, etc.
    #[must_use]
    pub fn layer(&self) -> &CAMetalLayer {
        &self.metal_layer
    }

    /// Sets the drawable size of the `CAMetalLayer`.
    pub fn set_drawable_size(&self, width: f64, height: f64) {
        self.metal_layer.setDrawableSize(CGSize::new(width, height));
    }

    /// Returns a raw pointer to the `CAMetalLayer` for use with external
    /// renderers (wgpu, Vello, etc.).
    ///
    /// The returned pointer is valid for the lifetime of this presenter.
    #[must_use]
    pub fn as_raw(&self) -> *mut c_void {
        let ptr: *const CAMetalLayer = &*self.metal_layer;
        ptr as *mut c_void
    }

    /// Updates the backdrop policy of the underlying metal layer.
    pub fn set_backdrop(&mut self, backdrop: Backdrop) {
        match backdrop {
            Backdrop::None => self.metal_layer.setBackgroundColor(None),
            Backdrop::Color(color) => {
                let [r, g, b, a] = color.components;
                let cg_color = CGColor::new_generic_rgb(
                    f64::from(r),
                    f64::from(g),
                    f64::from(b),
                    f64::from(a),
                );
                self.metal_layer.setBackgroundColor(Some(&cg_color));
            }
        }
    }

    /// Updates the backdrop to a solid color.
    pub fn set_backdrop_color(&mut self, color: Color) {
        self.set_backdrop(Backdrop::Color(color));
    }

    /// Removes any explicit backdrop.
    pub fn remove_backdrop(&mut self) {
        self.set_backdrop(Backdrop::None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_valid_layer() {
        let presenter = MetalLayerPresenter::new();
        assert!(!presenter.as_raw().is_null());
    }

    #[test]
    fn set_drawable_size_updates() {
        let presenter = MetalLayerPresenter::new();
        presenter.set_drawable_size(1920.0, 1080.0);
        let size = presenter.layer().drawableSize();
        assert!(
            (size.width - 1920.0).abs() < f64::EPSILON,
            "width should be 1920"
        );
        assert!(
            (size.height - 1080.0).abs() < f64::EPSILON,
            "height should be 1080"
        );
    }
}
