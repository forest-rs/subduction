// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `IDCompositionSurface` presenter for GPU-rendered content.
//!
//! Provides [`DCompSurfacePresenter`], which manages an
//! `IDCompositionSurface` for use with Direct3D rendering pipelines.
//! Two usage modes are supported:
//!
//! - **Backend-owned**: create the surface via
//!   [`DCompSurfacePresenter::new`], draw into it with [`begin_draw`] /
//!   [`end_draw`], and attach it to a visual with [`attach_to`].
//! - **External renderer**: call [`as_raw`](DCompSurfacePresenter::as_raw)
//!   to obtain a raw pointer for custom integration.
//!
//! [`begin_draw`]: DCompSurfacePresenter::begin_draw
//! [`end_draw`]: DCompSurfacePresenter::end_draw

use core::ffi::c_void;

use windows::Win32::Foundation::{POINT, RECT};
use windows::Win32::Graphics::DirectComposition::{
    IDCompositionDevice, IDCompositionSurface, IDCompositionVisual,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use windows::Win32::Graphics::Dxgi::IDXGISurface;
use windows_core::Interface;

/// Manages an `IDCompositionSurface` for GPU-rendered content.
///
/// This is the minimal presenter needed for integration with Direct3D
/// rendering. It owns the `IDCompositionSurface` and provides access for
/// drawing and attaching to visuals.
///
/// # Example
///
/// ```ignore
/// let surface = DCompSurfacePresenter::new(dcomp_device, 256, 256)?;
/// let (dxgi_surface, offset) = surface.begin_draw(None)?;
/// // ... render into dxgi_surface via D3D11 ...
/// surface.end_draw()?;
/// surface.attach_to(visual)?;
/// ```
pub struct DCompSurfacePresenter {
    surface: IDCompositionSurface,
    width: u32,
    height: u32,
}

impl std::fmt::Debug for DCompSurfacePresenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DCompSurfacePresenter")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl DCompSurfacePresenter {
    /// Creates a new presenter with a fresh `IDCompositionSurface`.
    ///
    /// The surface is created with `DXGI_FORMAT_B8G8R8A8_UNORM` and
    /// premultiplied alpha. Use [`begin_draw`](Self::begin_draw) /
    /// [`end_draw`](Self::end_draw) to render content.
    pub fn new(
        device: &IDCompositionDevice,
        width: u32,
        height: u32,
    ) -> windows::core::Result<Self> {
        // SAFETY: COM call to create a composition surface.
        let surface = unsafe {
            device.CreateSurface(
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_ALPHA_MODE_PREMULTIPLIED,
            )?
        };
        Ok(Self {
            surface,
            width,
            height,
        })
    }

    /// Returns a reference to the underlying `IDCompositionSurface`.
    #[must_use]
    pub fn surface(&self) -> &IDCompositionSurface {
        &self.surface
    }

    /// Recreates the surface at a new size.
    ///
    /// `IDCompositionSurface` does not support in-place resize, so this
    /// creates a new surface and replaces the old one. Any previously
    /// attached visual must be re-attached via [`attach_to`](Self::attach_to).
    pub fn resize(
        &mut self,
        device: &IDCompositionDevice,
        width: u32,
        height: u32,
    ) -> windows::core::Result<()> {
        // SAFETY: COM call to create a composition surface.
        let surface = unsafe {
            device.CreateSurface(
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_ALPHA_MODE_PREMULTIPLIED,
            )?
        };
        self.surface = surface;
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Begins a draw operation on the surface.
    ///
    /// Returns the underlying `IDXGISurface` and the pixel offset within
    /// it where drawing should start. Pass `None` to update the entire
    /// surface.
    ///
    /// Must be paired with [`end_draw`](Self::end_draw).
    pub fn begin_draw(
        &self,
        update_rect: Option<&RECT>,
    ) -> windows::core::Result<(IDXGISurface, POINT)> {
        let mut offset = POINT::default();
        let raw_rect = update_rect.map(|r| r as *const RECT);
        // SAFETY: COM call to begin drawing. The returned IDXGISurface is
        // valid until `EndDraw` is called. The raw pointer, if provided,
        // is derived from a valid reference that outlives this call.
        let dxgi_surface: IDXGISurface =
            unsafe { self.surface.BeginDraw(raw_rect, &mut offset)? };
        Ok((dxgi_surface, offset))
    }

    /// Ends a draw operation previously started with [`begin_draw`](Self::begin_draw).
    pub fn end_draw(&self) -> windows::core::Result<()> {
        // SAFETY: COM call to finalize the draw operation.
        unsafe { self.surface.EndDraw() }
    }

    /// Attaches this surface as content on a `DirectComposition` visual.
    pub fn attach_to(&self, visual: &IDCompositionVisual) -> windows::core::Result<()> {
        // SAFETY: COM call to set the visual's content.
        unsafe { visual.SetContent(&self.surface) }
    }

    /// Returns a raw pointer to the `IDCompositionSurface` for use with
    /// external renderers.
    ///
    /// The returned pointer is valid for the lifetime of this presenter.
    #[must_use]
    pub fn as_raw(&self) -> *mut c_void {
        self.surface.as_raw()
    }

    /// Returns the current width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the current height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}
