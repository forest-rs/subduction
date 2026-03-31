// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `IDCompositionSurface` presenter for GPU-rendered content.
//!
//! Draw with [`begin_draw`](DCompSurfacePresenter::begin_draw) /
//! [`end_draw`](DCompSurfacePresenter::end_draw), or use
//! [`as_raw`](DCompSurfacePresenter::as_raw) for external renderers.

use core::ffi::c_void;

use windows::Win32::Foundation::{POINT, RECT};
use windows::Win32::Graphics::DirectComposition::{
    IDCompositionDevice, IDCompositionSurface, IDCompositionVisual,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use windows::Win32::Graphics::Dxgi::IDXGISurface;
use windows_core::Interface;

/// Manages an `IDCompositionSurface` for GPU-rendered content.
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
    /// Creates a surface (`B8G8R8A8_UNORM`, premultiplied alpha).
    pub fn new(
        device: &IDCompositionDevice,
        width: u32,
        height: u32,
    ) -> windows::core::Result<Self> {
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

    /// Returns the underlying `IDCompositionSurface`.
    #[must_use]
    pub fn surface(&self) -> &IDCompositionSurface {
        &self.surface
    }

    /// Recreate at a new size (requires re-attach via [`attach_to`](Self::attach_to)).
    pub fn resize(
        &mut self,
        device: &IDCompositionDevice,
        width: u32,
        height: u32,
    ) -> windows::core::Result<()> {
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

    /// Begin drawing. Returns the `IDXGISurface` and pixel offset.
    /// Must be paired with [`end_draw`](Self::end_draw).
    pub fn begin_draw(
        &self,
        update_rect: Option<&RECT>,
    ) -> windows::core::Result<(IDXGISurface, POINT)> {
        let mut offset = POINT::default();
        let raw_rect = update_rect.map(|r| r as *const RECT);
        let dxgi_surface: IDXGISurface =
            unsafe { self.surface.BeginDraw(raw_rect, &mut offset)? };
        Ok((dxgi_surface, offset))
    }

    /// End a draw operation started with [`begin_draw`](Self::begin_draw).
    pub fn end_draw(&self) -> windows::core::Result<()> {
        unsafe { self.surface.EndDraw() }
    }

    /// Attach as content on a visual (`SetContent`).
    pub fn attach_to(&self, visual: &IDCompositionVisual) -> windows::core::Result<()> {
        unsafe { visual.SetContent(&self.surface) }
    }

    /// Raw pointer to the `IDCompositionSurface` (valid for `self`'s lifetime).
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
