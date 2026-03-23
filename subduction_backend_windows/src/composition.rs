// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! DirectComposition visual tree manager.
//!
//! Manages property-only visuals in the DWM composition tree. Each layer
//! owns an [`IDCompositionVisual`] that can be positioned, clipped,
//! transformed, and have its opacity set. The visuals carry no backing
//! surface — applications attach GPU content via
//! [`CompositionManager::visual`] + `SetContent`.
//!
//! # Visual tree
//!
//! ```text
//! IDCompositionTarget (bound to HWND)
//!   └── Root Visual
//!       ├── Layer A
//!       │   ├── Child A1
//!       │   └── Child A2
//!       └── Layer B
//! ```

use windows::core::Result;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct2D::Common::D2D_RECT_F;
use windows::Win32::Graphics::DirectComposition::*;
use windows::Win32::Graphics::Dxgi::IDXGIDevice2;
use windows_core::Interface;

/// Opaque handle to a layer in the composition tree.
///
/// Indices are reused via a free list after [`CompositionManager::destroy_layer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId(pub(crate) usize);

/// Per-layer state in the composition tree.
struct CompositionLayer {
    visual: IDCompositionVisual,
    /// Cached effect group — reused across transform/opacity updates
    /// to avoid allocating a new COM object every frame.
    effect_group: Option<IDCompositionEffectGroup>,
    /// Cached 3D transform object — reused when updating the matrix.
    transform_3d: Option<IDCompositionMatrixTransform3D>,
    /// Cached rounded-rectangle clip — reused across clip updates.
    rounded_clip: Option<IDCompositionRectangleClip>,
}

impl std::fmt::Debug for CompositionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositionLayer")
            .field("has_effect_group", &self.effect_group.is_some())
            .field("has_transform_3d", &self.transform_3d.is_some())
            .finish_non_exhaustive()
    }
}

/// DirectComposition visual tree manager.
///
/// Owns the composition device, target, and root visual. Layers are
/// property-only visuals (no swapchain or surface backing). Applications
/// attach GPU content by calling [`visual`](Self::visual) and then
/// `IDCompositionVisual::SetContent`.
pub struct CompositionManager {
    device: IDCompositionDevice,
    #[expect(dead_code, reason = "must be kept alive for the lifetime of the composition target")]
    target: IDCompositionTarget,
    root_visual: IDCompositionVisual,
    layers: Vec<Option<CompositionLayer>>,
    free_list: Vec<usize>,
}

impl std::fmt::Debug for CompositionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositionManager")
            .field("layer_count", &self.layer_count())
            .field("slot_count", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl CompositionManager {
    /// Create a composition manager bound to a window.
    ///
    /// The HWND must have `WS_EX_NOREDIRECTIONBITMAP`.
    ///
    /// `dxgi_device` enables GPU-backed content to be attached to visuals
    /// later. Pass a device obtained from `ID3D11Device::cast::<IDXGIDevice2>()`.
    pub fn with_device(dxgi_device: &IDXGIDevice2, hwnd: HWND) -> Result<Self> {
        // SAFETY: `DCompositionCreateDevice` creates a composition device
        // from the provided DXGI device. The returned device is valid for
        // the lifetime of the DXGI device.
        let device: IDCompositionDevice = unsafe { DCompositionCreateDevice(dxgi_device)? };
        Self::from_device(device, hwnd)
    }

    /// Create a composition manager without a DXGI device.
    ///
    /// Visuals will be property-only (transforms, opacity, clips). GPU
    /// content cannot be attached until a DXGI device is provided
    /// externally via `visual.SetContent(...)`.
    ///
    /// Useful for tests or lightweight composition without D3D11.
    pub fn new(hwnd: HWND) -> Result<Self> {
        // SAFETY: Passing `None` creates a device without a rendering device.
        let device: IDCompositionDevice = unsafe {
            DCompositionCreateDevice(None)?
        };
        Self::from_device(device, hwnd)
    }

    fn from_device(device: IDCompositionDevice, hwnd: HWND) -> Result<Self> {
        // SAFETY: COM calls to set up the composition target and root visual.
        let target = unsafe { device.CreateTargetForHwnd(hwnd, true)? };
        let root_visual = unsafe { device.CreateVisual()? };
        unsafe {
            target.SetRoot(&root_visual)?;
            device.Commit()?;
        }

        Ok(Self {
            device,
            target,
            root_visual,
            layers: Vec::new(),
            free_list: Vec::new(),
        })
    }

    // ── Layer lifecycle ────────────────────────────────────────

    /// Create a property-only layer visual and attach it to a parent.
    ///
    /// If `parent` is `None`, the layer is a child of the root visual.
    pub fn create_layer(&mut self, parent: Option<LayerId>) -> Result<LayerId> {
        // SAFETY: `CreateVisual` creates a new visual in the composition device.
        let visual = unsafe { self.device.CreateVisual()? };

        let parent_visual = self.parent_visual(parent);
        // SAFETY: `AddVisual` attaches the visual to its parent.
        unsafe { parent_visual.AddVisual(&visual, false, None)? };

        let id = self.alloc_slot(CompositionLayer {
            visual,
            effect_group: None,
            transform_3d: None,
            rounded_clip: None,
        });

        Ok(id)
    }

    /// Destroy a layer: detach from parent and return the slot for reuse.
    ///
    /// The caller must provide `parent` and `is_attached` — the manager
    /// does not track these. If the layer has children the caller must
    /// reparent them first.
    pub fn destroy_layer(
        &mut self,
        id: LayerId,
        parent: Option<LayerId>,
        is_attached: bool,
    ) -> Result<()> {
        if is_attached {
            self.set_visible(id, parent, false)?;
        }
        self.layers[id.0] = None;
        self.free_list.push(id.0);
        Ok(())
    }

    // ── Transforms ─────────────────────────────────────────────

    /// Set a layer's pixel offset (translation).
    ///
    /// This inherits through the visual tree — child visuals move with
    /// their parent.
    pub fn set_offset(&self, id: LayerId, x: f32, y: f32) -> Result<()> {
        let v = &self.layer(id).visual;
        // SAFETY: COM property setters on the visual.
        unsafe {
            v.SetOffsetX2(x)?;
            v.SetOffsetY2(y)?;
        }
        Ok(())
    }

    /// Set a 3D transform from a column-major 4×4 matrix.
    ///
    /// Applied via an [`IDCompositionEffectGroup`] — this transforms the
    /// visual's bitmap but does **not** affect child visual positions
    /// (use [`set_offset`](Self::set_offset) for positional inheritance).
    ///
    /// Lazily creates the effect group and transform COM objects; reuses
    /// them on subsequent calls.
    pub fn set_transform_3d(
        &mut self,
        id: LayerId,
        matrix: &[[f32; 4]; 4],
    ) -> Result<()> {
        let device2: IDCompositionDevice2 = self.device.cast()?;
        let layer = self.layer_mut(id);

        // Lazily create the effect group and bind it to the visual once.
        if layer.effect_group.is_none() {
            // SAFETY: COM object creation and property setting.
            let eg = unsafe { device2.CreateEffectGroup()? };
            unsafe { layer.visual.SetEffect(&eg)? };
            layer.effect_group = Some(eg);
        }
        let effect_group = layer.effect_group.as_ref().unwrap();

        let m = windows_numerics::Matrix4x4 {
            M11: matrix[0][0], M12: matrix[0][1], M13: matrix[0][2], M14: matrix[0][3],
            M21: matrix[1][0], M22: matrix[1][1], M23: matrix[1][2], M24: matrix[1][3],
            M31: matrix[2][0], M32: matrix[2][1], M33: matrix[2][2], M34: matrix[2][3],
            M41: matrix[3][0], M42: matrix[3][1], M43: matrix[3][2], M44: matrix[3][3],
        };

        // Lazily create the transform object and bind it once.
        if layer.transform_3d.is_none() {
            // SAFETY: COM object creation and binding.
            let t3d = unsafe { device2.CreateMatrixTransform3D()? };
            unsafe { effect_group.SetTransform3D(&t3d)? };
            layer.transform_3d = Some(t3d);
        }

        let t3d = layer.transform_3d.as_ref().unwrap();
        // SAFETY: `SetMatrix` sets the transform matrix.
        unsafe { t3d.SetMatrix(&m)? };
        Ok(())
    }

    /// Clear a layer's 3D transform, reverting to identity.
    pub fn clear_transform_3d(&mut self, id: LayerId) -> Result<()> {
        let layer = self.layer_mut(id);
        if layer.transform_3d.is_some() {
            layer.transform_3d = None;
            if let Some(eg) = &layer.effect_group {
                // SAFETY: `SetTransform3D(None)` clears the transform.
                unsafe { eg.SetTransform3D(None)? };
            }
        }
        Ok(())
    }

    // ── Opacity ────────────────────────────────────────────────

    /// Set a layer's opacity (0.0–1.0).
    ///
    /// Applied directly on the visual via [`IDCompositionVisual3::SetOpacity2`].
    pub fn set_opacity(&self, id: LayerId, opacity: f32) -> Result<()> {
        let visual3: IDCompositionVisual3 = self.layer(id).visual.cast()?;
        // SAFETY: COM property setter.
        unsafe { visual3.SetOpacity2(opacity) }
    }

    // ── Clips ──────────────────────────────────────────────────

    /// Set an axis-aligned clip rectangle (local coordinates).
    pub fn set_clip(
        &mut self,
        id: LayerId,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    ) -> Result<()> {
        // Drop any cached rounded clip — `SetClip2` with a rect replaces it.
        self.layer_mut(id).rounded_clip = None;
        let rect = D2D_RECT_F { left, top, right, bottom };
        // SAFETY: `SetClip2` sets an axis-aligned clip rectangle.
        unsafe { self.layer(id).visual.SetClip2(&rect) }
    }

    /// Set a rounded-rectangle clip with per-corner radii (local coordinates).
    ///
    /// Uses a cached [`IDCompositionRectangleClip`] — created lazily on
    /// first call and reused afterwards.
    pub fn set_rounded_clip(
        &mut self,
        id: LayerId,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        top_left_radius: f32,
        top_right_radius: f32,
        bottom_right_radius: f32,
        bottom_left_radius: f32,
    ) -> Result<()> {
        let layer = self.layer_mut(id);

        if layer.rounded_clip.is_none() {
            // SAFETY: COM object creation and binding.
            let clip = unsafe { self.device.CreateRectangleClip()? };
            let layer = self.layer_mut(id);
            unsafe { layer.visual.SetClip(&clip)? };
            layer.rounded_clip = Some(clip);
        }

        let clip = self.layer(id).rounded_clip.as_ref().unwrap();
        // SAFETY: COM property setters on the clip object.
        unsafe {
            clip.SetLeft2(left)?;
            clip.SetTop2(top)?;
            clip.SetRight2(right)?;
            clip.SetBottom2(bottom)?;
            clip.SetTopLeftRadiusX2(top_left_radius)?;
            clip.SetTopLeftRadiusY2(top_left_radius)?;
            clip.SetTopRightRadiusX2(top_right_radius)?;
            clip.SetTopRightRadiusY2(top_right_radius)?;
            clip.SetBottomRightRadiusX2(bottom_right_radius)?;
            clip.SetBottomRightRadiusY2(bottom_right_radius)?;
            clip.SetBottomLeftRadiusX2(bottom_left_radius)?;
            clip.SetBottomLeftRadiusY2(bottom_left_radius)?;
        }
        Ok(())
    }

    /// Remove any clip from a layer, restoring it to unclipped.
    pub fn clear_clip(&mut self, id: LayerId) -> Result<()> {
        self.layer_mut(id).rounded_clip = None;
        let clip: Option<&IDCompositionClip> = None;
        // SAFETY: `SetClip(None)` clears the clip.
        unsafe { self.layer(id).visual.SetClip(clip) }
    }

    // ── Visibility ─────────────────────────────────────────────

    /// Show or hide a layer by attaching/detaching its visual.
    ///
    /// Both operations are DWM-level — zero GPU cost.
    pub fn set_visible(
        &self,
        id: LayerId,
        parent: Option<LayerId>,
        visible: bool,
    ) -> Result<()> {
        let parent_visual = self.parent_visual(parent);
        let layer = self.layer(id);
        // SAFETY: `AddVisual`/`RemoveVisual` attach/detach the visual.
        if visible {
            unsafe { parent_visual.AddVisual(&layer.visual, false, None)? };
        } else {
            unsafe { parent_visual.RemoveVisual(&layer.visual)? };
        }
        Ok(())
    }

    /// Move a layer to a new parent.
    ///
    /// The caller must provide the old parent and whether the visual is
    /// currently attached.
    pub fn reparent(
        &self,
        id: LayerId,
        old_parent: Option<LayerId>,
        new_parent: Option<LayerId>,
        is_attached: bool,
    ) -> Result<()> {
        if old_parent == new_parent {
            return Ok(());
        }

        let visual = &self.layer(id).visual;

        if is_attached {
            let old_pv = self.parent_visual(old_parent);
            // SAFETY: `RemoveVisual` detaches from the old parent.
            unsafe { old_pv.RemoveVisual(visual)? };
        }

        if is_attached {
            let new_pv = self.parent_visual(new_parent);
            // SAFETY: `AddVisual` attaches to the new parent.
            unsafe { new_pv.AddVisual(visual, false, None)? };
        }

        Ok(())
    }

    // ── Commit ─────────────────────────────────────────────────

    /// Commit all pending changes atomically.
    ///
    /// The DWM sees all visual tree updates in one transaction — no
    /// tearing between layers.
    pub fn commit(&self) -> Result<()> {
        // SAFETY: `Commit` flushes all pending composition changes.
        unsafe { self.device.Commit() }
    }

    // ── Accessors ──────────────────────────────────────────────

    /// Returns the [`IDCompositionVisual`] for a layer.
    ///
    /// Applications use this to attach GPU content:
    /// ```ignore
    /// let visual = manager.visual(id);
    /// unsafe { visual.SetContent(&swapchain)?; }
    /// ```
    pub fn visual(&self, id: LayerId) -> &IDCompositionVisual {
        &self.layer(id).visual
    }

    /// Returns a reference to the root visual.
    pub fn root_visual(&self) -> &IDCompositionVisual {
        &self.root_visual
    }

    /// Returns a reference to the [`IDCompositionDevice`].
    ///
    /// Applications use this to create surfaces, animations, or other
    /// device-owned objects.
    pub fn device(&self) -> &IDCompositionDevice {
        &self.device
    }

    /// Number of live (non-destroyed) layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len() - self.free_list.len()
    }

    // ── Internal helpers ───────────────────────────────────────

    fn parent_visual(&self, parent: Option<LayerId>) -> &IDCompositionVisual {
        match parent {
            Some(pid) => &self.layer(pid).visual,
            None => &self.root_visual,
        }
    }

    fn alloc_slot(&mut self, layer: CompositionLayer) -> LayerId {
        if let Some(idx) = self.free_list.pop() {
            self.layers[idx] = Some(layer);
            LayerId(idx)
        } else {
            let idx = self.layers.len();
            self.layers.push(Some(layer));
            LayerId(idx)
        }
    }

    fn layer(&self, id: LayerId) -> &CompositionLayer {
        self.layers[id.0]
            .as_ref()
            .expect("access to destroyed layer")
    }

    fn layer_mut(&mut self, id: LayerId) -> &mut CompositionLayer {
        self.layers[id.0]
            .as_mut()
            .expect("access to destroyed layer")
    }
}
