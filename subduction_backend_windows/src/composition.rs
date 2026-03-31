// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `DirectComposition` visual tree manager.
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

use subduction_core::time::HostTime;

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct2D::Common::D2D_RECT_F;
use windows::Win32::Graphics::DirectComposition::*;
use windows::Win32::Graphics::Dxgi::IDXGIDevice2;
use windows::core::Result;
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
    /// Cached blur effect — reused across blur updates.
    cached_blur: Option<IDCompositionGaussianBlurEffect>,
}

/// Which property an animation targets (for completion snapping).
#[derive(Debug, Clone)]
pub enum AnimationProperty {
    /// Opacity animation.
    Opacity {
        /// Final value.
        target: f32,
    },
    /// Offset animation.
    Offset {
        /// Final X.
        target_x: f32,
        /// Final Y.
        target_y: f32,
    },
}

/// A pending `DComp` animation with timer-based completion tracking.
#[derive(Debug, Clone)]
pub struct PendingAnimation {
    /// Target layer.
    pub layer_id: LayerId,
    /// Animated property.
    pub property: AnimationProperty,
    /// Absolute time (seconds) when the animation completes.
    pub end_time: f64,
}

/// `DirectComposition` visual tree manager.
///
/// Layers are property-only visuals. Applications attach GPU content
/// via [`visual`](Self::visual) + `SetContent`.
pub struct CompositionManager {
    device: IDCompositionDevice,
    #[expect(
        dead_code,
        reason = "must be kept alive for the lifetime of the composition target"
    )]
    target: IDCompositionTarget,
    root_visual: IDCompositionVisual,
    layers: Vec<Option<CompositionLayer>>,
    free_list: Vec<usize>,
    /// Lazily cached `IDCompositionDevice3` for effects (Windows 10 1607+).
    device3: Option<IDCompositionDevice3>,
    /// Active `DComp` animations awaiting completion.
    active_animations: Vec<PendingAnimation>,
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
    pub fn with_device(dxgi_device: &IDXGIDevice2, hwnd: HWND) -> Result<Self> {
        let device: IDCompositionDevice = unsafe { DCompositionCreateDevice(dxgi_device)? };
        Self::from_device(device, hwnd)
    }

    /// Create a composition manager without a DXGI device (property-only visuals).
    pub fn new(hwnd: HWND) -> Result<Self> {
        let device: IDCompositionDevice = unsafe { DCompositionCreateDevice(None)? };
        Self::from_device(device, hwnd)
    }

    /// Create a composition manager sharing an existing device (multi-window).
    pub fn for_window(device: &IDCompositionDevice, hwnd: HWND) -> Result<Self> {
        Self::from_device(device.clone(), hwnd)
    }

    fn from_device(device: IDCompositionDevice, hwnd: HWND) -> Result<Self> {
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
            device3: None,
            active_animations: Vec::new(),
        })
    }

    // ── Layer lifecycle ────────────────────────────────────────

    /// Create a layer and attach it to `parent` (or the root if `None`).
    pub fn create_layer(&mut self, parent: Option<LayerId>) -> Result<LayerId> {
        let visual = unsafe { self.device.CreateVisual()? };
        let parent_visual = self.parent_visual(parent);
        unsafe { parent_visual.AddVisual(&visual, false, None)? };

        let id = self.alloc_slot(CompositionLayer {
            visual,
            effect_group: None,
            transform_3d: None,
            rounded_clip: None,
            cached_blur: None,
        });

        Ok(id)
    }

    /// Destroy a layer: detach from parent and recycle the slot.
    pub fn destroy_layer(
        &mut self,
        id: LayerId,
        parent: Option<LayerId>,
        is_attached: bool,
    ) -> Result<()> {
        if is_attached {
            self.set_visible(id, parent, false)?;
        }
        self.active_animations.retain(|a| a.layer_id != id);
        self.layers[id.0] = None;
        self.free_list.push(id.0);
        Ok(())
    }

    // ── Transforms ─────────────────────────────────────────────

    /// Set a layer's pixel offset (inherits to children).
    pub fn set_offset(&self, id: LayerId, x: f32, y: f32) -> Result<()> {
        let v = &self.layer(id).visual;
        unsafe {
            v.SetOffsetX2(x)?;
            v.SetOffsetY2(y)?;
        }
        Ok(())
    }

    /// Set a 3D transform (column-major 4×4). Does **not** inherit to children.
    pub fn set_transform_3d(&mut self, id: LayerId, matrix: &[[f32; 4]; 4]) -> Result<()> {
        let device2: IDCompositionDevice2 = self.device.cast()?;
        let layer = self.layer_mut(id);

        if layer.effect_group.is_none() {
            let eg = unsafe { device2.CreateEffectGroup()? };
            unsafe { layer.visual.SetEffect(&eg)? };
            layer.effect_group = Some(eg);
        }
        let effect_group = layer.effect_group.as_ref().unwrap();

        let m = windows_numerics::Matrix4x4 {
            M11: matrix[0][0],
            M12: matrix[0][1],
            M13: matrix[0][2],
            M14: matrix[0][3],
            M21: matrix[1][0],
            M22: matrix[1][1],
            M23: matrix[1][2],
            M24: matrix[1][3],
            M31: matrix[2][0],
            M32: matrix[2][1],
            M33: matrix[2][2],
            M34: matrix[2][3],
            M41: matrix[3][0],
            M42: matrix[3][1],
            M43: matrix[3][2],
            M44: matrix[3][3],
        };

        if layer.transform_3d.is_none() {
            let t3d = unsafe { device2.CreateMatrixTransform3D()? };
            unsafe { effect_group.SetTransform3D(&t3d)? };
            layer.transform_3d = Some(t3d);
        }

        let t3d = layer.transform_3d.as_ref().unwrap();
        unsafe { t3d.SetMatrix(&m)? };
        Ok(())
    }

    /// Clear a layer's 3D transform (revert to identity).
    pub fn clear_transform_3d(&mut self, id: LayerId) -> Result<()> {
        let layer = self.layer_mut(id);
        if layer.transform_3d.is_some() {
            layer.transform_3d = None;
            if let Some(eg) = &layer.effect_group {
                unsafe { eg.SetTransform3D(None)? };
            }
        }
        Ok(())
    }

    // ── Opacity ────────────────────────────────────────────────

    /// Set a layer's opacity (0.0–1.0).
    pub fn set_opacity(&self, id: LayerId, opacity: f32) -> Result<()> {
        let visual3: IDCompositionVisual3 = self.layer(id).visual.cast()?;
        unsafe { visual3.SetOpacity2(opacity) }
    }

    // ── Clips ──────────────────────────────────────────────────

    /// Set an axis-aligned clip rectangle.
    pub fn set_clip(
        &mut self,
        id: LayerId,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    ) -> Result<()> {
        self.layer_mut(id).rounded_clip = None;
        let rect = D2D_RECT_F {
            left,
            top,
            right,
            bottom,
        };
        unsafe { self.layer(id).visual.SetClip2(&rect) }
    }

    /// Set a rounded-rectangle clip with per-corner radii.
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
            let clip = unsafe { self.device.CreateRectangleClip()? };
            let layer = self.layer_mut(id);
            unsafe { layer.visual.SetClip(&clip)? };
            layer.rounded_clip = Some(clip);
        }

        let clip = self.layer(id).rounded_clip.as_ref().unwrap();
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

    /// Remove any clip from a layer.
    pub fn clear_clip(&mut self, id: LayerId) -> Result<()> {
        self.layer_mut(id).rounded_clip = None;
        let clip: Option<&IDCompositionClip> = None;
        unsafe { self.layer(id).visual.SetClip(clip) }
    }

    // ── Visibility ─────────────────────────────────────────────

    /// Show or hide a layer (DWM-level attach/detach, zero GPU cost).
    pub fn set_visible(&self, id: LayerId, parent: Option<LayerId>, visible: bool) -> Result<()> {
        let parent_visual = self.parent_visual(parent);
        let layer = self.layer(id);
        if visible {
            unsafe { parent_visual.AddVisual(&layer.visual, false, None)? };
        } else {
            unsafe { parent_visual.RemoveVisual(&layer.visual)? };
        }
        Ok(())
    }

    /// Move a layer to a new parent.
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
            unsafe { old_pv.RemoveVisual(visual)? };
        }

        if is_attached {
            let new_pv = self.parent_visual(new_parent);
            unsafe { new_pv.AddVisual(visual, false, None)? };
        }

        Ok(())
    }

    // ── Commit ─────────────────────────────────────────────────

    /// Commit all pending changes atomically.
    pub fn commit(&self) -> Result<()> {
        unsafe { self.device.Commit() }
    }

    // ── Scroll offset ───────────────────────────────────────

    /// DWM-level scroll offset (positive = content moves up/left).
    pub fn set_scroll_offset(&self, id: LayerId, dx: f32, dy: f32) -> Result<()> {
        let v = &self.layer(id).visual;
        unsafe {
            v.SetOffsetX2(-dx)?;
            v.SetOffsetY2(-dy)?;
        }
        Ok(())
    }

    // ── Effects (IDCompositionDevice3, Windows 10 1607+) ──

    /// Lazily acquire `IDCompositionDevice3`.
    fn device3(&mut self) -> Result<IDCompositionDevice3> {
        if self.device3.is_none() {
            self.device3 = Some(self.device.cast()?);
        }
        Ok(self.device3.as_ref().unwrap().clone())
    }

    /// Apply a Gaussian blur (`sigma` <= 0 removes it).
    pub fn set_blur(&mut self, id: LayerId, sigma: f32) -> Result<()> {
        let device3 = self.device3()?;
        let layer = self.layer_mut(id);

        if sigma <= 0.0 {
            layer.cached_blur = None;
            let visual3: IDCompositionVisual3 = layer.visual.cast()?;
            if let Some(eg) = &layer.effect_group {
                unsafe {
                    visual3.SetEffect(eg)?;
                }
            } else {
                unsafe {
                    visual3.SetEffect(None)?;
                }
            }
            return Ok(());
        }

        if layer.cached_blur.is_none() {
            let blur = unsafe { device3.CreateGaussianBlurEffect()? };
            layer.cached_blur = Some(blur);
        }
        let blur = layer.cached_blur.as_ref().unwrap();
        unsafe {
            blur.SetStandardDeviation2(sigma)?;
        }

        let visual3: IDCompositionVisual3 = layer.visual.cast()?;
        unsafe {
            visual3.SetEffect(blur)?;
        }
        Ok(())
    }

    /// Apply a saturation effect (0.0 = grayscale, 1.0 = identity).
    pub fn set_saturation(&mut self, id: LayerId, amount: f32) -> Result<()> {
        let device3 = self.device3()?;
        let effect = unsafe { device3.CreateSaturationEffect()? };
        unsafe {
            effect.SetSaturation2(amount)?;
        }

        let visual3: IDCompositionVisual3 = self.layer(id).visual.cast()?;
        unsafe {
            visual3.SetEffect(&effect)?;
        }
        Ok(())
    }

    /// Apply a 5×4 color matrix effect (row-major).
    pub fn set_color_matrix(&mut self, id: LayerId, matrix: &[f32; 20]) -> Result<()> {
        let device3 = self.device3()?;
        let effect = unsafe { device3.CreateColorMatrixEffect()? };
        for row in 0..5 {
            for col in 0..4 {
                unsafe {
                    effect.SetMatrixElement2(row, col, matrix[row as usize * 4 + col as usize])?;
                }
            }
        }

        let visual3: IDCompositionVisual3 = self.layer(id).visual.cast()?;
        unsafe {
            visual3.SetEffect(&effect)?;
        }
        Ok(())
    }

    /// Apply a brightness effect with white/black point curves.
    pub fn set_brightness(
        &mut self,
        id: LayerId,
        white: (f32, f32),
        black: (f32, f32),
    ) -> Result<()> {
        let device3 = self.device3()?;
        let effect = unsafe { device3.CreateBrightnessEffect()? };
        unsafe {
            effect.SetWhitePointX2(white.0)?;
            effect.SetWhitePointY2(white.1)?;
            effect.SetBlackPointX2(black.0)?;
            effect.SetBlackPointY2(black.1)?;
        }

        let visual3: IDCompositionVisual3 = self.layer(id).visual.cast()?;
        unsafe {
            visual3.SetEffect(&effect)?;
        }
        Ok(())
    }

    /// Remove all effects from a layer.
    pub fn clear_effects(&mut self, id: LayerId) -> Result<()> {
        let layer = self.layer_mut(id);
        layer.cached_blur = None;
        let visual3: IDCompositionVisual3 = layer.visual.cast()?;
        unsafe {
            visual3.SetEffect(None)?;
        }
        layer.effect_group = None;
        layer.transform_3d = None;
        Ok(())
    }

    // ── Animations (DComp-driven, zero per-frame app cost) ──

    /// Animate opacity linearly. Call [`tick_animations`](Self::tick_animations) to detect completion.
    pub fn animate_opacity(
        &mut self,
        id: LayerId,
        from: f32,
        to: f32,
        duration_s: f64,
        now: f64,
    ) -> Result<()> {
        let animation = unsafe { self.device.CreateAnimation()? };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Duration truncated from f64 to f32 for DComp slope"
        )]
        let slope = (to - from) / duration_s as f32;
        unsafe {
            animation.AddCubic(0.0, from, slope, 0.0, 0.0)?;
            animation.End(duration_s, to)?;
        }

        let visual3: IDCompositionVisual3 = self.layer(id).visual.cast()?;
        unsafe {
            visual3.SetOpacity(&animation)?;
        }

        self.active_animations.push(PendingAnimation {
            layer_id: id,
            property: AnimationProperty::Opacity { target: to },
            end_time: now + duration_s,
        });
        Ok(())
    }

    /// Animate offset linearly.
    pub fn animate_offset(
        &mut self,
        id: LayerId,
        from: (f32, f32),
        to: (f32, f32),
        duration_s: f64,
        now: f64,
    ) -> Result<()> {
        let anim_x = unsafe { self.device.CreateAnimation()? };
        let anim_y = unsafe { self.device.CreateAnimation()? };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Duration truncated from f64 to f32 for DComp slope"
        )]
        let slope_x = (to.0 - from.0) / duration_s as f32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Duration truncated from f64 to f32 for DComp slope"
        )]
        let slope_y = (to.1 - from.1) / duration_s as f32;
        unsafe {
            anim_x.AddCubic(0.0, from.0, slope_x, 0.0, 0.0)?;
            anim_x.End(duration_s, to.0)?;
            anim_y.AddCubic(0.0, from.1, slope_y, 0.0, 0.0)?;
            anim_y.End(duration_s, to.1)?;
        }

        let visual = &self.layer(id).visual;
        unsafe {
            visual.SetOffsetX(&anim_x)?;
            visual.SetOffsetY(&anim_y)?;
        }

        self.active_animations.push(PendingAnimation {
            layer_id: id,
            property: AnimationProperty::Offset {
                target_x: to.0,
                target_y: to.1,
            },
            end_time: now + duration_s,
        });
        Ok(())
    }

    /// Animate a single offset axis with linear interpolation.
    pub fn animate_offset_axis(
        &mut self,
        id: LayerId,
        is_x: bool,
        from: f32,
        to: f32,
        duration_s: f64,
        now: f64,
    ) -> Result<()> {
        let animation = unsafe { self.device.CreateAnimation()? };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Duration truncated from f64 to f32 for DComp slope"
        )]
        let slope = (to - from) / duration_s as f32;
        unsafe {
            animation.AddCubic(0.0, from, slope, 0.0, 0.0)?;
            animation.End(duration_s, to)?;
        }

        let visual = &self.layer(id).visual;
        unsafe {
            if is_x {
                visual.SetOffsetX(&animation)?;
            } else {
                visual.SetOffsetY(&animation)?;
            }
        }

        self.active_animations.push(PendingAnimation {
            layer_id: id,
            property: if is_x {
                AnimationProperty::Offset {
                    target_x: to,
                    target_y: 0.0,
                }
            } else {
                AnimationProperty::Offset {
                    target_x: 0.0,
                    target_y: to,
                }
            },
            end_time: now + duration_s,
        });
        Ok(())
    }

    /// Animate a single offset axis with a cubic polynomial (for deceleration).
    ///
    /// `value(t) = constant + linear*t + quadratic*t^2`
    /// At `t = t_stop`, snaps to `final_value`.
    pub fn animate_offset_cubic(
        &mut self,
        id: LayerId,
        is_x: bool,
        constant: f32,
        linear: f32,
        quadratic: f32,
        t_stop: f64,
        final_value: f32,
        now: f64,
    ) -> Result<()> {
        let animation = unsafe { self.device.CreateAnimation()? };
        unsafe {
            animation.AddCubic(0.0, constant, linear, quadratic, 0.0)?;
            animation.End(t_stop, final_value)?;
        }

        let visual = &self.layer(id).visual;
        unsafe {
            if is_x {
                visual.SetOffsetX(&animation)?;
            } else {
                visual.SetOffsetY(&animation)?;
            }
        }

        self.active_animations.push(PendingAnimation {
            layer_id: id,
            property: if is_x {
                AnimationProperty::Offset {
                    target_x: final_value,
                    target_y: 0.0,
                }
            } else {
                AnimationProperty::Offset {
                    target_x: 0.0,
                    target_y: final_value,
                }
            },
            end_time: now + t_stop,
        });
        Ok(())
    }

    /// Drain completed animations. Returns the count.
    pub fn tick_animations(&mut self, now: f64) -> usize {
        let before = self.active_animations.len();
        self.active_animations.retain(|anim| now < anim.end_time);
        before - self.active_animations.len()
    }

    /// Whether any animations are currently active.
    pub fn has_active_animations(&self) -> bool {
        !self.active_animations.is_empty()
    }

    /// Number of active animations.
    pub fn animation_count(&self) -> usize {
        self.active_animations.len()
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

    /// Returns the [`IDCompositionDevice`].
    pub fn device(&self) -> &IDCompositionDevice {
        &self.device
    }

    /// Returns DWM composition frame statistics (QPC ticks).
    pub fn frame_statistics(&self) -> Result<DCOMPOSITION_FRAME_STATISTICS> {
        unsafe { self.device.GetFrameStatistics() }
    }

    /// Actual present time of the last DWM composition frame.
    #[expect(
        clippy::cast_sign_loss,
        reason = "QPC values from DWM are always non-negative"
    )]
    pub fn last_present_time(&self) -> Result<HostTime> {
        let stats = self.frame_statistics()?;
        Ok(HostTime(stats.lastFrameTime as u64))
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
