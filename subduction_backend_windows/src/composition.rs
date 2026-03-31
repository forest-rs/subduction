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
    /// Cached rounded-rectangle clip — reused across clip updates.
    rounded_clip: Option<IDCompositionRectangleClip>,
    // Cached effects (chained in this order by `rebuild_effect_chain`)
    blur: Option<IDCompositionGaussianBlurEffect>,
    saturation: Option<IDCompositionSaturationEffect>,
    color_matrix: Option<IDCompositionColorMatrixEffect>,
    brightness: Option<IDCompositionBrightnessEffect>,
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
            rounded_clip: None,
            blur: None,
            saturation: None,
            color_matrix: None,
            brightness: None,
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

    /// Set a 2D affine transform on the visual. Inherits to children through
    /// the visual tree, composing correctly with parent transforms.
    pub fn set_transform(&self, id: LayerId, matrix: &windows_numerics::Matrix3x2) -> Result<()> {
        let v = &self.layer(id).visual;
        unsafe { v.SetTransform2(matrix)? };
        Ok(())
    }

    /// Clear a layer's transform (revert to identity).
    pub fn clear_transform(&self, id: LayerId) -> Result<()> {
        let v = &self.layer(id).visual;
        unsafe { v.SetTransform(None)? };
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
            layer.blur = None;
        } else {
            if layer.blur.is_none() {
                layer.blur = Some(unsafe { device3.CreateGaussianBlurEffect()? });
            }
            unsafe { layer.blur.as_ref().unwrap().SetStandardDeviation2(sigma)? };
        }

        rebuild_effect_chain(layer)
    }

    /// Apply a saturation effect (`amount` == 1.0 is identity; removes it).
    pub fn set_saturation(&mut self, id: LayerId, amount: f32) -> Result<()> {
        let device3 = self.device3()?;
        let layer = self.layer_mut(id);

        if amount == 1.0 {
            layer.saturation = None;
        } else {
            if layer.saturation.is_none() {
                layer.saturation = Some(unsafe { device3.CreateSaturationEffect()? });
            }
            unsafe { layer.saturation.as_ref().unwrap().SetSaturation2(amount)? };
        }

        rebuild_effect_chain(layer)
    }

    /// Apply a 5×4 color matrix effect (row-major).
    pub fn set_color_matrix(&mut self, id: LayerId, matrix: &[f32; 20]) -> Result<()> {
        let device3 = self.device3()?;
        let layer = self.layer_mut(id);

        if layer.color_matrix.is_none() {
            layer.color_matrix = Some(unsafe { device3.CreateColorMatrixEffect()? });
        }
        let effect = layer.color_matrix.as_ref().unwrap();
        for row in 0..5 {
            for col in 0..4 {
                unsafe {
                    effect.SetMatrixElement2(row, col, matrix[row as usize * 4 + col as usize])?;
                }
            }
        }

        rebuild_effect_chain(layer)
    }

    /// Apply a brightness effect with white/black point curves.
    pub fn set_brightness(
        &mut self,
        id: LayerId,
        white: (f32, f32),
        black: (f32, f32),
    ) -> Result<()> {
        let device3 = self.device3()?;
        let layer = self.layer_mut(id);

        if layer.brightness.is_none() {
            layer.brightness = Some(unsafe { device3.CreateBrightnessEffect()? });
        }
        let effect = layer.brightness.as_ref().unwrap();
        unsafe {
            effect.SetWhitePointX2(white.0)?;
            effect.SetWhitePointY2(white.1)?;
            effect.SetBlackPointX2(black.0)?;
            effect.SetBlackPointY2(black.1)?;
        }

        rebuild_effect_chain(layer)
    }

    /// Remove all effects from a layer.
    pub fn clear_effects(&mut self, id: LayerId) -> Result<()> {
        let layer = self.layer_mut(id);
        layer.blur = None;
        layer.saturation = None;
        layer.color_matrix = None;
        layer.brightness = None;
        rebuild_effect_chain(layer)
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

/// Rebuild the effect chain on a layer's visual.
///
/// Active effects are chained in a fixed order (blur → saturation →
/// color\_matrix → brightness) via [`IDCompositionFilterEffect::SetInput`].
/// Only the tail of the chain is set on the visual; `DComp` walks backwards
/// through the `SetInput` links to compose the full pipeline.
fn rebuild_effect_chain(layer: &CompositionLayer) -> Result<()> {
    let visual3: IDCompositionVisual3 = layer.visual.cast()?;

    // Collect active effects in chain order.
    let chain: [Option<IDCompositionFilterEffect>; 4] = [
        layer.blur.as_ref().map(|e| e.cast()).transpose()?,
        layer.saturation.as_ref().map(|e| e.cast()).transpose()?,
        layer.color_matrix.as_ref().map(|e| e.cast()).transpose()?,
        layer.brightness.as_ref().map(|e| e.cast()).transpose()?,
    ];
    let mut active = [None, None, None, None];
    let mut len = 0;
    for effect in chain.into_iter().flatten() {
        active[len] = Some(effect);
        len += 1;
    }

    if len == 0 {
        unsafe { visual3.SetEffect(None)? };
        return Ok(());
    }

    // First effect reads from the visual's own content.
    unsafe { active[0].as_ref().unwrap().SetInput(0, None, 0)?; }

    // Each subsequent effect reads from the previous one's output.
    for i in 1..len {
        unsafe {
            active[i]
                .as_ref()
                .unwrap()
                .SetInput(0, active[i - 1].as_ref().unwrap(), 0)?;
        };
    }

    // Set the tail of the chain on the visual.
    unsafe { visual3.SetEffect(active[len - 1].as_ref().unwrap())?; }
    Ok(())
}
