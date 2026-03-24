// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! [`Presenter`] trait implementation backed by `DirectComposition`.
//!
//! Maps subduction's [`LayerStore`] mutations to `DirectComposition` visual
//! tree operations via [`CompositionManager`].
//!
//! [`LayerStore`]: subduction_core::layer::LayerStore

use subduction_core::backend::Presenter;
use subduction_core::layer::{ClipShape, FrameChanges, LayerStore};

use crate::composition::{CompositionManager, LayerId};

use windows::Win32::Graphics::DirectComposition::IDCompositionVisual;

/// `DirectComposition` presenter for subduction.
///
/// Manages the mapping between subduction's layer tree and
/// `DirectComposition` visuals. Visuals are property-only — applications
/// attach GPU content via [`visual_for`](Self::visual_for).
///
/// # Transforms
///
/// Uses **local** transforms (not world transforms) because
/// `DirectComposition` composes parent transforms automatically through
/// the visual tree hierarchy. Translation goes through `SetOffsetX/Y`
/// (inherits to children), while rotation/scale goes through an
/// `IDCompositionEffectGroup` 3D transform (does **not** inherit).
///
/// # Opacity
///
/// Uses **local** opacity for the same reason — `DComp` multiplies parent
/// opacity into children automatically.
pub struct DCompPresenter {
    composition: CompositionManager,
    /// Maps subduction layer slot index → composition [`LayerId`].
    /// `None` if the slot hasn't been realized as a visual yet.
    layer_map: Vec<Option<LayerId>>,
    /// Tracks last-set parent for each layer (indexed by subduction slot).
    /// Used for topology reconciliation and reparenting.
    layer_parents: Vec<Option<Option<LayerId>>>,
}

impl std::fmt::Debug for DCompPresenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DCompPresenter")
            .field("composition", &self.composition)
            .field(
                "mapped_layers",
                &self.layer_map.iter().filter(|s| s.is_some()).count(),
            )
            .finish_non_exhaustive()
    }
}

impl DCompPresenter {
    /// Create a new presenter wrapping an existing [`CompositionManager`].
    #[must_use]
    pub fn new(composition: CompositionManager) -> Self {
        Self {
            composition,
            layer_map: Vec::new(),
            layer_parents: Vec::new(),
        }
    }

    /// Returns the underlying [`CompositionManager`].
    pub fn composition(&self) -> &CompositionManager {
        &self.composition
    }

    /// Returns a mutable reference to the underlying [`CompositionManager`].
    pub fn composition_mut(&mut self) -> &mut CompositionManager {
        &mut self.composition
    }

    /// Get the [`IDCompositionVisual`] for a subduction slot index.
    ///
    /// Applications use this to attach GPU content:
    /// ```ignore
    /// if let Some(visual) = presenter.visual_for(idx) {
    ///     unsafe { visual.SetContent(&swapchain)?; }
    /// }
    /// ```
    pub fn visual_for(&self, idx: u32) -> Option<&IDCompositionVisual> {
        self.mapped_id(idx).map(|id| self.composition.visual(id))
    }

    /// Get the composition [`LayerId`] for a subduction slot index.
    #[must_use]
    pub fn mapped_id(&self, idx: u32) -> Option<LayerId> {
        self.layer_map.get(idx as usize).copied().flatten()
    }

    /// Commit all pending `DirectComposition` changes atomically.
    pub fn commit(&self) -> windows::core::Result<()> {
        self.composition.commit()
    }

    // ── Effects (delegated to CompositionManager) ──────────

    /// Apply a Gaussian blur effect. `sigma` <= 0 removes the blur.
    pub fn set_blur(&mut self, idx: u32, sigma: f32) -> windows::core::Result<()> {
        let id = self.mapped_id(idx).expect("set_blur: unmapped layer");
        self.composition.set_blur(id, sigma)
    }

    /// Apply a saturation effect (0.0 = grayscale, 1.0 = identity).
    pub fn set_saturation(&mut self, idx: u32, amount: f32) -> windows::core::Result<()> {
        let id = self.mapped_id(idx).expect("set_saturation: unmapped layer");
        self.composition.set_saturation(id, amount)
    }

    /// Apply a 5x4 color matrix effect (20 floats, row-major).
    pub fn set_color_matrix(&mut self, idx: u32, matrix: &[f32; 20]) -> windows::core::Result<()> {
        let id = self
            .mapped_id(idx)
            .expect("set_color_matrix: unmapped layer");
        self.composition.set_color_matrix(id, matrix)
    }

    /// Apply a brightness effect with white/black point curves.
    pub fn set_brightness(
        &mut self,
        idx: u32,
        white: (f32, f32),
        black: (f32, f32),
    ) -> windows::core::Result<()> {
        let id = self.mapped_id(idx).expect("set_brightness: unmapped layer");
        self.composition.set_brightness(id, white, black)
    }

    /// Remove all effects from a layer.
    pub fn clear_effects(&mut self, idx: u32) -> windows::core::Result<()> {
        let id = self.mapped_id(idx).expect("clear_effects: unmapped layer");
        self.composition.clear_effects(id)
    }

    // ── Animations (delegated to CompositionManager) ─────

    /// Animate opacity from `from` to `to` over `duration_s` seconds.
    pub fn animate_opacity(
        &mut self,
        idx: u32,
        from: f32,
        to: f32,
        duration_s: f64,
        now: f64,
    ) -> windows::core::Result<()> {
        let id = self
            .mapped_id(idx)
            .expect("animate_opacity: unmapped layer");
        self.composition
            .animate_opacity(id, from, to, duration_s, now)
    }

    /// Animate offset from `from` to `to` over `duration_s` seconds.
    pub fn animate_offset(
        &mut self,
        idx: u32,
        from: (f32, f32),
        to: (f32, f32),
        duration_s: f64,
        now: f64,
    ) -> windows::core::Result<()> {
        let id = self.mapped_id(idx).expect("animate_offset: unmapped layer");
        self.composition
            .animate_offset(id, from, to, duration_s, now)
    }

    /// Check for completed animations. Returns the number completed.
    pub fn tick_animations(&mut self, now: f64) -> usize {
        self.composition.tick_animations(now)
    }

    /// Whether any animations are currently active.
    pub fn has_active_animations(&self) -> bool {
        self.composition.has_active_animations()
    }

    // ── Scroll offset ────────────────────────────────────

    /// DWM-level scroll: shift visual content without re-rendering.
    pub fn set_scroll_offset(&mut self, idx: u32, dx: f32, dy: f32) -> windows::core::Result<()> {
        let id = self
            .mapped_id(idx)
            .expect("set_scroll_offset: unmapped layer");
        self.composition.set_scroll_offset(id, dx, dy)
    }

    /// Ensure the maps have enough slots for the given index.
    fn ensure_slot(&mut self, idx: u32) {
        let needed = idx as usize + 1;
        if self.layer_map.len() < needed {
            self.layer_map.resize(needed, None);
        }
        if self.layer_parents.len() < needed {
            self.layer_parents.resize(needed, None);
        }
    }
}

/// Convert subduction's `Transform3d` (f64 column-major 4×4) to the
/// f32 format expected by [`CompositionManager::set_transform_3d`].
fn transform_to_f32(t: &subduction_core::transform::Transform3d) -> [[f32; 4]; 4] {
    let cols = t.to_cols_array_2d();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Transform values are intentionally truncated from f64 to f32 for DirectComposition"
    )]
    [
        [
            cols[0][0] as f32,
            cols[0][1] as f32,
            cols[0][2] as f32,
            cols[0][3] as f32,
        ],
        [
            cols[1][0] as f32,
            cols[1][1] as f32,
            cols[1][2] as f32,
            cols[1][3] as f32,
        ],
        [
            cols[2][0] as f32,
            cols[2][1] as f32,
            cols[2][2] as f32,
            cols[2][3] as f32,
        ],
        [
            cols[3][0] as f32,
            cols[3][1] as f32,
            cols[3][2] as f32,
            cols[3][3] as f32,
        ],
    ]
}

/// Apply a [`ClipShape`] to a layer via the composition manager.
fn apply_clip(
    composition: &mut CompositionManager,
    layer_id: LayerId,
    clip: &ClipShape,
) -> windows::core::Result<()> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Clip coordinates are intentionally truncated from f64 to f32"
    )]
    match clip {
        ClipShape::Rect(r) => {
            composition.set_clip(layer_id, r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32)
        }
        ClipShape::RoundedRect(rr) => {
            let r = rr.rect();
            let radii = rr.radii();
            composition.set_rounded_clip(
                layer_id,
                r.x0 as f32,
                r.y0 as f32,
                r.x1 as f32,
                r.y1 as f32,
                radii.top_left as f32,
                radii.top_right as f32,
                radii.bottom_right as f32,
                radii.bottom_left as f32,
            )
        }
    }
}

impl Presenter for DCompPresenter {
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        // ── Structural: added layers ────────────────────────────────
        for &idx in &changes.added {
            self.ensure_slot(idx);

            // Resolve parent: subduction slot → composition LayerId.
            let parent_id = store
                .parent_at(idx)
                .and_then(|parent_idx| self.mapped_id(parent_idx));

            let layer_id = self
                .composition
                .create_layer(parent_id)
                .expect("failed to create DirectComposition visual");

            self.layer_map[idx as usize] = Some(layer_id);
            self.layer_parents[idx as usize] = Some(parent_id);
        }

        // ── Structural: removed layers ──────────────────────────────
        for &idx in &changes.removed {
            if let Some(layer_id) = self.mapped_id(idx) {
                let parent = self.layer_parents[idx as usize].flatten();
                let _ = self.composition.destroy_layer(layer_id, parent, true);
                self.layer_map[idx as usize] = None;
                self.layer_parents[idx as usize] = None;
            }
        }

        // ── Topology: reparent layers whose parent changed ─────────
        if changes.topology_changed {
            for idx in 0..self.layer_map.len() {
                let Some(layer_id) = self.layer_map[idx] else {
                    continue;
                };
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Layer index fits in u32 by construction"
                )]
                let store_parent = store.parent_at(idx as u32).and_then(|p| self.mapped_id(p));
                let old_parent = self.layer_parents[idx].flatten();
                if store_parent != old_parent {
                    let _ = self
                        .composition
                        .reparent(layer_id, old_parent, store_parent, true);
                    self.layer_parents[idx] = Some(store_parent);
                }
            }
        }

        // ── Transforms ─────────────────────────────────────────────
        // Decompose each local transform into:
        //   - Translation → SetOffsetX/Y (inherits through visual tree)
        //   - Residual (rotation/scale) → EffectGroup 3D transform
        for &idx in &changes.transforms {
            if let Some(layer_id) = self.mapped_id(idx) {
                let t = store.local_transform_at(idx);
                let cols = t.to_cols_array_2d();

                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Translation is intentionally truncated from f64 to f32"
                )]
                let (tx, ty) = (cols[3][0] as f32, cols[3][1] as f32);
                let _ = self.composition.set_offset(layer_id, tx, ty);

                // Check if there's a non-identity residual (rotation/scale).
                let has_residual = cols[0][0] != 1.0
                    || cols[0][1] != 0.0
                    || cols[0][2] != 0.0
                    || cols[1][0] != 0.0
                    || cols[1][1] != 1.0
                    || cols[1][2] != 0.0
                    || cols[2][0] != 0.0
                    || cols[2][1] != 0.0
                    || cols[2][2] != 1.0;

                if has_residual {
                    let mut residual = transform_to_f32(&t);
                    // Zero out translation — offset handles it.
                    residual[3][0] = 0.0;
                    residual[3][1] = 0.0;
                    residual[3][2] = 0.0;
                    let _ = self.composition.set_transform_3d(layer_id, &residual);
                } else {
                    let _ = self.composition.clear_transform_3d(layer_id);
                }
            }
        }

        // ── Opacities ──────────────────────────────────────────────
        // Use local opacity: DComp composes parent opacity via the
        // visual tree hierarchy.
        for &idx in &changes.opacities {
            if let Some(layer_id) = self.mapped_id(idx) {
                let opacity = store.local_opacity_at(idx);
                let _ = self.composition.set_opacity(layer_id, opacity);
            }
        }

        // ── Clips ──────────────────────────────────────────────────
        for &idx in &changes.clips {
            if let Some(layer_id) = self.mapped_id(idx) {
                if let Some(clip) = store.clip_at(idx) {
                    let _ = apply_clip(&mut self.composition, layer_id, &clip);
                } else {
                    let _ = self.composition.clear_clip(layer_id);
                }
            }
        }

        // ── Visibility ─────────────────────────────────────────────
        for &idx in &changes.hidden {
            if let Some(layer_id) = self.mapped_id(idx) {
                let parent = self.layer_parents[idx as usize].flatten();
                let _ = self.composition.set_visible(layer_id, parent, false);
            }
        }
        for &idx in &changes.unhidden {
            if let Some(layer_id) = self.mapped_id(idx) {
                let parent = self.layer_parents[idx as usize].flatten();
                let _ = self.composition.set_visible(layer_id, parent, true);
            }
        }

        // ── Commit all visual tree changes atomically ──────────────
        let _ = self.composition.commit();
    }
}
