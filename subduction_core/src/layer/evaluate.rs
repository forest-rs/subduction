// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame evaluation and change tracking.
//!
//! Evaluation follows a drain-recompute pattern for each dirty channel:
//!
//! 1. **TRANSFORM** — Drain dirty indices, recompute each layer's
//!    `world_transform` as `parent_world * local_transform` and
//!    `effective_hidden` as `parent_effective_hidden || flags.hidden`.
//! 2. **OPACITY** — Drain dirty indices, recompute each layer's
//!    `effective_opacity` as `parent_effective * local_opacity`.
//! 3. **CLIP** / **CONTENT** — Drain dirty indices (no recomputation;
//!    backends read the current values directly from the store).
//! 4. **TOPOLOGY** — Drain and discard (the traversal order was already
//!    rebuilt at the start of evaluation if needed).
//!
//! [`FrameChanges`] uses raw slot indices (`u32`) rather than [`LayerId`]
//! handles so that backends can index directly into the store's `SoA` arrays
//! via the `*_at()` accessors (e.g.
//! [`world_transform_at`](super::LayerStore::world_transform_at)) without
//! paying for generation checks on every access.
//!
//! [`LayerId`]: super::LayerId

use alloc::vec::Vec;

use super::id::INVALID;
use super::store::LayerStore;
use crate::dirty;

/// The set of changes produced by a single [`LayerStore::evaluate`] call.
///
/// Each field contains the raw slot indices of layers that changed in the
/// corresponding category. Backends use these to apply incremental updates.
#[derive(Clone, Debug, Default)]
pub struct FrameChanges {
    /// Layers whose world transform was recomputed.
    pub transforms: Vec<u32>,
    /// Layers whose effective opacity was recomputed.
    pub opacities: Vec<u32>,
    /// Layers whose clip shape changed.
    pub clips: Vec<u32>,
    /// Layers whose surface content changed.
    pub content: Vec<u32>,
    /// Layers whose bounds changed.
    pub bounds: Vec<u32>,
    /// Layers that transitioned from visible to effectively hidden.
    pub hidden: Vec<u32>,
    /// Layers that transitioned from effectively hidden to visible.
    pub unhidden: Vec<u32>,
    /// Layers added since the last evaluate.
    pub added: Vec<u32>,
    /// Layers removed since the last evaluate.
    pub removed: Vec<u32>,
    /// Whether the tree topology changed (traversal order was rebuilt).
    pub topology_changed: bool,
}

impl FrameChanges {
    /// Clears all change lists.
    pub fn clear(&mut self) {
        self.transforms.clear();
        self.opacities.clear();
        self.clips.clear();
        self.content.clear();
        self.bounds.clear();
        self.hidden.clear();
        self.unhidden.clear();
        self.added.clear();
        self.removed.clear();
        self.topology_changed = false;
    }
}

impl LayerStore {
    /// Evaluates the layer tree, recomputing dirty properties and returning
    /// the set of changes.
    ///
    /// This rebuilds the traversal order if topology changed, then drains each
    /// dirty channel and recomputes world transforms and effective opacities
    /// in parent-before-child order.
    pub fn evaluate(&mut self) -> FrameChanges {
        let mut changes = FrameChanges::default();
        self.evaluate_into(&mut changes);
        changes
    }

    /// Like [`evaluate`](Self::evaluate), but reuses a caller-provided buffer
    /// to avoid allocation.
    pub fn evaluate_into(&mut self, changes: &mut FrameChanges) {
        changes.clear();

        // Rebuild traversal order if needed.
        if self.traversal_dirty {
            self.rebuild_traversal_order();
            changes.topology_changed = true;
            self.traversal_dirty = false;
        }

        // Drain TRANSFORM channel — collect dirty indices, then recompute.
        let dirty_transforms: Vec<u32> = self
            .dirty
            .drain(dirty::TRANSFORM)
            .affected()
            .deterministic()
            .run()
            .collect();
        for &idx in &dirty_transforms {
            let parent_idx = self.parent[idx as usize];
            let parent_world = if parent_idx != INVALID {
                self.world_transform[parent_idx as usize]
            } else {
                crate::transform::Transform3d::IDENTITY
            };
            self.world_transform[idx as usize] = parent_world * self.local_transform[idx as usize];

            // Compute effective hidden: parent_effective_hidden || self.flags.hidden
            let parent_hidden = if parent_idx != INVALID {
                self.effective_hidden[parent_idx as usize]
            } else {
                false
            };
            let new_hidden = parent_hidden || self.flags[idx as usize].hidden;
            let old_hidden = self.effective_hidden[idx as usize];
            if new_hidden != old_hidden {
                if new_hidden {
                    changes.hidden.push(idx);
                } else {
                    changes.unhidden.push(idx);
                }
                self.effective_hidden[idx as usize] = new_hidden;
            }
        }
        changes.transforms = dirty_transforms;

        // Drain OPACITY channel.
        let dirty_opacities: Vec<u32> = self
            .dirty
            .drain(dirty::OPACITY)
            .affected()
            .deterministic()
            .run()
            .collect();
        for &idx in &dirty_opacities {
            let parent_opacity = if self.parent[idx as usize] != INVALID {
                self.effective_opacity[self.parent[idx as usize] as usize]
            } else {
                1.0
            };
            self.effective_opacity[idx as usize] =
                parent_opacity * self.local_opacity[idx as usize];
        }
        changes.opacities = dirty_opacities;

        // Drain CLIP channel — no recomputation, just collect.
        changes.clips = self
            .dirty
            .drain(dirty::CLIP)
            .deterministic()
            .run()
            .collect();

        // Drain CONTENT channel.
        changes.content = self
            .dirty
            .drain(dirty::CONTENT)
            .deterministic()
            .run()
            .collect();

        // Drain BOUNDS channel — no recomputation, just collect.
        changes.bounds = self
            .dirty
            .drain(dirty::BOUNDS)
            .deterministic()
            .run()
            .collect();

        // Drain TOPOLOGY channel (just consume, changes are structural).
        let _: Vec<u32> = self
            .dirty
            .drain(dirty::TOPOLOGY)
            .deterministic()
            .run()
            .collect();

        // Move lifecycle lists.
        core::mem::swap(&mut self.pending_added, &mut changes.added);
        core::mem::swap(&mut self.pending_removed, &mut changes.removed);
    }

    /// Returns the current traversal order (depth-first pre-order).
    ///
    /// Only valid after [`evaluate`](Self::evaluate) has been called at least
    /// once (or if the traversal has been manually rebuilt).
    #[must_use]
    pub fn traversal_order(&self) -> &[u32] {
        &self.traversal_order
    }

    /// Rebuilds the depth-first pre-order traversal of all live layers.
    fn rebuild_traversal_order(&mut self) {
        self.traversal_order.clear();
        // Start from roots.
        for idx in 0..self.len {
            if self.parent[idx as usize] == INVALID && !self.free_list.contains(&idx) {
                self.dfs_collect(idx);
            }
        }
    }

    /// Depth-first pre-order collection starting from `idx`.
    fn dfs_collect(&mut self, idx: u32) {
        self.traversal_order.push(idx);
        let mut child = self.first_child[idx as usize];
        while child != INVALID {
            self.dfs_collect(child);
            child = self.next_sibling[child as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::transform::Transform3d;

    use super::*;

    #[test]
    fn evaluate_computes_world_transforms() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();

        let parent_xf = Transform3d::from_translation(10.0, 0.0, 0.0);
        let child_xf = Transform3d::from_translation(0.0, 5.0, 0.0);

        store.set_transform(parent, parent_xf);
        store.set_transform(child, child_xf);
        store.add_child(parent, child);

        let _changes = store.evaluate();

        assert_eq!(store.world_transform(parent), parent_xf);
        let expected = parent_xf * child_xf;
        assert_eq!(store.world_transform(child), expected);
    }

    #[test]
    fn evaluate_computes_effective_opacity() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();

        store.set_opacity(parent, 0.5);
        store.set_opacity(child, 0.8);
        store.add_child(parent, child);

        let _changes = store.evaluate();

        let eps = 1e-6;
        assert!((store.effective_opacity(parent) - 0.5).abs() < eps);
        assert!((store.effective_opacity(child) - 0.4).abs() < eps);
    }

    #[test]
    fn no_change_evaluate_returns_empty() {
        let mut store = LayerStore::new();
        let _root = store.create_layer();

        // First evaluate processes initial creation.
        let _ = store.evaluate();

        // Second evaluate should have no changes.
        let changes = store.evaluate();
        assert!(changes.transforms.is_empty());
        assert!(changes.opacities.is_empty());
        assert!(changes.clips.is_empty());
        assert!(changes.content.is_empty());
        assert!(changes.bounds.is_empty());
        assert!(changes.added.is_empty());
        assert!(changes.removed.is_empty());
        assert!(!changes.topology_changed);
    }

    #[test]
    fn traversal_order_is_depth_first() {
        let mut store = LayerStore::new();
        let a = store.create_layer();
        let b = store.create_layer();
        let c = store.create_layer();
        let d = store.create_layer();

        // Tree: a -> [b -> [d], c]
        store.add_child(a, b);
        store.add_child(a, c);
        store.add_child(b, d);

        let _ = store.evaluate();

        let order = store.traversal_order();
        assert_eq!(order, &[a.idx, b.idx, d.idx, c.idx]);
    }

    #[test]
    fn evaluate_tracks_clip_and_content_changes() {
        use crate::layer::{ClipShape, SurfaceId};

        let mut store = LayerStore::new();
        let id = store.create_layer();
        let _ = store.evaluate();

        store.set_clip(
            id,
            Some(ClipShape::Rect(kurbo::Rect::new(0.0, 0.0, 50.0, 50.0))),
        );
        store.set_content(id, Some(SurfaceId(1)));
        let changes = store.evaluate();
        assert!(changes.clips.contains(&id.idx));
        assert!(changes.content.contains(&id.idx));
    }

    #[test]
    fn evaluate_multiple_roots() {
        let mut store = LayerStore::new();
        let root_a = store.create_layer();
        let child_a = store.create_layer();
        let root_b = store.create_layer();

        store.add_child(root_a, child_a);

        let parent_xf = Transform3d::from_translation(1.0, 0.0, 0.0);
        let child_xf = Transform3d::from_translation(0.0, 2.0, 0.0);
        let root_b_xf = Transform3d::from_translation(3.0, 0.0, 0.0);

        store.set_transform(root_a, parent_xf);
        store.set_transform(child_a, child_xf);
        store.set_transform(root_b, root_b_xf);

        let _ = store.evaluate();

        assert_eq!(store.world_transform(root_a), parent_xf);
        assert_eq!(store.world_transform(child_a), parent_xf * child_xf);
        assert_eq!(store.world_transform(root_b), root_b_xf);
    }

    #[test]
    fn evaluate_propagates_opacity_to_descendants() {
        let mut store = LayerStore::new();
        let grandparent = store.create_layer();
        let parent = store.create_layer();
        let child = store.create_layer();

        store.add_child(grandparent, parent);
        store.add_child(parent, child);

        store.set_opacity(grandparent, 0.5);
        store.set_opacity(parent, 0.8);
        store.set_opacity(child, 0.5);

        let _ = store.evaluate();

        let eps = 1e-6;
        assert!((store.effective_opacity(grandparent) - 0.5).abs() < eps);
        assert!((store.effective_opacity(parent) - 0.4).abs() < eps);
        assert!((store.effective_opacity(child) - 0.2).abs() < eps);
    }

    #[test]
    fn evaluate_added_and_removed_lifecycle() {
        let mut store = LayerStore::new();
        let id = store.create_layer();

        // First evaluate: layer should appear in `added`.
        let changes = store.evaluate();
        assert!(changes.added.contains(&id.idx));
        assert!(changes.removed.is_empty());

        // Second evaluate: no lifecycle events.
        let changes = store.evaluate();
        assert!(changes.added.is_empty());
        assert!(changes.removed.is_empty());

        // Destroy: should appear in `removed` on next evaluate.
        store.destroy_layer(id);
        let changes = store.evaluate();
        assert!(changes.removed.contains(&id.idx));
        assert!(changes.added.is_empty());
    }

    #[test]
    fn hidden_layer_is_effectively_hidden() {
        use crate::layer::LayerFlags;

        let mut store = LayerStore::new();
        let root = store.create_layer();
        let _ = store.evaluate();

        store.set_flags(root, LayerFlags { hidden: true });
        let changes = store.evaluate();

        assert!(store.effective_hidden(root));
        assert!(changes.hidden.contains(&root.idx));
        assert!(changes.unhidden.is_empty());
    }

    #[test]
    fn hidden_propagates_to_children() {
        use crate::layer::LayerFlags;

        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();
        store.add_child(parent, child);
        let _ = store.evaluate();

        store.set_flags(parent, LayerFlags { hidden: true });
        let changes = store.evaluate();

        assert!(store.effective_hidden(parent));
        assert!(store.effective_hidden(child));
        assert!(changes.hidden.contains(&parent.idx));
        assert!(changes.hidden.contains(&child.idx));
    }

    #[test]
    fn unhide_restores_visibility() {
        use crate::layer::LayerFlags;

        let mut store = LayerStore::new();
        let root = store.create_layer();
        let _ = store.evaluate();

        // Hide
        store.set_flags(root, LayerFlags { hidden: true });
        let _ = store.evaluate();
        assert!(store.effective_hidden(root));

        // Unhide
        store.set_flags(root, LayerFlags { hidden: false });
        let changes = store.evaluate();

        assert!(!store.effective_hidden(root));
        assert!(changes.unhidden.contains(&root.idx));
        assert!(changes.hidden.is_empty());
    }

    #[test]
    fn hidden_layer_still_computes_transform() {
        use crate::layer::LayerFlags;

        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();
        store.add_child(parent, child);

        let parent_xf = Transform3d::from_translation(10.0, 0.0, 0.0);
        let child_xf = Transform3d::from_translation(0.0, 5.0, 0.0);
        store.set_transform(parent, parent_xf);
        store.set_transform(child, child_xf);
        store.set_flags(parent, LayerFlags { hidden: true });

        let _ = store.evaluate();

        // World transforms are still computed even though hidden.
        assert_eq!(store.world_transform(parent), parent_xf);
        assert_eq!(store.world_transform(child), parent_xf * child_xf);
        assert!(store.effective_hidden(parent));
        assert!(store.effective_hidden(child));
    }

    #[test]
    fn mutation_while_hidden() {
        use crate::layer::LayerFlags;

        let mut store = LayerStore::new();
        let root = store.create_layer();
        store.set_flags(root, LayerFlags { hidden: true });
        let _ = store.evaluate();

        // Mutate transform while hidden.
        let xf = Transform3d::from_translation(42.0, 0.0, 0.0);
        store.set_transform(root, xf);
        let _ = store.evaluate();
        assert_eq!(store.world_transform(root), xf);

        // Unhide — transform should reflect the mutation.
        store.set_flags(root, LayerFlags { hidden: false });
        let changes = store.evaluate();

        assert!(!store.effective_hidden(root));
        assert!(changes.unhidden.contains(&root.idx));
        assert_eq!(store.world_transform(root), xf);
    }

    #[test]
    fn evaluate_into_reuses_buffer() {
        let mut store = LayerStore::new();
        let a = store.create_layer();
        let b = store.create_layer();

        let mut changes = FrameChanges::default();

        // First evaluate: both layers added.
        store.evaluate_into(&mut changes);
        assert_eq!(changes.added.len(), 2);

        // Mutate one layer.
        store.set_opacity(a, 0.5);
        store.evaluate_into(&mut changes);

        // Buffer should be cleared and refilled (not accumulating).
        assert!(changes.added.is_empty(), "added should be cleared");
        assert!(
            changes.opacities.contains(&a.idx),
            "opacity change should be present"
        );
        assert!(
            !changes.opacities.contains(&b.idx),
            "unchanged layer should not appear"
        );
    }
}
