// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Struct-of-arrays layer storage with allocation, topology, and property management.

use alloc::vec::Vec;

use invalidation::{CycleHandling, EagerPolicy, InvalidationTracker};
use kurbo::{Rect, Size};

use crate::transform::Transform3d;

use super::clip::ClipShape;
use super::id::{INVALID, LayerId, SurfaceId};
use super::traverse::Children;
use crate::dirty;

/// Per-layer boolean flags.
///
/// Setting [`hidden`](Self::hidden) suppresses all visual contribution of the
/// layer and its entire subtree. Properties can still be mutated while hidden;
/// unhiding restores state immediately without re-evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LayerFlags {
    /// Whether the layer (and its subtree) is hidden.
    pub hidden: bool,
}

/// Struct-of-arrays storage for all layers.
///
/// Layers are addressed by [`LayerId`] handles. Internally, each layer occupies
/// a slot in parallel arrays. Destroyed layers are recycled via a free list,
/// and generation counters prevent stale handle access.
#[derive(Debug)]
pub struct LayerStore {
    // -- Topology --
    pub(crate) parent: Vec<u32>,
    pub(crate) first_child: Vec<u32>,
    pub(crate) next_sibling: Vec<u32>,
    pub(crate) prev_sibling: Vec<u32>,

    // -- Local properties (set by callers) --
    pub(crate) local_transform: Vec<Transform3d>,
    pub(crate) local_opacity: Vec<f32>,
    pub(crate) clip: Vec<Option<ClipShape>>,
    pub(crate) content: Vec<Option<SurfaceId>>,
    pub(crate) flags: Vec<LayerFlags>,
    pub(crate) bounds: Vec<Size>,
    pub(crate) hit_rect: Vec<Option<Rect>>,

    // -- Computed properties (written by evaluate) --
    pub(crate) world_transform: Vec<Transform3d>,
    pub(crate) effective_opacity: Vec<f32>,
    pub(crate) effective_hidden: Vec<bool>,

    // -- Allocation --
    pub(crate) generation: Vec<u32>,
    pub(crate) free_list: Vec<u32>,
    pub(crate) len: u32,

    // -- Dirty tracking --
    pub(crate) dirty: InvalidationTracker<u32>,

    // -- Traversal cache --
    pub(crate) traversal_order: Vec<u32>,
    pub(crate) traversal_dirty: bool,

    // -- Lifecycle tracking --
    pub(crate) pending_added: Vec<u32>,
    pub(crate) pending_removed: Vec<u32>,
}

impl Default for LayerStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerStore {
    /// Creates an empty layer store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parent: Vec::new(),
            first_child: Vec::new(),
            next_sibling: Vec::new(),
            prev_sibling: Vec::new(),
            local_transform: Vec::new(),
            local_opacity: Vec::new(),
            clip: Vec::new(),
            content: Vec::new(),
            flags: Vec::new(),
            bounds: Vec::new(),
            hit_rect: Vec::new(),
            world_transform: Vec::new(),
            effective_opacity: Vec::new(),
            effective_hidden: Vec::new(),
            generation: Vec::new(),
            free_list: Vec::new(),
            len: 0,
            dirty: InvalidationTracker::with_cycle_handling(CycleHandling::Error),
            traversal_order: Vec::new(),
            traversal_dirty: true,
            pending_added: Vec::new(),
            pending_removed: Vec::new(),
        }
    }

    /// Returns the number of live layers in the store.
    ///
    /// Destroyed layers are not counted, even though their slots may remain
    /// allocated internally for reuse.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize - self.free_list.len()
    }

    /// Returns whether the store contains no live layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn mark_inherited_dirty(&mut self, idx: u32) {
        self.dirty.mark_with(idx, dirty::TRANSFORM, &EagerPolicy);
        self.dirty.mark_with(idx, dirty::OPACITY, &EagerPolicy);
    }

    // -- Allocation API --

    /// Creates a new layer and returns its handle.
    ///
    /// The layer starts with an identity transform, full opacity, no clip,
    /// no content, and no parent.
    pub fn create_layer(&mut self) -> LayerId {
        let idx = if let Some(idx) = self.free_list.pop() {
            // Reuse a freed slot.
            self.generation[idx as usize] += 1;
            self.parent[idx as usize] = INVALID;
            self.first_child[idx as usize] = INVALID;
            self.next_sibling[idx as usize] = INVALID;
            self.prev_sibling[idx as usize] = INVALID;
            self.local_transform[idx as usize] = Transform3d::IDENTITY;
            self.local_opacity[idx as usize] = 1.0;
            self.clip[idx as usize] = None;
            self.content[idx as usize] = None;
            self.flags[idx as usize] = LayerFlags::default();
            self.bounds[idx as usize] = Size::ZERO;
            self.hit_rect[idx as usize] = None;
            self.world_transform[idx as usize] = Transform3d::IDENTITY;
            self.effective_opacity[idx as usize] = 1.0;
            self.effective_hidden[idx as usize] = false;
            idx
        } else {
            // Allocate a new slot.
            let idx = self.len;
            self.len += 1;
            self.parent.push(INVALID);
            self.first_child.push(INVALID);
            self.next_sibling.push(INVALID);
            self.prev_sibling.push(INVALID);
            self.local_transform.push(Transform3d::IDENTITY);
            self.local_opacity.push(1.0);
            self.clip.push(None);
            self.content.push(None);
            self.flags.push(LayerFlags::default());
            self.bounds.push(Size::ZERO);
            self.hit_rect.push(None);
            self.world_transform.push(Transform3d::IDENTITY);
            self.effective_opacity.push(1.0);
            self.effective_hidden.push(false);
            self.generation.push(0);
            idx
        };

        self.traversal_dirty = true;
        self.pending_added.push(idx);
        self.dirty.mark(idx, dirty::TOPOLOGY);

        LayerId {
            idx,
            generation: self.generation[idx as usize],
        }
    }

    /// Destroys a layer, freeing its slot for reuse.
    ///
    /// # Panics
    ///
    /// Panics if the layer has children (remove them first) or if the handle
    /// is stale.
    pub fn destroy_layer(&mut self, id: LayerId) {
        self.validate(id);
        let idx = id.idx;
        assert!(
            self.first_child[idx as usize] == INVALID,
            "cannot destroy layer with children"
        );

        // Remove from parent's child list if attached.
        if self.parent[idx as usize] != INVALID {
            self.unlink_from_parent(idx);
        }

        // Remove dirty tracking dependencies.
        self.dirty.remove_key(idx);

        // Bump generation so old handles immediately fail validation.
        self.generation[idx as usize] += 1;

        self.free_list.push(idx);
        self.traversal_dirty = true;
        self.pending_removed.push(idx);
        self.dirty.mark(idx, dirty::TOPOLOGY);
    }

    /// Returns whether the given handle refers to a live layer.
    #[must_use]
    pub fn is_alive(&self, id: LayerId) -> bool {
        (id.idx < self.len)
            && self.generation[id.idx as usize] == id.generation
            && !self.free_list.contains(&id.idx)
    }

    // -- Topology API --

    /// Adds `child` as the last child of `parent`.
    ///
    /// Sibling order is back-to-front, so the appended child is the front-most
    /// child of `parent`.
    ///
    /// # Panics
    ///
    /// Panics if either handle is stale, or if `child` already has a parent.
    pub fn add_child(&mut self, parent: LayerId, child: LayerId) {
        self.validate(parent);
        self.validate(child);
        let p = parent.idx;
        let c = child.idx;
        assert!(
            self.parent[c as usize] == INVALID,
            "child already has a parent"
        );

        self.parent[c as usize] = p;
        self.prev_sibling[c as usize] = INVALID;
        self.next_sibling[c as usize] = INVALID;

        if self.first_child[p as usize] == INVALID {
            self.first_child[p as usize] = c;
        } else {
            // Walk to last child.
            let mut last = self.first_child[p as usize];
            while self.next_sibling[last as usize] != INVALID {
                last = self.next_sibling[last as usize];
            }
            self.next_sibling[last as usize] = c;
            self.prev_sibling[c as usize] = last;
        }

        // Add dirty dependency edges: child depends on parent for TRANSFORM and OPACITY.
        let _ = self.dirty.add_dependency(c, p, dirty::TRANSFORM);
        let _ = self.dirty.add_dependency(c, p, dirty::OPACITY);

        self.traversal_dirty = true;
        self.mark_inherited_dirty(c);
        self.dirty.mark(p, dirty::TOPOLOGY);
    }

    /// Moves `layer` directly before `sibling` in their parent's child list.
    ///
    /// Sibling order is back-to-front, so this places `layer` immediately
    /// behind `sibling`. This is a same-parent reorder operation; it does not
    /// change inherited transform or opacity state.
    ///
    /// # Panics
    ///
    /// Panics if either handle is stale, either layer has no parent, or the
    /// layers do not share the same parent.
    pub fn move_before(&mut self, layer: LayerId, sibling: LayerId) {
        self.validate(layer);
        self.validate(sibling);
        if layer == sibling {
            return;
        }

        let layer_idx = layer.idx;
        let sibling_idx = sibling.idx;
        if self.next_sibling[layer_idx as usize] == sibling_idx {
            return;
        }

        let parent = self.shared_reorder_parent(layer_idx, sibling_idx);
        self.unlink_from_parent(layer_idx);
        self.insert_unlinked_before(layer_idx, sibling_idx, parent);
        self.mark_topology_reordered(parent);
    }

    /// Moves `layer` directly after `sibling` in their parent's child list.
    ///
    /// Sibling order is back-to-front, so this places `layer` immediately in
    /// front of `sibling`. This is a same-parent reorder operation; it does not
    /// change inherited transform or opacity state.
    ///
    /// # Panics
    ///
    /// Panics if either handle is stale, either layer has no parent, or the
    /// layers do not share the same parent.
    pub fn move_after(&mut self, layer: LayerId, sibling: LayerId) {
        self.validate(layer);
        self.validate(sibling);
        if layer == sibling {
            return;
        }

        let layer_idx = layer.idx;
        let sibling_idx = sibling.idx;
        if self.prev_sibling[layer_idx as usize] == sibling_idx {
            return;
        }

        let parent = self.shared_reorder_parent(layer_idx, sibling_idx);
        self.unlink_from_parent(layer_idx);
        self.insert_unlinked_after(layer_idx, sibling_idx, parent);
        self.mark_topology_reordered(parent);
    }

    /// Moves `layer` to the front of its parent's child list.
    ///
    /// Sibling order is back-to-front, so the front-most child is the last
    /// sibling visited during normal traversal and the first sibling considered
    /// during hit testing.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale or the layer has no parent.
    pub fn move_to_front(&mut self, layer: LayerId) {
        self.validate(layer);
        let layer_idx = layer.idx;
        let parent = self.reorder_parent(layer_idx);
        if self.next_sibling[layer_idx as usize] == INVALID {
            return;
        }

        self.unlink_from_parent(layer_idx);
        self.append_unlinked_child(parent, layer_idx);
        self.mark_topology_reordered(parent);
    }

    /// Moves `layer` to the back of its parent's child list.
    ///
    /// Sibling order is back-to-front, so the back-most child is the first
    /// sibling visited during normal traversal and the last sibling considered
    /// during hit testing.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale or the layer has no parent.
    pub fn move_to_back(&mut self, layer: LayerId) {
        self.validate(layer);
        let layer_idx = layer.idx;
        let parent = self.reorder_parent(layer_idx);
        if self.prev_sibling[layer_idx as usize] == INVALID {
            return;
        }

        self.unlink_from_parent(layer_idx);
        self.prepend_unlinked_child(parent, layer_idx);
        self.mark_topology_reordered(parent);
    }

    /// Removes `child` from its current parent.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale or the layer has no parent.
    pub fn remove_from_parent(&mut self, child: LayerId) {
        self.validate(child);
        let c = child.idx;
        assert!(self.parent[c as usize] != INVALID, "layer has no parent");

        let p = self.parent[c as usize];
        self.unlink_from_parent(c);

        // Remove dirty dependency edges.
        self.dirty.remove_dependency(c, p, dirty::TRANSFORM);
        self.dirty.remove_dependency(c, p, dirty::OPACITY);

        self.traversal_dirty = true;
        self.mark_inherited_dirty(c);
        self.dirty.mark(p, dirty::TOPOLOGY);
    }

    /// Destroys `root` and all descendants in postorder.
    ///
    /// Descendants are destroyed before their parents, using the current
    /// back-to-front child order. This preserves the same generation
    /// invalidation and lifecycle reporting semantics as repeated
    /// [`destroy_layer`](Self::destroy_layer) calls.
    ///
    /// # Panics
    ///
    /// Panics if `root` is stale.
    pub fn destroy_subtree(&mut self, root: LayerId) {
        self.validate(root);

        let mut postorder = Vec::new();
        self.collect_subtree_postorder(root.idx, &mut postorder);

        for idx in postorder {
            self.destroy_layer(LayerId {
                idx,
                generation: self.generation[idx as usize],
            });
        }
    }

    /// Moves `child` to be a child of `new_parent`.
    ///
    /// If `child` already has a parent, it is removed first.
    ///
    /// # Panics
    ///
    /// Panics if either handle is stale.
    pub fn reparent(&mut self, child: LayerId, new_parent: LayerId) {
        self.validate(child);
        self.validate(new_parent);

        if self.parent[child.idx as usize] != INVALID {
            let old_p = self.parent[child.idx as usize];
            self.unlink_from_parent(child.idx);
            self.dirty
                .remove_dependency(child.idx, old_p, dirty::TRANSFORM);
            self.dirty
                .remove_dependency(child.idx, old_p, dirty::OPACITY);
            self.dirty.mark(old_p, dirty::TOPOLOGY);
        }

        // Now add as child of new parent (inline the logic to avoid double-validate).
        let p = new_parent.idx;
        let c = child.idx;
        self.parent[c as usize] = p;
        self.prev_sibling[c as usize] = INVALID;
        self.next_sibling[c as usize] = INVALID;

        if self.first_child[p as usize] == INVALID {
            self.first_child[p as usize] = c;
        } else {
            let mut last = self.first_child[p as usize];
            while self.next_sibling[last as usize] != INVALID {
                last = self.next_sibling[last as usize];
            }
            self.next_sibling[last as usize] = c;
            self.prev_sibling[c as usize] = last;
        }

        let _ = self.dirty.add_dependency(c, p, dirty::TRANSFORM);
        let _ = self.dirty.add_dependency(c, p, dirty::OPACITY);

        self.traversal_dirty = true;
        self.mark_inherited_dirty(c);
        self.dirty.mark(p, dirty::TOPOLOGY);
    }

    /// Inserts `child` before `sibling` in the sibling list.
    ///
    /// `child` must not already have a parent. `sibling` must have a parent.
    ///
    /// # Panics
    ///
    /// Panics if handles are stale, `child` already has a parent, or `sibling`
    /// has no parent.
    pub fn insert_before(&mut self, child: LayerId, sibling: LayerId) {
        self.validate(child);
        self.validate(sibling);
        let c = child.idx;
        let s = sibling.idx;
        assert!(
            self.parent[c as usize] == INVALID,
            "child already has a parent"
        );
        let p = self.parent[s as usize];
        assert!(p != INVALID, "sibling has no parent");

        self.parent[c as usize] = p;
        self.next_sibling[c as usize] = s;
        self.prev_sibling[c as usize] = self.prev_sibling[s as usize];

        if self.prev_sibling[s as usize] != INVALID {
            self.next_sibling[self.prev_sibling[s as usize] as usize] = c;
        } else {
            // `sibling` was the first child.
            self.first_child[p as usize] = c;
        }
        self.prev_sibling[s as usize] = c;

        let _ = self.dirty.add_dependency(c, p, dirty::TRANSFORM);
        let _ = self.dirty.add_dependency(c, p, dirty::OPACITY);

        self.traversal_dirty = true;
        self.mark_inherited_dirty(c);
        self.dirty.mark(p, dirty::TOPOLOGY);
    }

    /// Returns the parent of a layer, if any.
    #[must_use]
    pub fn parent(&self, id: LayerId) -> Option<LayerId> {
        self.validate(id);
        let p = self.parent[id.idx as usize];
        if p == INVALID {
            None
        } else {
            Some(LayerId {
                idx: p,
                generation: self.generation[p as usize],
            })
        }
    }

    /// Returns an iterator over the direct children of a layer.
    #[must_use]
    pub fn children(&self, id: LayerId) -> Children<'_> {
        self.validate(id);
        Children::new(self, self.first_child[id.idx as usize])
    }

    /// Returns the raw slot indices of root layers (those with no parent).
    ///
    /// Roots are layers whose parent is [`INVALID`] and that are not in the
    /// free list.
    #[must_use]
    pub fn roots(&self) -> Vec<LayerId> {
        let mut roots = Vec::new();
        for idx in 0..self.len {
            if self.parent[idx as usize] == INVALID && !self.free_list.contains(&idx) {
                roots.push(LayerId {
                    idx,
                    generation: self.generation[idx as usize],
                });
            }
        }
        roots
    }

    // -- Property getters (read-only, no dirty marking) --

    /// Returns the local transform of a layer.
    #[must_use]
    pub fn local_transform(&self, id: LayerId) -> Transform3d {
        self.validate(id);
        self.local_transform[id.idx as usize]
    }

    /// Returns the local opacity of a layer.
    #[must_use]
    pub fn local_opacity(&self, id: LayerId) -> f32 {
        self.validate(id);
        self.local_opacity[id.idx as usize]
    }

    /// Returns the clip shape of a layer.
    #[must_use]
    pub fn clip(&self, id: LayerId) -> Option<ClipShape> {
        self.validate(id);
        self.clip[id.idx as usize]
    }

    /// Returns the surface content of a layer.
    #[must_use]
    pub fn content(&self, id: LayerId) -> Option<SurfaceId> {
        self.validate(id);
        self.content[id.idx as usize]
    }

    /// Returns the flags of a layer.
    #[must_use]
    pub fn flags(&self, id: LayerId) -> LayerFlags {
        self.validate(id);
        self.flags[id.idx as usize]
    }

    /// Returns the bounds (width × height) of a layer.
    #[must_use]
    pub fn bounds(&self, id: LayerId) -> Size {
        self.validate(id);
        self.bounds[id.idx as usize]
    }

    /// Returns the optional hit-test rect of a layer.
    ///
    /// When `Some`, [`hit_test`](Self::hit_test) checks containment against
    /// this rect instead of the layer's full bounds. When `None` (the
    /// default), the full bounds are used.
    #[must_use]
    pub fn hit_rect(&self, id: LayerId) -> Option<Rect> {
        self.validate(id);
        self.hit_rect[id.idx as usize]
    }

    /// Returns the computed world transform of a layer.
    ///
    /// Only valid after [`evaluate`](Self::evaluate) has been called.
    #[must_use]
    pub fn world_transform(&self, id: LayerId) -> Transform3d {
        self.validate(id);
        self.world_transform[id.idx as usize]
    }

    /// Returns the computed effective opacity of a layer.
    ///
    /// Only valid after [`evaluate`](Self::evaluate) has been called.
    #[must_use]
    pub fn effective_opacity(&self, id: LayerId) -> f32 {
        self.validate(id);
        self.effective_opacity[id.idx as usize]
    }

    /// Returns whether the layer is effectively hidden (including by an
    /// ancestor's hidden flag).
    ///
    /// Only valid after [`evaluate`](Self::evaluate) has been called.
    #[must_use]
    pub fn effective_hidden(&self, id: LayerId) -> bool {
        self.validate(id);
        self.effective_hidden[id.idx as usize]
    }

    // -- Mutation API (auto-marks dirty) --

    /// Sets the local transform of a layer.
    ///
    /// Marks the TRANSFORM channel dirty with eager propagation to descendants.
    pub fn set_transform(&mut self, id: LayerId, transform: Transform3d) {
        self.validate(id);
        self.local_transform[id.idx as usize] = transform;
        self.dirty.mark_with(id.idx, dirty::TRANSFORM, &EagerPolicy);
    }

    /// Sets the local opacity of a layer.
    ///
    /// Marks the OPACITY channel dirty with eager propagation to descendants.
    pub fn set_opacity(&mut self, id: LayerId, opacity: f32) {
        self.validate(id);
        self.local_opacity[id.idx as usize] = opacity;
        self.dirty.mark_with(id.idx, dirty::OPACITY, &EagerPolicy);
    }

    /// Sets the clip shape of a layer.
    pub fn set_clip(&mut self, id: LayerId, clip: Option<ClipShape>) {
        self.validate(id);
        self.clip[id.idx as usize] = clip;
        self.dirty.mark(id.idx, dirty::CLIP);
    }

    /// Sets the surface content of a layer.
    pub fn set_content(&mut self, id: LayerId, content: Option<SurfaceId>) {
        self.validate(id);
        self.content[id.idx as usize] = content;
        self.dirty.mark(id.idx, dirty::CONTENT);
    }

    /// Sets the flags of a layer.
    pub fn set_flags(&mut self, id: LayerId, flags: LayerFlags) {
        self.validate(id);
        self.flags[id.idx as usize] = flags;
        // Flags can affect both transform computation (hidden) and topology.
        self.dirty.mark_with(id.idx, dirty::TRANSFORM, &EagerPolicy);
    }

    /// Sets the bounds (width × height) of a layer.
    pub fn set_bounds(&mut self, id: LayerId, bounds: Size) {
        self.validate(id);
        self.bounds[id.idx as usize] = bounds;
        self.dirty.mark(id.idx, dirty::BOUNDS);
    }

    /// Sets an optional hit-test rect for a layer (in local coordinates).
    ///
    /// When set, [`hit_test`](Self::hit_test) checks containment against this
    /// rect instead of the full bounds. Use this when the layer's surface is
    /// larger than its interactive area (e.g. to accommodate shadows or glow).
    ///
    /// No dirty channel is marked — hit testing is a read-only query.
    pub fn set_hit_rect(&mut self, id: LayerId, hit_rect: Option<Rect>) {
        self.validate(id);
        self.hit_rect[id.idx as usize] = hit_rect;
    }

    // -- Raw-index accessors for backends --
    //
    // These accept raw slot indices (as found in `FrameChanges`) rather than
    // `LayerId` handles, skipping generation validation. Only use with indices
    // that came from `FrameChanges` or `traversal_order()`.

    /// Returns the computed world transform at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn world_transform_at(&self, idx: u32) -> Transform3d {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.world_transform[idx as usize]
    }

    /// Returns the local (non-inherited) transform at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn local_transform_at(&self, idx: u32) -> Transform3d {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.local_transform[idx as usize]
    }

    /// Returns the local (non-inherited) opacity at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn local_opacity_at(&self, idx: u32) -> f32 {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.local_opacity[idx as usize]
    }

    /// Returns the computed effective opacity at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn effective_opacity_at(&self, idx: u32) -> f32 {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.effective_opacity[idx as usize]
    }

    /// Returns whether the layer at raw slot `idx` is effectively hidden.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn effective_hidden_at(&self, idx: u32) -> bool {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.effective_hidden[idx as usize]
    }

    /// Returns the clip shape at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn clip_at(&self, idx: u32) -> Option<ClipShape> {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.clip[idx as usize]
    }

    /// Returns the surface content at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn content_at(&self, idx: u32) -> Option<SurfaceId> {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.content[idx as usize]
    }

    /// Returns the flags at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn flags_at(&self, idx: u32) -> LayerFlags {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.flags[idx as usize]
    }

    /// Returns the bounds at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn bounds_at(&self, idx: u32) -> Size {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.bounds[idx as usize]
    }

    /// Returns the hit-test rect at raw slot `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn hit_rect_at(&self, idx: u32) -> Option<Rect> {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        self.hit_rect[idx as usize]
    }

    /// Returns the raw parent slot index at raw slot `idx`, or `None` if
    /// the layer is a root (has no parent).
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    #[must_use]
    pub fn parent_at(&self, idx: u32) -> Option<u32> {
        assert!(
            idx < self.len,
            "slot index {idx} out of range (len {})",
            self.len
        );
        let p = self.parent[idx as usize];
        if p == INVALID { None } else { Some(p) }
    }

    // -- Internal helpers --

    /// Panics if the handle is stale.
    fn validate(&self, id: LayerId) {
        assert!(
            id.idx < self.len && self.generation[id.idx as usize] == id.generation,
            "stale LayerId: {id:?} (current gen: {})",
            if id.idx < self.len {
                self.generation[id.idx as usize]
            } else {
                u32::MAX
            }
        );
    }

    fn reorder_parent(&self, idx: u32) -> u32 {
        let parent = self.parent[idx as usize];
        assert!(parent != INVALID, "layer has no parent");
        parent
    }

    fn shared_reorder_parent(&self, layer: u32, sibling: u32) -> u32 {
        let parent = self.reorder_parent(layer);
        let sibling_parent = self.reorder_parent(sibling);
        assert_eq!(parent, sibling_parent, "layers must share a parent");
        parent
    }

    fn mark_topology_reordered(&mut self, parent: u32) {
        self.traversal_dirty = true;
        self.dirty.mark(parent, dirty::TOPOLOGY);
    }

    fn prepend_unlinked_child(&mut self, parent: u32, child: u32) {
        self.parent[child as usize] = parent;
        self.prev_sibling[child as usize] = INVALID;
        self.next_sibling[child as usize] = self.first_child[parent as usize];

        if self.first_child[parent as usize] != INVALID {
            self.prev_sibling[self.first_child[parent as usize] as usize] = child;
        }
        self.first_child[parent as usize] = child;
    }

    fn append_unlinked_child(&mut self, parent: u32, child: u32) {
        self.parent[child as usize] = parent;
        self.prev_sibling[child as usize] = INVALID;
        self.next_sibling[child as usize] = INVALID;

        if self.first_child[parent as usize] == INVALID {
            self.first_child[parent as usize] = child;
            return;
        }

        let mut last = self.first_child[parent as usize];
        while self.next_sibling[last as usize] != INVALID {
            last = self.next_sibling[last as usize];
        }
        self.next_sibling[last as usize] = child;
        self.prev_sibling[child as usize] = last;
    }

    fn insert_unlinked_before(&mut self, child: u32, sibling: u32, parent: u32) {
        self.parent[child as usize] = parent;
        self.next_sibling[child as usize] = sibling;
        self.prev_sibling[child as usize] = self.prev_sibling[sibling as usize];

        if self.prev_sibling[sibling as usize] != INVALID {
            self.next_sibling[self.prev_sibling[sibling as usize] as usize] = child;
        } else {
            self.first_child[parent as usize] = child;
        }
        self.prev_sibling[sibling as usize] = child;
    }

    fn insert_unlinked_after(&mut self, child: u32, sibling: u32, parent: u32) {
        self.parent[child as usize] = parent;
        self.prev_sibling[child as usize] = sibling;
        self.next_sibling[child as usize] = self.next_sibling[sibling as usize];

        if self.next_sibling[sibling as usize] != INVALID {
            self.prev_sibling[self.next_sibling[sibling as usize] as usize] = child;
        }
        self.next_sibling[sibling as usize] = child;
    }

    fn collect_subtree_postorder(&self, idx: u32, out: &mut Vec<u32>) {
        let mut child = self.first_child[idx as usize];
        while child != INVALID {
            let next = self.next_sibling[child as usize];
            self.collect_subtree_postorder(child, out);
            child = next;
        }
        out.push(idx);
    }

    /// Removes `idx` from its parent's child list without touching dirty state.
    fn unlink_from_parent(&mut self, idx: u32) {
        let p = self.parent[idx as usize];
        let prev = self.prev_sibling[idx as usize];
        let next = self.next_sibling[idx as usize];

        if prev != INVALID {
            self.next_sibling[prev as usize] = next;
        } else {
            // Was first child.
            self.first_child[p as usize] = next;
        }

        if next != INVALID {
            self.prev_sibling[next as usize] = prev;
        }

        self.parent[idx as usize] = INVALID;
        self.prev_sibling[idx as usize] = INVALID;
        self.next_sibling[idx as usize] = INVALID;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn child_order(store: &LayerStore, parent: LayerId) -> Vec<LayerId> {
        store.children(parent).collect()
    }

    #[test]
    fn create_and_destroy() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        assert!(store.is_alive(id));
        store.destroy_layer(id);
        assert!(!store.is_alive(id));
    }

    #[test]
    fn len_counts_live_layers() {
        let mut store = LayerStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        let first = store.create_layer();
        let second = store.create_layer();
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());

        store.destroy_layer(first);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let reused = store.create_layer();
        assert_eq!(store.len(), 2);
        assert_ne!(first.generation(), reused.generation());

        store.destroy_layer(second);
        store.destroy_layer(reused);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn generation_prevents_stale_access() {
        let mut store = LayerStore::new();
        let id1 = store.create_layer();
        store.destroy_layer(id1);
        let id2 = store.create_layer();
        // id2 reuses the same slot but has a different generation.
        assert!(!store.is_alive(id1));
        assert!(store.is_alive(id2));
        assert_eq!(id1.idx, id2.idx);
        assert_ne!(id1.generation, id2.generation);
    }

    #[test]
    fn add_child_and_query() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child1 = store.create_layer();
        let child2 = store.create_layer();

        store.add_child(parent, child1);
        store.add_child(parent, child2);

        assert_eq!(store.parent(child1), Some(parent));
        assert_eq!(store.parent(child2), Some(parent));

        let kids: Vec<_> = store.children(parent).collect();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], child1);
        assert_eq!(kids[1], child2);
    }

    #[test]
    fn remove_from_parent_works() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();

        store.add_child(parent, child);
        assert_eq!(store.parent(child), Some(parent));

        store.remove_from_parent(child);
        assert_eq!(store.parent(child), None);
        assert!(store.children(parent).next().is_none());
    }

    #[test]
    fn insert_before_works() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();
        let c = store.create_layer();

        store.add_child(parent, a);
        store.add_child(parent, c);
        store.insert_before(b, c);

        let kids: Vec<_> = store.children(parent).collect();
        assert_eq!(kids, vec![a, b, c]);
    }

    #[test]
    fn reorder_apis_update_child_order() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();
        let c = store.create_layer();

        store.add_child(parent, a);
        store.add_child(parent, b);
        store.add_child(parent, c);
        assert_eq!(child_order(&store, parent), vec![a, b, c]);

        store.move_to_front(a);
        assert_eq!(child_order(&store, parent), vec![b, c, a]);

        store.move_to_back(a);
        assert_eq!(child_order(&store, parent), vec![a, b, c]);

        store.move_after(a, c);
        assert_eq!(child_order(&store, parent), vec![b, c, a]);

        store.move_before(a, b);
        assert_eq!(child_order(&store, parent), vec![a, b, c]);

        store.move_before(c, a);
        assert_eq!(child_order(&store, parent), vec![c, a, b]);

        store.move_after(c, b);
        assert_eq!(child_order(&store, parent), vec![a, b, c]);
    }

    #[test]
    fn reorder_marks_topology_without_inherited_dirty() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();

        store.add_child(parent, a);
        store.add_child(parent, b);
        let _ = store.evaluate();

        store.move_to_front(a);
        let changes = store.evaluate();
        assert!(changes.topology_changed);
        assert!(changes.transforms.is_empty());
        assert!(changes.opacities.is_empty());
    }

    #[test]
    fn redundant_reorder_does_not_mark_topology() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();

        store.add_child(parent, a);
        store.add_child(parent, b);
        let _ = store.evaluate();

        store.move_before(a, b);
        store.move_after(b, a);
        store.move_to_back(a);
        store.move_to_front(b);

        let changes = store.evaluate();
        assert!(!changes.topology_changed);
    }

    #[test]
    #[should_panic(expected = "layers must share a parent")]
    fn move_before_requires_shared_parent() {
        let mut store = LayerStore::new();
        let first_parent = store.create_layer();
        let second_parent = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();

        store.add_child(first_parent, a);
        store.add_child(second_parent, b);
        store.move_before(a, b);
    }

    #[test]
    #[should_panic(expected = "layer has no parent")]
    fn move_to_front_requires_parent() {
        let mut store = LayerStore::new();
        let root = store.create_layer();
        store.move_to_front(root);
    }

    #[test]
    fn reparent_works() {
        let mut store = LayerStore::new();
        let p1 = store.create_layer();
        let p2 = store.create_layer();
        let child = store.create_layer();

        store.add_child(p1, child);
        assert_eq!(store.parent(child), Some(p1));

        store.reparent(child, p2);
        assert_eq!(store.parent(child), Some(p2));
        assert!(store.children(p1).next().is_none());
    }

    #[test]
    fn roots_returns_parentless_layers() {
        let mut store = LayerStore::new();
        let a = store.create_layer();
        let b = store.create_layer();
        let c = store.create_layer();

        store.add_child(a, c);

        let roots = store.roots();
        assert!(roots.contains(&a));
        assert!(roots.contains(&b));
        assert!(!roots.contains(&c));
    }

    #[test]
    #[should_panic(expected = "cannot destroy layer with children")]
    fn destroy_with_children_panics() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();
        store.add_child(parent, child);
        store.destroy_layer(parent);
    }

    #[test]
    fn destroy_subtree_unlinks_from_parent_and_reports_postorder() {
        let mut store = LayerStore::new();
        let root = store.create_layer();
        let branch = store.create_layer();
        let sibling = store.create_layer();
        let leaf = store.create_layer();

        store.add_child(root, branch);
        store.add_child(root, sibling);
        store.add_child(branch, leaf);
        let _ = store.evaluate();

        store.destroy_subtree(branch);

        assert!(store.is_alive(root));
        assert!(store.is_alive(sibling));
        assert!(!store.is_alive(branch));
        assert!(!store.is_alive(leaf));
        assert_eq!(store.len(), 2);
        assert_eq!(child_order(&store, root), vec![sibling]);

        let changes = store.evaluate();
        assert_eq!(changes.removed, vec![leaf.idx, branch.idx]);
        assert!(changes.topology_changed);
    }

    #[test]
    #[should_panic(expected = "stale LayerId")]
    fn destroyed_handle_panics_on_get_transform() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.destroy_layer(id);
        let _ = store.world_transform(id);
    }

    #[test]
    #[should_panic(expected = "stale LayerId")]
    fn destroyed_handle_panics_on_set_transform() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.destroy_layer(id);
        store.set_transform(id, Transform3d::IDENTITY);
    }

    #[test]
    #[should_panic(expected = "stale LayerId")]
    fn destroyed_handle_panics_on_add_child() {
        let mut store = LayerStore::new();
        let root = store.create_layer();
        let id = store.create_layer();
        store.destroy_layer(id);
        store.add_child(root, id);
    }

    #[test]
    #[should_panic(expected = "stale LayerId")]
    fn destroyed_handle_panics_on_parent() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.destroy_layer(id);
        let _ = store.parent(id);
    }

    #[test]
    fn set_transform_marks_dirty() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_transform(id, Transform3d::from_scale(2.0, 2.0, 2.0));
        assert_eq!(
            store.local_transform(id),
            Transform3d::from_scale(2.0, 2.0, 2.0)
        );
    }

    #[test]
    fn set_opacity_marks_dirty() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        // Consume initial creation dirtiness.
        let _ = store.evaluate();

        store.set_opacity(id, 0.5);
        let changes = store.evaluate();
        assert!(
            changes.opacities.contains(&id.idx),
            "opacity channel should contain the layer"
        );
    }

    #[test]
    fn set_clip_marks_dirty() {
        use crate::layer::ClipShape;

        let mut store = LayerStore::new();
        let id = store.create_layer();
        let _ = store.evaluate();

        store.set_clip(id, Some(ClipShape::Rect(Rect::new(0.0, 0.0, 100.0, 100.0))));
        let changes = store.evaluate();
        assert!(
            changes.clips.contains(&id.idx),
            "clip channel should contain the layer"
        );
    }

    #[test]
    fn set_content_marks_dirty() {
        use crate::layer::SurfaceId;

        let mut store = LayerStore::new();
        let id = store.create_layer();
        let _ = store.evaluate();

        store.set_content(id, Some(SurfaceId(42)));
        let changes = store.evaluate();
        assert!(
            changes.content.contains(&id.idx),
            "content channel should contain the layer"
        );
    }

    #[test]
    fn set_flags_marks_dirty() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        let _ = store.evaluate();

        store.set_flags(id, LayerFlags { hidden: true });
        let changes = store.evaluate();
        assert!(
            changes.transforms.contains(&id.idx),
            "flags marks TRANSFORM channel"
        );
    }

    #[test]
    fn bounds_default_is_zero() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        let b = store.bounds(id);
        assert_eq!(b.width, 0.0);
        assert_eq!(b.height, 0.0);
    }

    #[test]
    fn set_bounds_marks_dirty() {
        use kurbo::Size;

        let mut store = LayerStore::new();
        let id = store.create_layer();
        let _ = store.evaluate();

        store.set_bounds(id, Size::new(320.0, 240.0));
        let changes = store.evaluate();
        assert!(
            changes.bounds.contains(&id.idx),
            "bounds channel should contain the layer"
        );
    }

    #[test]
    fn parent_at_root_is_none() {
        let mut store = LayerStore::new();
        let root = store.create_layer();
        assert_eq!(store.parent_at(root.idx), None);
    }

    #[test]
    fn parent_at_returns_parent_slot() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();
        store.add_child(parent, child);
        assert_eq!(store.parent_at(child.idx), Some(parent.idx));
    }

    #[test]
    fn parent_at_reflects_reparent() {
        let mut store = LayerStore::new();
        let a = store.create_layer();
        let b = store.create_layer();
        let child = store.create_layer();
        store.add_child(a, child);
        assert_eq!(store.parent_at(child.idx), Some(a.idx));

        store.reparent(child, b);
        assert_eq!(store.parent_at(child.idx), Some(b.idx));
    }

    #[test]
    fn parent_at_none_after_remove() {
        let mut store = LayerStore::new();
        let parent = store.create_layer();
        let child = store.create_layer();
        store.add_child(parent, child);
        store.remove_from_parent(child);
        assert_eq!(store.parent_at(child.idx), None);
    }

    #[test]
    fn local_transform_at_default_is_identity() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        assert_eq!(store.local_transform_at(id.idx), Transform3d::IDENTITY);
    }

    #[test]
    fn local_transform_at_returns_set_value() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        let xf = Transform3d::from_translation(7.0, 3.0, 0.0);
        store.set_transform(id, xf);
        assert_eq!(store.local_transform_at(id.idx), xf);
    }

    #[test]
    fn local_opacity_at_default_is_one() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        assert!((store.local_opacity_at(id.idx) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn local_opacity_at_returns_set_value() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_opacity(id, 0.42);
        assert!((store.local_opacity_at(id.idx) - 0.42).abs() < f32::EPSILON);
    }
}
