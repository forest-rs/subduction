// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer and surface identity types.

use alloc::vec::Vec;
use core::fmt;

/// Sentinel value indicating "no layer" or "no surface" in index fields.
pub const INVALID: u32 = u32::MAX;

/// A handle to a layer in a [`LayerStore`](super::LayerStore).
///
/// This is the public identity for a compositor node. Callers use `LayerId` to
/// mutate topology and layer properties such as transform, opacity, bounds,
/// clips, and attached content.
///
/// Contains both a slot index and a generation counter so that stale handles
/// can be detected after a layer is destroyed and the slot is reused.
/// [`index`](Self::index) exposes the raw storage slot for diagnostics and
/// backend interop, but the slot alone is not a lifetime-safe layer handle.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId {
    /// Slot index into the store's arrays.
    pub(crate) idx: u32,
    /// Generation counter — must match the store's generation for this slot.
    pub(crate) generation: u32,
}

impl LayerId {
    /// Returns the raw slot index (for diagnostics only).
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.idx
    }

    /// Returns the generation counter.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl fmt::Debug for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LayerId({}@gen{})", self.idx, self.generation)
    }
}

/// An opaque handle to externally owned content.
///
/// A `SurfaceId` identifies a host-owned renderable resource such as cached
/// widget contents, scroll content, a video plane, or a native platform layer.
///
/// This is content identity, not layer identity. The same surface can be
/// detached from one layer and attached to another without becoming a different
/// surface. Conversely, destroying a layer does not imply that the host should
/// destroy the resource keyed by its `SurfaceId`.
///
/// [`LayerStore`](super::LayerStore) stores `SurfaceId` only as an attachment
/// token in a layer's content slot. Presenters use that token to find or expose
/// backend resources, but subduction core does not create, destroy, retain, or
/// otherwise own the underlying surface resource.
///
/// IDs allocated by [`SurfaceIds`] include a generation counter. Releasing an ID
/// invalidates stale tokens before their slot can be reused.
///
/// A live `SurfaceId` should be attached to at most one live layer in an
/// evaluated tree. Presenters maintain transient `SurfaceId`-to-layer mappings
/// for the current frame; callers that move content should detach it from the
/// old layer and attach it to the new one before evaluation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceId {
    idx: u32,
    generation: u32,
}

impl SurfaceId {
    /// Creates a surface ID from raw parts.
    ///
    /// Prefer [`SurfaceIds::create`] for ordinary allocation. This constructor
    /// exists for integration with external registries that already provide
    /// their own uniqueness and lifetime discipline.
    ///
    /// # Panics
    ///
    /// Panics if `index` is [`INVALID`].
    #[must_use]
    pub fn from_raw_parts(index: u32, generation: u32) -> Self {
        assert!(index != INVALID, "surface index cannot be INVALID");
        Self {
            idx: index,
            generation,
        }
    }

    /// Returns the raw slot index.
    ///
    /// This is intended for diagnostics and external registry integration.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.idx
    }

    /// Returns the generation counter.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl fmt::Debug for SurfaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceId({}@gen{})", self.idx, self.generation)
    }
}

/// Allocates stable [`SurfaceId`] tokens without owning surface resources.
///
/// This is a small identity table for hosts and UI frameworks that need stable
/// keys for rendered scenes, cached widgets, scroll content, video/native
/// layers, or other externally managed surfaces. It tracks which IDs are live
/// and bumps the generation on release so stale tokens stop comparing equal to
/// future allocations from the same slot.
///
/// `SurfaceIds` is not a resource registry. It does not know whether a surface
/// has an associated GPU texture, platform surface, cached scene, or widget
/// object. It only provides freshness for the key used by such registries.
///
/// The caller remains responsible for the resource registry keyed by
/// [`SurfaceId`], for detaching released IDs from layers, and for destroying any
/// associated GPU or platform resources.
#[derive(Debug, Default)]
pub struct SurfaceIds {
    generation: Vec<u32>,
    live: Vec<bool>,
    free_list: Vec<u32>,
    len: usize,
}

impl SurfaceIds {
    /// Creates an empty surface ID allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            generation: Vec::new(),
            live: Vec::new(),
            free_list: Vec::new(),
            len: 0,
        }
    }

    /// Allocates a live surface ID.
    ///
    /// The returned ID is suitable for use as a resource-registry key and for
    /// attachment to a layer via [`LayerStore::set_content`](super::LayerStore::set_content).
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX - 1` surface ID slots are allocated over
    /// the allocator's lifetime.
    #[must_use]
    pub fn create(&mut self) -> SurfaceId {
        let idx = if let Some(idx) = self.free_list.pop() {
            self.live[idx as usize] = true;
            idx
        } else {
            let idx = u32::try_from(self.generation.len()).expect("too many surface ids");
            assert!(idx != INVALID, "too many surface ids");
            self.generation.push(0);
            self.live.push(true);
            idx
        };

        self.len += 1;
        SurfaceId {
            idx,
            generation: self.generation[idx as usize],
        }
    }

    /// Releases a live surface ID and invalidates stale copies of it.
    ///
    /// Returns `true` when `id` was live and is now released. Returns `false`
    /// when `id` is stale, out of range, or already released.
    ///
    /// Releasing an ID does not detach it from any layer and does not destroy
    /// the host resource keyed by that ID.
    #[must_use]
    pub fn release(&mut self, id: SurfaceId) -> bool {
        if !self.is_alive(id) {
            return false;
        }

        let idx = id.idx as usize;
        self.live[idx] = false;
        self.generation[idx] = self.generation[idx].wrapping_add(1);
        self.free_list.push(id.idx);
        self.len -= 1;
        true
    }

    /// Returns whether the given ID is currently live in this allocator.
    #[must_use]
    pub fn is_alive(&self, id: SurfaceId) -> bool {
        let Some((&generation, &live)) = self
            .generation
            .get(id.idx as usize)
            .zip(self.live.get(id.idx as usize))
        else {
            return false;
        };

        live && generation == id.generation
    }

    /// Returns the number of live surface IDs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether there are no live surface IDs.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{SurfaceId, SurfaceIds};

    #[test]
    fn raw_surface_id_parts_round_trip() {
        let id = SurfaceId::from_raw_parts(7, 3);
        assert_eq!(id.index(), 7);
        assert_eq!(id.generation(), 3);
    }

    #[test]
    fn surface_ids_allocate_live_ids() {
        let mut ids = SurfaceIds::new();

        let a = ids.create();
        let b = ids.create();

        assert_eq!(ids.len(), 2);
        assert!(ids.is_alive(a));
        assert!(ids.is_alive(b));
        assert_ne!(a, b);
    }

    #[test]
    fn surface_ids_release_invalidates_id() {
        let mut ids = SurfaceIds::new();
        let id = ids.create();

        assert!(ids.release(id));

        assert!(ids.is_empty());
        assert!(!ids.is_alive(id));
        assert!(!ids.release(id));
    }

    #[test]
    fn surface_ids_reuse_slot_with_new_generation() {
        let mut ids = SurfaceIds::new();
        let old = ids.create();

        assert!(ids.release(old));
        let new = ids.create();

        assert_eq!(new.index(), old.index());
        assert_ne!(new.generation(), old.generation());
        assert!(!ids.is_alive(old));
        assert!(ids.is_alive(new));
    }
}
