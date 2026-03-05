// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland subsurface presenter.
//!
//! [`WaylandPresenter`] implements [`Presenter`] by maintaining one
//! `wl_surface` + `wl_subsurface` per layer, all parented to a single root
//! surface. Geometry is driven by [`world_transform_at()`] — the same flat
//! topology pattern used by `DomPresenter` and `LayerPresenter`.
//!
//! The presenter is protocol-only: it positions subsurfaces but does **not**
//! attach buffers or manage rendering. Content producers (wgpu, SHM, dmabuf)
//! attach buffers to per-layer surfaces independently and call
//! `surface.commit()` on each.
//!
//! All subsurfaces are created in **sync mode**, so their state latches
//! atomically when the root surface is committed. The root surface **must** be
//! the same surface registered with [`WaylandState::set_surface`] so that
//! [`WaylandState::commit_frame`] triggers the latch. The constructor
//! enforces this by reading the root surface from [`WaylandState`]
//! directly.
//!
//! # Unsupported change channels
//!
//! The following [`FrameChanges`] channels have no effect in this presenter
//! because Wayland's core protocol lacks the corresponding primitives. Callers
//! that need these capabilities should either handle them in the content
//! pipeline or gate on protocol extensions:
//!
//! - **Opacities**: requires `wp_alpha_modifier` (not yet wired). Content
//!   producers should bake effective opacity into their buffers.
//! - **Bounds**: requires `wp_viewporter` (not yet wired). Subsurface size is
//!   determined by the attached buffer.
//! - **Clips**: requires compositor-side masking (no standard protocol).
//!
//! # Usage (owned mode)
//!
//! ```rust,ignore
//! use subduction_backend_wayland::{OwnedQueueMode, WaylandPresenterConfig};
//! use subduction_core::backend::Presenter;
//! use subduction_core::layer::LayerStore;
//!
//! // After bootstrap + set_surface:
//! let mut presenter = mode
//!     .create_presenter(WaylandPresenterConfig::default())
//!     .unwrap();
//!
//! let mut store = LayerStore::new();
//! // ... build layer tree, set transforms, content, etc. ...
//!
//! loop {
//!     mode.blocking_dispatch().unwrap();
//!
//!     while let Some(_tick) = mode.poll_tick() {
//!         // Evaluate the layer tree.
//!         let changes = store.evaluate();
//!
//!         // Update subsurface positions and stacking order.
//!         presenter.apply(&store, &changes);
//!
//!         // Render and attach buffers to per-layer surfaces via
//!         // presenter.get_surface(slot) or
//!         // presenter.surface_for_content(id).
//!
//!         // Root commit latches all synced subsurface state atomically.
//!         mode.commit_frame().unwrap();
//!     }
//! }
//! ```
//!
//! [`world_transform_at()`]: subduction_core::layer::LayerStore::world_transform_at
//! [`WaylandState::set_surface`]: crate::WaylandState::set_surface
//! [`WaylandState::commit_frame`]: crate::WaylandState::commit_frame
//! [`WaylandState`]: crate::WaylandState
//! [`FrameChanges`]: subduction_core::layer::FrameChanges

use std::collections::HashMap;

use subduction_core::backend::Presenter;
use subduction_core::layer::{FrameChanges, LayerStore, SurfaceId};

use wayland_client::protocol::{wl_compositor, wl_subcompositor, wl_subsurface, wl_surface};
use wayland_client::{Dispatch, QueueHandle};

use crate::event_loop::{CreatePresenterError, WaylandState};
use crate::protocol::{LayerSubsurfaceData, LayerSurfaceData};

/// How fractional pixel positions are rounded to integer subsurface
/// coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PositionRounding {
    /// Round toward negative infinity.
    Floor,
    /// Round to nearest integer (default).
    #[default]
    Round,
    /// Round toward positive infinity.
    Ceil,
}

/// Configuration for [`WaylandPresenter`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct WaylandPresenterConfig {
    /// How fractional pixel positions are rounded.
    pub rounding: PositionRounding,
}

/// Internal per-slot state: one `wl_surface` + `wl_subsurface`.
#[derive(Debug)]
struct Entry {
    surface: wl_surface::WlSurface,
    subsurface: wl_subsurface::WlSubsurface,
}

/// A [`Presenter`] that maps each layer to a Wayland `wl_subsurface`.
///
/// All subsurfaces are children of a single root surface, positioned via
/// translation from [`world_transform_at()`]. Rotation, scale, and
/// perspective components of the world transform are **not** applied
/// (Wayland subsurfaces only support integer translation).
///
/// # Type parameter
///
/// `D` is the dispatch state type — either `WaylandState` (owned mode) or
/// the host's state struct (embedded mode). The presenter stores a
/// `QueueHandle<D>` so it can create protocol objects during [`apply`].
///
/// # Content mapping
///
/// Like [`WgpuPresenter`], the presenter maintains a bidirectional mapping
/// between [`SurfaceId`] and slot index. Use [`surface_for_content`] to
/// look up the `wl_surface` for a given content ID.
///
/// # Hidden layers
///
/// When a layer transitions to hidden, the presenter detaches its buffer
/// (`wl_surface.attach(null)` + commit) to unmap it from the compositor.
/// Content producers do **not** need to re-attach on unhide — the next
/// `attach` + `commit` from the content producer restores the surface
/// naturally.
///
/// [`world_transform_at()`]: subduction_core::layer::LayerStore::world_transform_at
/// [`WgpuPresenter`]: https://docs.rs/subduction_backend_wgpu
/// [`apply`]: Presenter::apply
/// [`surface_for_content`]: Self::surface_for_content
#[derive(Debug)]
pub struct WaylandPresenter<D> {
    root: wl_surface::WlSurface,
    compositor: wl_compositor::WlCompositor,
    subcompositor: wl_subcompositor::WlSubcompositor,
    qh: QueueHandle<D>,
    entries: Vec<Option<Entry>>,
    surface_to_slot: HashMap<SurfaceId, u32>,
    slot_to_surface: HashMap<u32, SurfaceId>,
    config: WaylandPresenterConfig,
}

impl<D> WaylandPresenter<D>
where
    D: Dispatch<wl_surface::WlSurface, LayerSurfaceData>
        + Dispatch<wl_subsurface::WlSubsurface, LayerSubsurfaceData>
        + 'static,
{
    /// Creates a new presenter.
    ///
    /// The root surface is read from `state` — it must have been registered
    /// via [`WaylandState::set_surface`](crate::WaylandState::set_surface)
    /// before calling this.
    ///
    /// # Errors
    ///
    /// Returns [`CreatePresenterError::NoSurface`] if `state` has no
    /// surface set.
    pub fn new(
        state: &WaylandState,
        compositor: wl_compositor::WlCompositor,
        subcompositor: wl_subcompositor::WlSubcompositor,
        qh: QueueHandle<D>,
        config: WaylandPresenterConfig,
    ) -> Result<Self, CreatePresenterError> {
        let root = state
            .surface()
            .ok_or(CreatePresenterError::NoSurface)?
            .clone();
        Ok(Self {
            root,
            compositor,
            subcompositor,
            qh,
            entries: Vec::new(),
            surface_to_slot: HashMap::new(),
            slot_to_surface: HashMap::new(),
            config,
        })
    }

    /// Returns a reference to the root surface.
    #[must_use]
    pub fn root_surface(&self) -> &wl_surface::WlSurface {
        &self.root
    }

    /// Returns the `wl_surface` for the given slot index, if it exists.
    #[must_use]
    pub fn get_surface(&self, slot: u32) -> Option<&wl_surface::WlSurface> {
        self.entries
            .get(slot as usize)?
            .as_ref()
            .map(|e| &e.surface)
    }

    /// Returns the `wl_surface` for a [`SurfaceId`], if mapped.
    #[must_use]
    pub fn surface_for_content(&self, id: SurfaceId) -> Option<&wl_surface::WlSurface> {
        let &slot = self.surface_to_slot.get(&id)?;
        self.get_surface(slot)
    }

    /// Explicitly destroys all subsurfaces and surfaces managed by this
    /// presenter.
    ///
    /// After calling this, the presenter is empty. Dropping a
    /// `WaylandPresenter` without calling `destroy` also cleans up via
    /// proxy `Drop` impls, but explicit destruction is preferred for
    /// deterministic protocol ordering.
    pub fn destroy(&mut self) {
        for entry in self.entries.drain(..).flatten() {
            entry.subsurface.destroy();
            entry.surface.destroy();
        }
        self.surface_to_slot.clear();
        self.slot_to_surface.clear();
    }

    /// Creates a new entry (surface + subsurface) for the given slot.
    fn create_entry(&self, slot: u32) -> Entry {
        let surface = self
            .compositor
            .create_surface(&self.qh, LayerSurfaceData { slot });
        let subsurface =
            self.subcompositor
                .get_subsurface(&surface, &self.root, &self.qh, LayerSubsurfaceData);
        // Sync mode: state latches on root commit.
        subsurface.set_sync();
        Entry {
            surface,
            subsurface,
        }
    }

    /// Takes an entry out of the slot, leaving `None`.
    fn take_entry(&mut self, slot: u32) -> Option<Entry> {
        self.entries.get_mut(slot as usize)?.take()
    }

    /// Stores an entry at the given slot index, growing the vec if needed.
    fn put_entry(&mut self, slot: u32, entry: Entry) {
        let i = slot as usize;
        if self.entries.len() <= i {
            self.entries.resize_with(i + 1, || None);
        }
        self.entries[i] = Some(entry);
    }

    /// Removes the content mapping for a given slot, if any.
    fn remove_surface_mapping_for_slot(&mut self, slot: u32) {
        if let Some(sid) = self.slot_to_surface.remove(&slot) {
            self.surface_to_slot.remove(&sid);
        }
    }

    /// Removes the reverse mapping for any *other* slot that previously
    /// held `id`, then inserts the new forward + reverse mapping.
    fn set_surface_mapping(&mut self, id: SurfaceId, slot: u32) {
        // If this SurfaceId was previously mapped to a different slot,
        // clean up that slot's reverse entry to keep both directions
        // consistent.
        if let Some(old_slot) = self.surface_to_slot.insert(id, slot)
            && old_slot != slot
        {
            self.slot_to_surface.remove(&old_slot);
        }
        self.slot_to_surface.insert(slot, id);
    }

    /// Detaches the buffer from a surface and commits, unmapping it from
    /// the compositor.
    fn detach_buffer(surface: &wl_surface::WlSurface) {
        surface.attach(None, 0, 0);
        surface.commit();
    }

    /// Rounds a fractional position to an integer according to the config.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "subsurface positions are pixel coordinates that fit in i32"
    )]
    fn round_pos(&self, v: f64) -> i32 {
        match self.config.rounding {
            PositionRounding::Floor => v.floor() as i32,
            PositionRounding::Round => v.round() as i32,
            PositionRounding::Ceil => v.ceil() as i32,
        }
    }
}

impl<D> Presenter for WaylandPresenter<D>
where
    D: Dispatch<wl_surface::WlSurface, LayerSurfaceData>
        + Dispatch<wl_subsurface::WlSubsurface, LayerSubsurfaceData>
        + 'static,
{
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        // 1. Removals
        for &slot in &changes.removed {
            self.remove_surface_mapping_for_slot(slot);
            if let Some(entry) = self.take_entry(slot) {
                entry.subsurface.destroy();
                entry.surface.destroy();
            }
        }

        // 2. Additions
        for &slot in &changes.added {
            let entry = self.create_entry(slot);

            let world = store.world_transform_at(slot);
            let x = self.round_pos(world.cols[3][0]);
            let y = self.round_pos(world.cols[3][1]);
            entry.subsurface.set_position(x, y);

            // If the layer is already hidden at creation time, detach to
            // ensure the surface starts unmapped (no buffer = not composited).
            if store.effective_hidden_at(slot) {
                Self::detach_buffer(&entry.surface);
            }

            self.put_entry(slot, entry);
        }

        // 3. Content mapping (SurfaceId ↔ slot)
        for &slot in &changes.content {
            self.remove_surface_mapping_for_slot(slot);
            if let Some(id) = store.content_at(slot) {
                self.set_surface_mapping(id, slot);
            }
        }

        // 4. Transforms (translation only)
        for &slot in &changes.transforms {
            if let Some(Some(entry)) = self.entries.get(slot as usize) {
                let world = store.world_transform_at(slot);
                let x = self.round_pos(world.cols[3][0]);
                let y = self.round_pos(world.cols[3][1]);
                entry.subsurface.set_position(x, y);
            }
        }

        // 5. Opacities — no-op (see module-level docs for rationale).

        // 6. Hidden: detach buffer to unmap the surface from the compositor.
        for &slot in &changes.hidden {
            if let Some(Some(entry)) = self.entries.get(slot as usize) {
                Self::detach_buffer(&entry.surface);
            }
        }

        // Unhidden: no action needed. The content producer's next
        // attach + commit restores the surface naturally.

        // 7. Bounds — no-op (see module-level docs for rationale).

        // 8. Clips — no-op (see module-level docs for rationale).

        // 9. Topology reorder
        if changes.topology_changed {
            let mut prev: Option<&wl_surface::WlSurface> = None;
            for &slot in store.traversal_order() {
                let Some(Some(entry)) = self.entries.get(slot as usize) else {
                    continue;
                };
                if let Some(prev_surface) = prev {
                    entry.subsurface.place_above(prev_surface);
                }
                prev = Some(&entry.surface);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_loop::WaylandState;
    use subduction_core::layer::LayerStore;
    use wayland_client::backend::ObjectId;
    use wayland_client::protocol::{wl_compositor, wl_subcompositor};
    use wayland_client::{Connection, EventQueue, Proxy};

    /// Creates a socketpair-backed connection, event queue, and queue handle
    /// suitable for sending protocol requests into the void.
    fn test_env() -> (
        Connection,
        EventQueue<WaylandState>,
        QueueHandle<WaylandState>,
    ) {
        let (s1, _s2) = std::os::unix::net::UnixStream::pair().unwrap();
        let conn = Connection::from_socket(s1).unwrap();
        let eq: EventQueue<WaylandState> = conn.new_event_queue();
        let qh = eq.handle();
        (conn, eq, qh)
    }

    /// Creates a `WaylandPresenter<WaylandState>` backed by inert proxies.
    fn test_presenter() -> WaylandPresenter<WaylandState> {
        test_presenter_with_config(WaylandPresenterConfig::default())
    }

    fn test_presenter_with_config(
        config: WaylandPresenterConfig,
    ) -> WaylandPresenter<WaylandState> {
        let (conn, _eq, qh) = test_env();
        let root = wl_surface::WlSurface::from_id(&conn, ObjectId::null()).unwrap();
        let compositor = wl_compositor::WlCompositor::from_id(&conn, ObjectId::null()).unwrap();
        let subcompositor =
            wl_subcompositor::WlSubcompositor::from_id(&conn, ObjectId::null()).unwrap();
        let mut ws = WaylandState::new();
        ws.set_surface(root).unwrap();
        WaylandPresenter::new(&ws, compositor, subcompositor, qh, config).unwrap()
    }

    // -----------------------------------------------------------------------
    // Construction and accessors
    // -----------------------------------------------------------------------

    #[test]
    fn new_presenter_is_empty() {
        let p = test_presenter();
        assert!(p.get_surface(0).is_none());
        assert!(p.surface_for_content(SurfaceId(0)).is_none());
    }

    // -----------------------------------------------------------------------
    // apply(): additions, removals, content mapping
    // -----------------------------------------------------------------------

    #[test]
    fn apply_additions_creates_entries() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let root = store.create_layer();
        let child = store.create_layer();
        store.add_child(root, child);
        let changes = store.evaluate();

        p.apply(&store, &changes);

        // Both slots should now have surfaces.
        assert!(p.get_surface(root.index()).is_some());
        assert!(p.get_surface(child.index()).is_some());
    }

    #[test]
    fn apply_removals_destroys_entries() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let root = store.create_layer();
        let child = store.create_layer();
        store.add_child(root, child);
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Now remove the child.
        store.destroy_layer(child);
        let changes = store.evaluate();
        p.apply(&store, &changes);

        assert!(p.get_surface(child.index()).is_none());
        assert!(p.get_surface(root.index()).is_some());
    }

    #[test]
    fn apply_content_mapping_basic() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let layer = store.create_layer();
        let sid = SurfaceId(42);
        store.set_content(layer, Some(sid));
        let changes = store.evaluate();
        p.apply(&store, &changes);

        assert_eq!(
            p.surface_for_content(sid).map(|s| s.id()),
            p.get_surface(layer.index()).map(|s| s.id()),
        );
    }

    #[test]
    fn apply_content_mapping_reassign_to_new_slot() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let a = store.create_layer();
        let b = store.create_layer();
        let sid = SurfaceId(7);

        // Assign SurfaceId to slot A.
        store.set_content(a, Some(sid));
        let changes = store.evaluate();
        p.apply(&store, &changes);
        assert_eq!(
            p.surface_for_content(sid).map(|s| s.id()),
            p.get_surface(a.index()).map(|s| s.id()),
        );

        // Move SurfaceId to slot B.
        store.set_content(a, None);
        store.set_content(b, Some(sid));
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Forward mapping points to B.
        assert_eq!(
            p.surface_for_content(sid).map(|s| s.id()),
            p.get_surface(b.index()).map(|s| s.id()),
        );
    }

    #[test]
    fn apply_removal_of_old_slot_does_not_corrupt_reassigned_mapping() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let root = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();
        store.add_child(root, a);
        store.add_child(root, b);
        let sid = SurfaceId(10);

        // Assign SurfaceId to A.
        store.set_content(a, Some(sid));
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Reassign SurfaceId to B, then remove A entirely.
        store.set_content(a, None);
        store.set_content(b, Some(sid));
        store.destroy_layer(a);
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Mapping must still resolve to B.
        assert_eq!(
            p.surface_for_content(sid).map(|s| s.id()),
            p.get_surface(b.index()).map(|s| s.id()),
        );
    }

    #[test]
    fn apply_content_cleared_removes_mapping() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let layer = store.create_layer();
        let sid = SurfaceId(5);
        store.set_content(layer, Some(sid));
        let changes = store.evaluate();
        p.apply(&store, &changes);

        store.set_content(layer, None);
        let changes = store.evaluate();
        p.apply(&store, &changes);

        assert!(p.surface_for_content(sid).is_none());
    }

    // -----------------------------------------------------------------------
    // apply(): hidden state
    // -----------------------------------------------------------------------

    #[test]
    fn apply_hidden_at_creation_detaches_buffer() {
        // This test verifies the code path is reached without panicking.
        // Actual buffer detach verification would require a compositor.
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        use subduction_core::layer::LayerFlags;
        let layer = store.create_layer();
        store.set_flags(layer, LayerFlags { hidden: true });
        let changes = store.evaluate();

        // Should not panic — detach_buffer on an inert proxy is a no-op send.
        p.apply(&store, &changes);
        assert!(p.get_surface(layer.index()).is_some());
    }

    #[test]
    fn apply_hidden_transition_detaches_buffer() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        use subduction_core::layer::LayerFlags;
        let layer = store.create_layer();
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Transition to hidden.
        store.set_flags(layer, LayerFlags { hidden: true });
        let changes = store.evaluate();

        // Should not panic; the hidden slot list should include our layer.
        assert!(changes.hidden.contains(&layer.index()));
        p.apply(&store, &changes);
    }

    #[test]
    fn apply_unhidden_transition_is_handled() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        use subduction_core::layer::LayerFlags;
        let layer = store.create_layer();
        store.set_flags(layer, LayerFlags { hidden: true });
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Unhide.
        store.set_flags(layer, LayerFlags { hidden: false });
        let changes = store.evaluate();
        assert!(changes.unhidden.contains(&layer.index()));
        p.apply(&store, &changes);
    }

    // -----------------------------------------------------------------------
    // apply(): topology reorder
    // -----------------------------------------------------------------------

    #[test]
    fn apply_topology_reorder_exercises_place_above() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let root = store.create_layer();
        let a = store.create_layer();
        let b = store.create_layer();
        let c = store.create_layer();
        store.add_child(root, a);
        store.add_child(root, b);
        store.add_child(root, c);
        let changes = store.evaluate();
        p.apply(&store, &changes);

        // Reorder: move c before a.
        store.remove_from_parent(c);
        store.insert_before(c, a);
        let changes = store.evaluate();
        assert!(changes.topology_changed);
        p.apply(&store, &changes);
    }

    // -----------------------------------------------------------------------
    // destroy()
    // -----------------------------------------------------------------------

    #[test]
    fn destroy_clears_all_state() {
        let mut p = test_presenter();
        let mut store = LayerStore::new();
        let layer = store.create_layer();
        let sid = SurfaceId(1);
        store.set_content(layer, Some(sid));
        let changes = store.evaluate();
        p.apply(&store, &changes);

        p.destroy();
        assert!(p.get_surface(layer.index()).is_none());
        assert!(p.surface_for_content(sid).is_none());
    }

    // -----------------------------------------------------------------------
    // Constructor enforcement
    // -----------------------------------------------------------------------

    #[test]
    fn new_returns_error_when_no_surface() {
        let (conn, _eq, qh) = test_env();
        let compositor = wl_compositor::WlCompositor::from_id(&conn, ObjectId::null()).unwrap();
        let subcompositor =
            wl_subcompositor::WlSubcompositor::from_id(&conn, ObjectId::null()).unwrap();
        let ws = WaylandState::new(); // no surface set
        let result = WaylandPresenter::new(
            &ws,
            compositor,
            subcompositor,
            qh,
            WaylandPresenterConfig::default(),
        );
        assert_eq!(result.unwrap_err(), CreatePresenterError::NoSurface);
    }

    // -----------------------------------------------------------------------
    // PositionRounding
    // -----------------------------------------------------------------------

    #[test]
    fn rounding_floor() {
        let p = test_presenter_with_config(WaylandPresenterConfig {
            rounding: PositionRounding::Floor,
        });
        assert_eq!(p.round_pos(2.7), 2);
        assert_eq!(p.round_pos(2.3), 2);
        assert_eq!(p.round_pos(-1.2), -2);
    }

    #[test]
    fn rounding_round() {
        let p = test_presenter(); // default is Round
        assert_eq!(p.round_pos(2.7), 3);
        assert_eq!(p.round_pos(2.3), 2);
        assert_eq!(p.round_pos(2.5), 3);
    }

    #[test]
    fn rounding_ceil() {
        let p = test_presenter_with_config(WaylandPresenterConfig {
            rounding: PositionRounding::Ceil,
        });
        assert_eq!(p.round_pos(2.3), 3);
        assert_eq!(p.round_pos(2.0), 2);
        assert_eq!(p.round_pos(-1.2), -1);
    }

    #[test]
    fn default_config_uses_round() {
        let config = WaylandPresenterConfig::default();
        assert_eq!(config.rounding, PositionRounding::Round);
    }
}
