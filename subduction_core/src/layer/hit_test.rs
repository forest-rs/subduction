// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer-level hit testing.
//!
//! Provides point-in-layer queries against the compositing tree, walking layers
//! front-to-back and respecting transforms, bounds, clips, and hidden state.
//! Frameworks use the returned [`HitEntry`] results for coarse hit detection,
//! then perform their own widget-level testing within each hit layer.

use alloc::vec::Vec;

use kurbo::{Point, Rect};

use super::id::{INVALID, LayerId};
use super::store::LayerStore;

/// A layer intersected by a point query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitEntry {
    /// The layer that was hit.
    pub layer: LayerId,
    /// The query point in this layer's local coordinate space.
    ///
    /// Frameworks use this directly for widget-level hit testing within
    /// the layer's content surface.
    pub local_point: Point,
}

impl LayerStore {
    /// Returns all hittable layers under `point` (in root/world coordinates),
    /// ordered front-to-back.
    ///
    /// A layer is hittable when:
    /// - it is not effectively hidden,
    /// - it has content (`content` is `Some`),
    /// - its world transform is invertible,
    /// - the point (in local space) falls within the layer's hit rect
    ///   (or bounds, if no hit rect is set),
    /// - the point is not excluded by the layer's own clip shape, and
    /// - the point is not excluded by any ancestor's clip shape.
    ///
    /// Only valid after [`evaluate`](Self::evaluate) has been called.
    #[must_use]
    pub fn hit_test(&self, point: Point) -> Vec<HitEntry> {
        let mut results = Vec::new();
        self.hit_test_into(point, &mut results);
        results
    }

    /// Like [`hit_test`](Self::hit_test), but reuses a caller-provided buffer
    /// to avoid allocation on repeated queries.
    pub fn hit_test_into(&self, point: Point, results: &mut Vec<HitEntry>) {
        results.clear();

        for &idx in self.traversal_order.iter().rev() {
            let i = idx as usize;

            if self.effective_hidden[i] {
                continue;
            }

            if self.content[i].is_none() {
                continue;
            }

            // Hit area: explicit hit_rect if set, otherwise full bounds.
            let hit_area = match self.hit_rect[i] {
                Some(r) => r,
                None => {
                    let b = self.bounds[i];
                    Rect::new(0.0, 0.0, b.width, b.height)
                }
            };
            if hit_area.width() <= 0.0 || hit_area.height() <= 0.0 {
                continue;
            }

            let world_inv = match self.world_transform[i].inverse() {
                Some(inv) => inv,
                None => continue,
            };

            let local_point = match world_inv.transform_point(point) {
                Some(p) => p,
                None => continue,
            };

            if !hit_area.contains(local_point) {
                continue;
            }

            if let Some(clip) = &self.clip[i]
                && !clip.contains(local_point)
            {
                continue;
            }

            if !self.passes_ancestor_clips(idx, point) {
                continue;
            }

            results.push(HitEntry {
                layer: LayerId {
                    idx,
                    generation: self.generation[i],
                },
                local_point,
            });
        }
    }

    /// Walks the parent chain of `idx`, testing the screen-space `point`
    /// against each ancestor's clip shape in that ancestor's local space.
    fn passes_ancestor_clips(&self, idx: u32, screen_point: Point) -> bool {
        let mut ancestor = self.parent[idx as usize];
        while ancestor != INVALID {
            let ai = ancestor as usize;
            if let Some(clip) = &self.clip[ai] {
                let inv = match self.world_transform[ai].inverse() {
                    Some(inv) => inv,
                    None => return false,
                };
                let local = match inv.transform_point(screen_point) {
                    Some(p) => p,
                    None => return false,
                };
                if !clip.contains(local) {
                    return false;
                }
            }
            ancestor = self.parent[ai];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{ClipShape, LayerFlags, SurfaceId};
    use crate::transform::Transform3d;
    use kurbo::{RoundedRect, Size};

    /// Helper: create a store with a single content layer at the origin.
    fn single_layer_store(w: f64, h: f64) -> (LayerStore, LayerId) {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(w, h));
        store.set_content(id, Some(SurfaceId(1)));
        store.evaluate();
        (store, id)
    }

    #[test]
    fn hit_inside_single_layer() {
        let (store, id) = single_layer_store(100.0, 80.0);
        let hits = store.hit_test(Point::new(50.0, 40.0));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].layer, id);
        assert_eq!(hits[0].local_point, Point::new(50.0, 40.0));
    }

    #[test]
    fn miss_outside_single_layer() {
        let (store, _) = single_layer_store(100.0, 80.0);
        assert!(store.hit_test(Point::new(150.0, 40.0)).is_empty());
        assert!(store.hit_test(Point::new(50.0, 90.0)).is_empty());
        assert!(store.hit_test(Point::new(-1.0, 40.0)).is_empty());
    }

    #[test]
    fn front_to_back_ordering() {
        let mut store = LayerStore::new();
        let back = store.create_layer();
        store.set_bounds(back, Size::new(200.0, 200.0));
        store.set_content(back, Some(SurfaceId(1)));

        let front = store.create_layer();
        store.set_bounds(front, Size::new(200.0, 200.0));
        store.set_content(front, Some(SurfaceId(2)));

        store.evaluate();

        let hits = store.hit_test(Point::new(100.0, 100.0));
        assert_eq!(hits.len(), 2);
        // Front (later in traversal) comes first in results.
        assert_eq!(hits[0].layer, front);
        assert_eq!(hits[1].layer, back);
    }

    #[test]
    fn hidden_layer_excluded() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 100.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_flags(id, LayerFlags { hidden: true });
        store.evaluate();

        assert!(store.hit_test(Point::new(50.0, 50.0)).is_empty());
    }

    #[test]
    fn contentless_layer_excluded() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 100.0));
        // No set_content — remains None.
        store.evaluate();

        assert!(store.hit_test(Point::new(50.0, 50.0)).is_empty());
    }

    #[test]
    fn parent_clip_excludes_child() {
        let mut store = LayerStore::new();

        // Parent with a clip region.
        let parent = store.create_layer();
        store.set_bounds(parent, Size::new(200.0, 200.0));
        store.set_clip(
            parent,
            Some(ClipShape::Rect(Rect::new(0.0, 0.0, 100.0, 100.0))),
        );

        // Child extends beyond the parent's clip.
        let child = store.create_layer();
        store.reparent(child, parent);
        store.set_bounds(child, Size::new(200.0, 200.0));
        store.set_content(child, Some(SurfaceId(1)));

        store.evaluate();

        // Inside parent clip — hit.
        assert_eq!(store.hit_test(Point::new(50.0, 50.0)).len(), 1);
        // Outside parent clip but inside child bounds — miss.
        assert!(store.hit_test(Point::new(150.0, 50.0)).is_empty());
    }

    #[test]
    fn grandparent_clip_chain() {
        let mut store = LayerStore::new();

        let grandparent = store.create_layer();
        store.set_bounds(grandparent, Size::new(300.0, 300.0));
        store.set_clip(
            grandparent,
            Some(ClipShape::Rect(Rect::new(0.0, 0.0, 80.0, 80.0))),
        );

        let parent = store.create_layer();
        store.reparent(parent, grandparent);
        store.set_bounds(parent, Size::new(300.0, 300.0));

        let child = store.create_layer();
        store.reparent(child, parent);
        store.set_bounds(child, Size::new(300.0, 300.0));
        store.set_content(child, Some(SurfaceId(1)));

        store.evaluate();

        // Inside grandparent clip.
        assert_eq!(store.hit_test(Point::new(40.0, 40.0)).len(), 1);
        // Outside grandparent clip.
        assert!(store.hit_test(Point::new(100.0, 50.0)).is_empty());
    }

    #[test]
    fn translated_layer() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 100.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_transform(id, Transform3d::from_translation(50.0, 50.0, 0.0));
        store.evaluate();

        // World point (70, 70) → local (20, 20).
        let hits = store.hit_test(Point::new(70.0, 70.0));
        assert_eq!(hits.len(), 1);
        let eps = 1e-10;
        assert!((hits[0].local_point.x - 20.0).abs() < eps);
        assert!((hits[0].local_point.y - 20.0).abs() < eps);

        // World point (30, 30) → local (-20, -20) — outside bounds.
        assert!(store.hit_test(Point::new(30.0, 30.0)).is_empty());
    }

    #[test]
    fn scaled_layer() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(50.0, 50.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_transform(id, Transform3d::from_scale(2.0, 2.0, 1.0));
        store.evaluate();

        // World (80, 80) → local (40, 40) — inside 50×50 bounds.
        let hits = store.hit_test(Point::new(80.0, 80.0));
        assert_eq!(hits.len(), 1);
        let eps = 1e-10;
        assert!((hits[0].local_point.x - 40.0).abs() < eps);
        assert!((hits[0].local_point.y - 40.0).abs() < eps);

        // World (110, 10) → local (55, 5) — outside bounds width.
        assert!(store.hit_test(Point::new(110.0, 10.0)).is_empty());
    }

    #[test]
    fn rotated_layer() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 100.0));
        store.set_content(id, Some(SurfaceId(1)));
        // 90° rotation: local +X → world +Y, local +Y → world −X.
        store.set_transform(
            id,
            Transform3d::from_rotation_z(core::f64::consts::FRAC_PI_2),
        );
        store.evaluate();

        // World point (-50, 50) → local (50, 50) after inverse rotation.
        let hits = store.hit_test(Point::new(-50.0, 50.0));
        assert_eq!(hits.len(), 1);
        let eps = 1e-6;
        assert!((hits[0].local_point.x - 50.0).abs() < eps);
        assert!((hits[0].local_point.y - 50.0).abs() < eps);
    }

    #[test]
    fn non_invertible_transform_excluded() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 100.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_transform(id, Transform3d::from_scale(0.0, 1.0, 1.0));
        store.evaluate();

        assert!(store.hit_test(Point::new(50.0, 50.0)).is_empty());
    }

    #[test]
    fn zero_bounds_excluded() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(0.0, 100.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.evaluate();

        assert!(store.hit_test(Point::new(0.0, 50.0)).is_empty());
    }

    #[test]
    fn own_clip_constrains_hittability() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(200.0, 200.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_clip(id, Some(ClipShape::Rect(Rect::new(10.0, 10.0, 90.0, 90.0))));
        store.evaluate();

        // Inside both bounds and clip.
        assert_eq!(store.hit_test(Point::new(50.0, 50.0)).len(), 1);
        // Inside bounds but outside clip.
        assert!(store.hit_test(Point::new(5.0, 50.0)).is_empty());
        assert!(store.hit_test(Point::new(150.0, 50.0)).is_empty());
    }

    #[test]
    fn rounded_clip_on_parent() {
        let mut store = LayerStore::new();

        let parent = store.create_layer();
        store.set_bounds(parent, Size::new(100.0, 100.0));
        store.set_clip(
            parent,
            Some(ClipShape::RoundedRect(RoundedRect::from_rect(
                Rect::new(0.0, 0.0, 100.0, 100.0),
                20.0,
            ))),
        );

        let child = store.create_layer();
        store.reparent(child, parent);
        store.set_bounds(child, Size::new(100.0, 100.0));
        store.set_content(child, Some(SurfaceId(1)));

        store.evaluate();

        // Center — clearly inside.
        assert_eq!(store.hit_test(Point::new(50.0, 50.0)).len(), 1);
        // Corner (2, 2) — outside the rounded corner arc.
        assert!(store.hit_test(Point::new(2.0, 2.0)).is_empty());
    }

    #[test]
    fn empty_store_returns_empty() {
        let store = LayerStore::new();
        assert!(store.hit_test(Point::new(0.0, 0.0)).is_empty());
    }

    #[test]
    fn hit_test_into_reuses_buffer() {
        let (store, _) = single_layer_store(100.0, 80.0);
        let mut buf = Vec::new();

        store.hit_test_into(Point::new(50.0, 40.0), &mut buf);
        assert_eq!(buf.len(), 1);

        // Second call clears and refills.
        store.hit_test_into(Point::new(200.0, 200.0), &mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn child_inherits_parent_transform() {
        let mut store = LayerStore::new();

        let parent = store.create_layer();
        store.set_bounds(parent, Size::new(200.0, 200.0));
        store.set_transform(parent, Transform3d::from_translation(100.0, 100.0, 0.0));

        let child = store.create_layer();
        store.reparent(child, parent);
        store.set_bounds(child, Size::new(50.0, 50.0));
        store.set_content(child, Some(SurfaceId(1)));

        store.evaluate();

        // Child's world origin is at (100, 100). Point (120, 120) → local (20, 20).
        let hits = store.hit_test(Point::new(120.0, 120.0));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].layer, child);
        let eps = 1e-10;
        assert!((hits[0].local_point.x - 20.0).abs() < eps);
        assert!((hits[0].local_point.y - 20.0).abs() < eps);

        // Point (50, 50) is before parent origin — no hit.
        assert!(store.hit_test(Point::new(50.0, 50.0)).is_empty());
    }

    // ── hit_rect tests ───────────────────────────────────────────────

    #[test]
    fn hit_rect_restricts_hit_area() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        // Surface is 200×200, but only the inset region is interactive.
        store.set_bounds(id, Size::new(200.0, 200.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_hit_rect(id, Some(Rect::new(30.0, 30.0, 170.0, 170.0)));
        store.evaluate();

        // Inside hit_rect — hit.
        assert_eq!(store.hit_test(Point::new(100.0, 100.0)).len(), 1);
        // Inside bounds but outside hit_rect (shadow area) — miss.
        assert!(store.hit_test(Point::new(10.0, 10.0)).is_empty());
        assert!(store.hit_test(Point::new(180.0, 180.0)).is_empty());
    }

    #[test]
    fn hit_rect_none_falls_back_to_bounds() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 80.0));
        store.set_content(id, Some(SurfaceId(1)));
        // No hit_rect set — should use full bounds.
        store.evaluate();

        assert_eq!(store.hit_test(Point::new(50.0, 40.0)).len(), 1);
        assert!(store.hit_test(Point::new(110.0, 40.0)).is_empty());
    }

    #[test]
    fn hit_rect_with_transform() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(220.0, 180.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_transform(id, Transform3d::from_translation(100.0, 100.0, 0.0));
        // Inset hit_rect: only (30,25)-(190,155) in local space.
        store.set_hit_rect(id, Some(Rect::new(30.0, 25.0, 190.0, 155.0)));
        store.evaluate();

        // Screen (140, 130) → local (40, 30) — inside hit_rect.
        let hits = store.hit_test(Point::new(140.0, 130.0));
        assert_eq!(hits.len(), 1);

        // Screen (110, 110) → local (10, 10) — inside bounds, outside hit_rect.
        assert!(store.hit_test(Point::new(110.0, 110.0)).is_empty());
    }

    #[test]
    fn hit_rect_cleared_restores_bounds() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(100.0, 100.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_hit_rect(id, Some(Rect::new(40.0, 40.0, 60.0, 60.0)));
        store.evaluate();

        // (10, 10) is outside hit_rect.
        assert!(store.hit_test(Point::new(10.0, 10.0)).is_empty());

        // Clear hit_rect — falls back to full bounds.
        store.set_hit_rect(id, None);
        assert_eq!(store.hit_test(Point::new(10.0, 10.0)).len(), 1);
    }

    #[test]
    fn hit_rect_local_point_is_in_layer_space() {
        let mut store = LayerStore::new();
        let id = store.create_layer();
        store.set_bounds(id, Size::new(200.0, 200.0));
        store.set_content(id, Some(SurfaceId(1)));
        store.set_transform(id, Transform3d::from_translation(50.0, 50.0, 0.0));
        store.set_hit_rect(id, Some(Rect::new(20.0, 20.0, 180.0, 180.0)));
        store.evaluate();

        // Screen (100, 100) → local (50, 50).
        let hits = store.hit_test(Point::new(100.0, 100.0));
        assert_eq!(hits.len(), 1);
        let eps = 1e-10;
        assert!((hits[0].local_point.x - 50.0).abs() < eps);
        assert!((hits[0].local_point.y - 50.0).abs() < eps);
    }
}
