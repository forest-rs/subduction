// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `CALayer` tree presenter.
//!
//! Translates [`LayerStore`] state into a native `CALayer` hierarchy by
//! applying incremental updates from [`FrameChanges`].
//!
//! [`LayerStore`]: subduction_core::layer::LayerStore
//! [`FrameChanges`]: subduction_core::layer::FrameChanges

use alloc::vec::Vec;
use hashbrown::HashMap;

use objc2::rc::Retained;
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::NSArray;
use objc2_quartz_core::{CALayer, CATransaction, CATransform3D};
use subduction_core::backend::Presenter;
use subduction_core::layer::{ClipShape, FrameChanges, LayerStore};
use subduction_core::transform::Transform3d;

#[cfg(feature = "appkit")]
use objc2_app_kit::NSView;

/// Maps a [`LayerStore`] to a live `CALayer` tree, applying incremental
/// updates from [`FrameChanges`].
///
/// The presenter owns a root `CALayer` to which child layers are added and
/// removed. Call [`apply`](Self::apply) each frame with the latest
/// `FrameChanges` to synchronize the `CALayer` tree with the store.
#[derive(Debug)]
pub struct LayerPresenter {
    root_layer: Retained<CALayer>,
    layers: HashMap<u32, Retained<CALayer>>,
    #[cfg(feature = "appkit")]
    views: HashMap<u32, Retained<NSView>>,
}

impl LayerPresenter {
    /// Creates a new presenter that manages sublayers of `root_layer`.
    #[must_use]
    pub fn new(root_layer: Retained<CALayer>) -> Self {
        Self {
            root_layer,
            layers: HashMap::new(),
            #[cfg(feature = "appkit")]
            views: HashMap::new(),
        }
    }

    /// Returns a reference to the root `CALayer`.
    #[must_use]
    pub fn root_layer(&self) -> &CALayer {
        &self.root_layer
    }

    /// Returns the `CALayer` for the given slot index, if it exists.
    #[must_use]
    pub fn get_layer(&self, idx: u32) -> Option<&CALayer> {
        self.layers.get(&idx).map(|r| &**r)
    }

    /// Attaches an `NSView` to the given slot index.
    ///
    /// On each [`apply`](Self::apply), the presenter will sync the view's
    /// frame origin (centered on the layer's world-transform position) and
    /// alpha value from the store's effective opacity.
    ///
    /// The caller is responsible for creating the view and adding it as a
    /// subview of the appropriate parent. The presenter does **not** call
    /// `addSubview`.
    ///
    /// If a layer is removed (via [`FrameChanges::removed`]), any attached
    /// view is automatically removed from its superview and dropped.
    #[cfg(feature = "appkit")]
    pub fn attach_view(&mut self, idx: u32, view: Retained<NSView>) {
        self.views.insert(idx, view);
    }

    /// Detaches and returns the `NSView` for the given slot index, if any.
    ///
    /// The view is **not** removed from its superview — the caller manages
    /// that.
    #[cfg(feature = "appkit")]
    pub fn detach_view(&mut self, idx: u32) -> Option<Retained<NSView>> {
        self.views.remove(&idx)
    }

    /// Returns a reference to the attached `NSView` for the given slot index.
    #[cfg(feature = "appkit")]
    #[must_use]
    pub fn get_view(&self, idx: u32) -> Option<&NSView> {
        self.views.get(&idx).map(|r| &**r)
    }

    /// Reorders sublayers to match the store's traversal order.
    fn reorder_sublayers(&self, store: &LayerStore) {
        let order = store.traversal_order();
        let mut ordered: Vec<&CALayer> = Vec::with_capacity(order.len());
        for &idx in order {
            if let Some(layer) = self.layers.get(&idx) {
                ordered.push(layer);
            }
        }

        if ordered.is_empty() {
            return;
        }

        // Build an NSArray of the ordered layers and set as sublayers.
        let array = NSArray::from_slice(&ordered);
        // SAFETY: the layers in the array are valid CALayers owned by this
        // presenter and already sublayers of root_layer.
        unsafe { self.root_layer.setSublayers(Some(&array)) };
    }
}

impl Presenter for LayerPresenter {
    /// Applies incremental changes from a [`FrameChanges`] to the `CALayer`
    /// tree.
    ///
    /// Must be called on the main thread. Wraps all mutations in a
    /// `CATransaction` with implicit animations disabled.
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        CATransaction::begin();
        CATransaction::setDisableActions(true);

        // 1. Removals
        for &idx in &changes.removed {
            if let Some(layer) = self.layers.remove(&idx) {
                layer.removeFromSuperlayer();
            }
            #[cfg(feature = "appkit")]
            if let Some(view) = self.views.remove(&idx) {
                view.removeFromSuperview();
            }
        }

        // 2. Additions
        for &idx in &changes.added {
            let layer = CALayer::new();
            // Center anchor point (default) — position sets the center.
            layer.setAnchorPoint(CGPoint::new(0.5, 0.5));
            if store.effective_hidden_at(idx) {
                layer.setHidden(true);
            }
            self.root_layer.addSublayer(&layer);
            self.layers.insert(idx, layer);
        }

        // 3. Transforms
        for &idx in &changes.transforms {
            if let Some(layer) = self.layers.get(&idx) {
                let world = store.world_transform_at(idx);
                apply_transform(layer, world);
            }
        }

        // 4. Opacities
        for &idx in &changes.opacities {
            if let Some(layer) = self.layers.get(&idx) {
                let opacity = store.effective_opacity_at(idx);
                layer.setOpacity(opacity);
            }
        }

        // 5. Hidden/unhidden
        for &idx in &changes.hidden {
            if let Some(layer) = self.layers.get(&idx) {
                layer.setHidden(true);
            }
            #[cfg(feature = "appkit")]
            if let Some(view) = self.views.get(&idx) {
                view.setHidden(true);
            }
        }
        for &idx in &changes.unhidden {
            if let Some(layer) = self.layers.get(&idx) {
                layer.setHidden(false);
            }
            #[cfg(feature = "appkit")]
            if let Some(view) = self.views.get(&idx) {
                view.setHidden(false);
            }
        }

        // 6. Bounds
        for &idx in &changes.bounds {
            if let Some(layer) = self.layers.get(&idx) {
                let bounds = store.bounds_at(idx);
                layer.setBounds(CGRect::new(
                    CGPoint::new(0.0, 0.0),
                    CGSize::new(bounds.width, bounds.height),
                ));
            }
        }

        // 7. Clips
        for &idx in &changes.clips {
            if let Some(layer) = self.layers.get(&idx) {
                let clip = store.clip_at(idx);
                apply_clip(layer, clip);
            }
        }

        // 8. Topology reorder
        if changes.topology_changed {
            self.reorder_sublayers(store);
        }

        // 9. Sync attached NSViews (position + opacity).
        #[cfg(feature = "appkit")]
        for (&idx, view) in &self.views {
            let world = store.world_transform_at(idx);
            let tx = world.cols[3][0];
            let ty = world.cols[3][1];
            let size = view.frame().size;
            view.setFrameOrigin(CGPoint::new(tx - size.width / 2.0, ty - size.height / 2.0));
            view.setAlphaValue(f64::from(store.effective_opacity_at(idx)));
        }

        CATransaction::commit();
    }
}

/// Applies a world transform to a `CALayer` by splitting it into position
/// (translation) and rotation+scale (the rest of the matrix).
fn apply_transform(layer: &CALayer, world: Transform3d) {
    // Extract position from column 3.
    let col3 = world.col(3);
    layer.setPosition(CGPoint::new(col3[0], col3[1]));

    // Build a CATransform3D without the translation component.
    let mut m = world;
    m.cols[3][0] = 0.0;
    m.cols[3][1] = 0.0;
    m.cols[3][2] = 0.0;
    layer.setTransform(transform3d_to_ca(&m));
}

/// Applies a clip shape (or clears clipping) on a `CALayer`.
fn apply_clip(layer: &CALayer, clip: Option<ClipShape>) {
    match clip {
        None => {
            layer.setMasksToBounds(false);
        }
        Some(ClipShape::Rect(rect)) => {
            layer.setMasksToBounds(true);
            layer.setBounds(CGRect::new(
                CGPoint::new(rect.x0, rect.y0),
                CGSize::new(rect.width(), rect.height()),
            ));
            layer.setCornerRadius(0.0);
        }
        Some(ClipShape::RoundedRect(rrect)) => {
            layer.setMasksToBounds(true);
            let rect = rrect.rect();
            layer.setBounds(CGRect::new(
                CGPoint::new(rect.x0, rect.y0),
                CGSize::new(rect.width(), rect.height()),
            ));
            // CALayer only supports uniform corner radius.
            // Use the maximum radius from the rounded rect's radii.
            let radii = rrect.radii();
            let max_r = radii
                .as_single_radius()
                .unwrap_or_else(|| radii_max(&radii));
            layer.setCornerRadius(max_r);
        }
    }
}

/// Returns the maximum corner radius from a `kurbo::RoundedRectRadii`.
fn radii_max(radii: &kurbo::RoundedRectRadii) -> f64 {
    radii
        .top_left
        .max(radii.top_right)
        .max(radii.bottom_left)
        .max(radii.bottom_right)
}

/// Converts a [`Transform3d`] to a `CATransform3D`.
fn transform3d_to_ca(m: &Transform3d) -> CATransform3D {
    let c0 = m.col(0);
    let c1 = m.col(1);
    let c2 = m.col(2);
    let c3 = m.col(3);
    CATransform3D {
        m11: c0[0],
        m12: c0[1],
        m13: c0[2],
        m14: c0[3],
        m21: c1[0],
        m22: c1[1],
        m23: c1[2],
        m24: c1[3],
        m31: c2[0],
        m32: c2[1],
        m33: c2[2],
        m34: c2[3],
        m41: c3[0],
        m42: c3[1],
        m43: c3[2],
        m44: c3[3],
    }
}
