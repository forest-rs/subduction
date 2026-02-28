// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! DOM element management.
//!
//! Translates [`LayerStore`] state into a set of positioned `<div>` elements by
//! applying incremental updates from [`FrameChanges`].
//!
//! [`LayerStore`]: subduction_core::layer::LayerStore
//! [`FrameChanges`]: subduction_core::layer::FrameChanges

use alloc::format;
use alloc::vec::Vec;

use subduction_core::backend::Presenter;
use subduction_core::layer::{ClipShape, FrameChanges, LayerStore};
use subduction_core::transform::Transform3d;
use wasm_bindgen::JsCast as _;
use web_sys::HtmlElement;

/// Maps a [`LayerStore`] to live DOM elements, applying incremental updates
/// from [`FrameChanges`].
///
/// The presenter owns a container `HtmlElement` to which child `<div>` elements
/// are added and removed. Call [`apply`](Self::apply) each frame with the
/// latest `FrameChanges` to synchronize the DOM with the store.
pub struct DomPresenter {
    container: HtmlElement,
    elements: Vec<Option<HtmlElement>>,
}

impl core::fmt::Debug for DomPresenter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DomPresenter")
            .field("container", &"HtmlElement")
            .field("elements_len", &self.elements.len())
            .finish()
    }
}

impl DomPresenter {
    /// Creates a new presenter that manages child elements of `container`.
    #[must_use]
    pub fn new(container: HtmlElement) -> Self {
        Self {
            container,
            elements: Vec::new(),
        }
    }

    /// Returns a reference to the container element.
    #[must_use]
    pub fn container(&self) -> &HtmlElement {
        &self.container
    }

    /// Returns the DOM element for the given slot index, if it exists.
    #[must_use]
    pub fn get_element(&self, idx: u32) -> Option<&HtmlElement> {
        self.elements
            .get(idx as usize)
            .and_then(|slot| slot.as_ref())
    }

    /// Takes an element out of the slot, leaving `None`.
    fn take_element(&mut self, idx: u32) -> Option<HtmlElement> {
        self.elements.get_mut(idx as usize)?.take()
    }

    /// Stores an element at the given slot index, growing the vec if needed.
    fn put_element(&mut self, idx: u32, el: HtmlElement) {
        let slot = idx as usize;
        if self.elements.len() <= slot {
            self.elements.resize_with(slot + 1, || None);
        }
        self.elements[slot] = Some(el);
    }
}

impl Presenter for DomPresenter {
    /// Applies incremental changes from a [`FrameChanges`] to the DOM.
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        // 1. Removals
        for &idx in &changes.removed {
            if let Some(el) = self.take_element(idx) {
                el.remove();
            }
        }

        // 2. Additions
        for &idx in &changes.added {
            let doc = self.container.owner_document().expect("no owner document");
            let el: HtmlElement = doc
                .create_element("div")
                .expect("create_element failed")
                .unchecked_into();
            let s = el.style();
            let _ = s.set_property("position", "absolute");
            let _ = s.set_property("left", "0");
            let _ = s.set_property("top", "0");
            let _ = s.set_property("transform-origin", "0 0");
            if store.effective_hidden_at(idx) {
                let _ = s.set_property("display", "none");
            }
            let _ = self.container.append_child(&el);
            self.put_element(idx, el);
        }

        // 3. Transforms
        for &idx in &changes.transforms {
            if let Some(el) = self.get_element(idx) {
                let world = store.world_transform_at(idx);
                apply_css_transform(el, &world);
            }
        }

        // 4. Opacities
        for &idx in &changes.opacities {
            if let Some(el) = self.get_element(idx) {
                let opacity = store.effective_opacity_at(idx);
                let _ = el.style().set_property("opacity", &format!("{opacity}"));
            }
        }

        // 5. Hidden/unhidden
        for &idx in &changes.hidden {
            if let Some(el) = self.get_element(idx) {
                let _ = el.style().set_property("display", "none");
            }
        }
        for &idx in &changes.unhidden {
            if let Some(el) = self.get_element(idx) {
                let _ = el.style().remove_property("display");
            }
        }

        // 6. Bounds
        for &idx in &changes.bounds {
            if let Some(el) = self.get_element(idx) {
                let bounds = store.bounds_at(idx);
                let s = el.style();
                if bounds.width > 0.0 && bounds.height > 0.0 {
                    let _ = s.set_property("width", &format!("{}px", bounds.width));
                    let _ = s.set_property("height", &format!("{}px", bounds.height));
                } else {
                    let _ = s.remove_property("width");
                    let _ = s.remove_property("height");
                }
            }
        }

        // 7. Clips
        for &idx in &changes.clips {
            if let Some(el) = self.get_element(idx) {
                let clip = store.clip_at(idx);
                apply_css_clip(el, clip);
            }
        }

        // 8. Topology reorder
        if changes.topology_changed {
            for &idx in store.traversal_order() {
                if let Some(el) = self.get_element(idx) {
                    // DOM re-append moves an existing child, reordering it.
                    let _ = self.container.append_child(el);
                }
            }
        }
    }
}

/// Applies a world transform as a CSS `matrix3d()` value.
fn apply_css_transform(el: &HtmlElement, xf: &Transform3d) {
    let c0 = xf.col(0);
    let c1 = xf.col(1);
    let c2 = xf.col(2);
    let c3 = xf.col(3);

    let css = format!(
        "matrix3d({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})",
        c0[0],
        c0[1],
        c0[2],
        c0[3],
        c1[0],
        c1[1],
        c1[2],
        c1[3],
        c2[0],
        c2[1],
        c2[2],
        c2[3],
        c3[0],
        c3[1],
        c3[2],
        c3[3],
    );

    let _ = el.style().set_property("transform", &css);
}

/// Applies a clip shape (or clears clipping) as CSS properties.
fn apply_css_clip(el: &HtmlElement, clip: Option<ClipShape>) {
    let s = el.style();
    match clip {
        None => {
            let _ = s.set_property("overflow", "visible");
            let _ = s.remove_property("width");
            let _ = s.remove_property("height");
            let _ = s.remove_property("border-radius");
        }
        Some(ClipShape::Rect(rect)) => {
            let _ = s.set_property("overflow", "hidden");
            let _ = s.set_property("width", &format!("{}px", rect.width()));
            let _ = s.set_property("height", &format!("{}px", rect.height()));
            let _ = s.set_property("border-radius", "0");
        }
        Some(ClipShape::RoundedRect(rrect)) => {
            let rect = rrect.rect();
            let _ = s.set_property("overflow", "hidden");
            let _ = s.set_property("width", &format!("{}px", rect.width()));
            let _ = s.set_property("height", &format!("{}px", rect.height()));
            let radii = rrect.radii();
            let _ = s.set_property(
                "border-radius",
                &format!(
                    "{}px {}px {}px {}px",
                    radii.top_left, radii.top_right, radii.bottom_right, radii.bottom_left,
                ),
            );
        }
    }
}
