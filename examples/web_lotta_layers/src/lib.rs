// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Web stress-test: 1000+ animated DOM elements in grouped orbits.
//!
//! Creates a dark container with configurable layer count (via `?layers=N`
//! query parameter, default 1000). Layers are arranged as groups orbiting
//! the center, with children orbiting each group. An FPS counter is overlaid.
//!
//! Build with: `wasm-pack build --target web examples/web-lotta-layers`
//!
//! Then serve `examples/web-lotta-layers/` and open `index.html` in a browser.

// This crate only runs in the browser; suppress dead-code warnings when
// cargo-checking on a native host target.
#![no_std]
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "this crate only runs in the browser")
)]

extern crate alloc;

use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use lotta_layers_common::LAYER_SIZE;
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlElement};

use subduction_backend_web::DomPresenter;
use subduction_backend_web::LayerRoot;
use subduction_backend_web::Presenter as _;
use subduction_backend_web::RafLoop;
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::output::{Color, OutputId};
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::Duration;
use subduction_core::timing::{FrameTick, PresentFeedback};

const CONTAINER_W: f64 = 1024.0;
const CONTAINER_H: f64 = 768.0;
const DEFAULT_LAYERS: usize = 1000;

/// Returns a CSS `rgba()` string for a given index using golden-angle hue
/// spacing in HSL space.
fn layer_color_css(index: usize) -> String {
    let hue = (index as f64 * 137.508) % 360.0;
    let [r, g, b] = lotta_layers_common::hsl_to_rgb(hue, 0.7, 0.6);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "CSS color channels intentionally clamp to u8-like integer domain"
    )]
    let ri = (r * 255.0) as u32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "CSS color channels intentionally clamp to u8-like integer domain"
    )]
    let gi = (g * 255.0) as u32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "CSS color channels intentionally clamp to u8-like integer domain"
    )]
    let bi = (b * 255.0) as u32;
    format!("rgba({ri},{gi},{bi},0.9)")
}

struct AnimState {
    store: LayerStore,
    scheduler: Scheduler,
    presenter: DomPresenter,
    num_groups: usize,
    layers_per_group: usize,
    group_ids: Vec<LayerId>,
    child_ids: Vec<LayerId>,
    fps_element: HtmlElement,
    start_us: u64,
    timebase: subduction_core::time::Timebase,
    prev_time: f64,
}

/// Entry point — called automatically by `wasm_bindgen(start)`.
#[cfg_attr(all(target_arch = "wasm32", not(test)), wasm_bindgen(start))]
pub fn main() -> Result<(), JsValue> {
    let window = web_sys::window().expect("no global window");
    let document = window.document().expect("no document");

    // Parse ?layers=N from URL.
    let total_layers = parse_layer_count(&window);

    // Compute group/child split: at least 1 group.
    let num_groups = if total_layers >= 10 { 10 } else { 1 };
    let layers_per_group = total_layers / num_groups;

    let container = create_container(&document)?;
    document.body().expect("no body").append_child(&container)?;

    // --- Build layer tree ---
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut group_ids = Vec::with_capacity(num_groups);
    let mut child_ids = Vec::with_capacity(num_groups * layers_per_group);

    for _ in 0..num_groups {
        let group = store.create_layer();
        store.add_child(root_id, group);
        group_ids.push(group);

        for _ in 0..layers_per_group {
            let child = store.create_layer();
            store.add_child(group, child);
            child_ids.push(child);
        }
    }

    // Initial evaluate.
    let initial = store.evaluate();
    let backdrop_color = Color::from_rgba8(0x1a, 0x1a, 0x24, 0xff);
    let root = LayerRoot::new(container.clone()).with_backdrop_color(backdrop_color);
    let mut presenter = DomPresenter::new(root);
    presenter.apply(&store, &initial);

    // --- Style child layers ---
    for (i, &child_id) in child_ids.iter().enumerate() {
        let el = presenter
            .get_element(child_id.index())
            .expect("element just created");
        let s = el.style();
        s.set_property("width", &format!("{LAYER_SIZE}px"))?;
        s.set_property("height", &format!("{LAYER_SIZE}px"))?;
        s.set_property("background", &layer_color_css(i))?;
        s.set_property("border-radius", "2px")?;
    }

    // Group elements are invisible containers (no style needed).

    // --- FPS overlay ---
    let fps_element = create_fps_overlay(&document)?;
    container.append_child(&fps_element)?;

    let timebase = subduction_backend_web::timebase();
    let start_us = subduction_backend_web::now().ticks();

    let state = Rc::new(RefCell::new(AnimState {
        store,
        scheduler: Scheduler::new(SchedulerConfig::web()),
        presenter,
        num_groups,
        layers_per_group,
        group_ids,
        child_ids,
        fps_element,
        start_us,
        timebase,
        prev_time: 0.0,
    }));

    let state_cb = Rc::clone(&state);
    let raf = RafLoop::new(move |tick| on_tick(&state_cb, tick), OutputId(0));
    raf.start();

    // Keep the `RafLoop` alive.
    core::mem::forget(raf);

    Ok(())
}

fn parse_layer_count(window: &web_sys::Window) -> usize {
    window
        .location()
        .search()
        .ok()
        .and_then(|search| {
            // Parse "?layers=N" or "&layers=N".
            search.trim_start_matches('?').split('&').find_map(|pair| {
                let (key, value) = pair.split_once('=')?;
                if key == "layers" {
                    value.parse::<usize>().ok()
                } else {
                    None
                }
            })
        })
        .unwrap_or(DEFAULT_LAYERS)
}

fn create_container(doc: &Document) -> Result<HtmlElement, JsValue> {
    let el: HtmlElement = doc.create_element("div")?.unchecked_into();
    let s = el.style();
    s.set_property("width", &format!("{CONTAINER_W}px"))?;
    s.set_property("height", &format!("{CONTAINER_H}px"))?;
    s.set_property("position", "relative")?;
    s.set_property("overflow", "hidden")?;
    s.set_property("border-radius", "16px")?;
    s.set_property("box-shadow", "0 8px 32px rgba(0,0,0,0.5)")?;
    Ok(el)
}

fn create_fps_overlay(doc: &Document) -> Result<HtmlElement, JsValue> {
    let el: HtmlElement = doc.create_element("div")?.unchecked_into();
    let s = el.style();
    s.set_property("position", "absolute")?;
    s.set_property("top", "8px")?;
    s.set_property("left", "8px")?;
    s.set_property("padding", "4px 8px")?;
    s.set_property("background", "rgba(0,0,0,0.6)")?;
    s.set_property("color", "#fff")?;
    s.set_property("font-family", "monospace")?;
    s.set_property("font-size", "14px")?;
    s.set_property("border-radius", "4px")?;
    s.set_property("z-index", "1000")?;
    s.set_property("pointer-events", "none")?;
    el.set_text_content(Some("FPS: --"));
    Ok(el)
}

fn on_tick(state: &Rc<RefCell<AnimState>>, tick: FrameTick) {
    let mut s = state.borrow_mut();

    let build_start = subduction_backend_web::now();

    let safety = Duration(s.scheduler.safety_margin_ticks());
    let hints = subduction_backend_web::compute_present_hints(&tick, safety);
    let plan = s.scheduler.plan(&tick, &hints);

    let elapsed_us = plan.semantic_time.ticks().saturating_sub(s.start_us);
    let elapsed_nanos = s.timebase.ticks_to_nanos(elapsed_us);
    let t = elapsed_nanos as f64 / 1_000_000_000.0;

    // FPS calculation.
    let dt = t - s.prev_time;
    if dt > 0.0 {
        let fps = 1.0 / dt;
        s.fps_element
            .set_text_content(Some(&format!("FPS: {fps:.0}")));
    }
    s.prev_time = t;

    // Animate.
    let num_groups = s.num_groups;
    let layers_per_group = s.layers_per_group;
    let AnimState {
        ref mut store,
        ref mut presenter,
        ref group_ids,
        ref child_ids,
        ..
    } = *s;
    lotta_layers_common::animate_groups(
        store,
        group_ids,
        child_ids,
        num_groups,
        layers_per_group,
        CONTAINER_W / 2.0,
        CONTAINER_H / 2.0,
        t,
    );

    let changes = store.evaluate();
    presenter.apply(store, &changes);

    let submitted_at = subduction_backend_web::now();

    let feedback = PresentFeedback::new(&hints, build_start, submitted_at, None);
    s.scheduler.observe(&feedback);
}
