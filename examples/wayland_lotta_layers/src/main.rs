// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland stress-test: 100 animated subsurface layers in grouped orbits.
//!
//! Creates an xdg toplevel window with 10 group layers, each containing 10
//! child layers. Groups orbit the window center; children orbit their group.
//! Uses shared animation logic from `lotta_layers_common`.
//!
//! Note: Wayland subsurfaces are heavyweight compositor objects (unlike macOS
//! `CALayer`s), so 100 layers is a realistic stress-test for this backend.
//! The macOS counterpart runs 1000+ because `CALayer` compositing is
//! GPU-accelerated.
//!
//! Run with: `cargo run -p wayland_lotta_layers`

use lotta_layers_common::LAYER_SIZE;
use wayland_client::Connection;

use subduction_backend_wayland::{Presenter as _, WaylandPresenter, WaylandPresenterConfig};
use subduction_core::layer::LayerStore;
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::Duration;
use subduction_core::timing::PresentFeedback;

use wayland_example_common::{ExampleState, ShmBuffer, ShmPool};

const DEFAULT_W: u32 = 1024;
const DEFAULT_H: u32 = 768;
const NUM_GROUPS: usize = 10;
const LAYERS_PER_GROUP: usize = 10;

/// Returns an `[r, g, b]` triple in 0–255 for a given index using
/// golden-angle hue spacing.
fn layer_color(index: usize) -> [u8; 3] {
    let hue = (index as f64 * 137.508) % 360.0;
    let [r, g, b] = lotta_layers_common::hsl_to_rgb(hue, 0.7, 0.6);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "RGB channels are bounded 0.0..=1.0 and intentionally narrowed to u8"
    )]
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn effective_size(state: &ExampleState) -> (u32, u32) {
    let w = if state.xdg.width > 0 {
        state.xdg.width
    } else {
        DEFAULT_W
    };
    let h = if state.xdg.height > 0 {
        state.xdg.height
    } else {
        DEFAULT_H
    };
    (w, h)
}

fn main() {
    let connection = Connection::connect_to_env().expect("failed to connect to Wayland");
    let mut event_queue = connection.new_event_queue::<ExampleState>();
    let qh = event_queue.handle();

    let mut state = ExampleState::new();

    // Create registry and perform initial roundtrip for globals.
    let display = connection.display();
    let registry = display.get_registry(&qh, ());
    state.wayland.set_registry(registry);
    event_queue
        .roundtrip(&mut state)
        .expect("initial roundtrip failed");

    // Create xdg toplevel window.
    let window = wayland_example_common::create_window(
        &mut state,
        &qh,
        "Subduction · Wayland Lotta Layers (100)",
        DEFAULT_W as i32,
        DEFAULT_H as i32,
    );

    // Roundtrip to receive the initial configure.
    event_queue
        .roundtrip(&mut state)
        .expect("configure roundtrip failed");

    // --- Build the subduction layer tree ---
    let total_children = NUM_GROUPS * LAYERS_PER_GROUP;
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut group_ids = Vec::with_capacity(NUM_GROUPS);
    let mut child_ids = Vec::with_capacity(total_children);

    for _ in 0..NUM_GROUPS {
        let group = store.create_layer();
        store.add_child(root_id, group);
        group_ids.push(group);

        for _ in 0..LAYERS_PER_GROUP {
            let child = store.create_layer();
            store.add_child(group, child);
            child_ids.push(child);
        }
    }

    // Initial evaluate.
    let changes = store.evaluate();

    // Create the presenter.
    let compositor = state.wayland.compositor().expect("no compositor").clone();
    let subcompositor = state
        .wayland
        .subcompositor()
        .expect("no subcompositor")
        .clone();
    let mut presenter = WaylandPresenter::new(
        &state.wayland,
        compositor,
        subcompositor,
        qh.clone(),
        WaylandPresenterConfig::default(),
    )
    .expect("failed to create presenter");
    presenter.apply(&store, &changes);

    // --- Allocate solid-color buffers for each child layer ---
    let shm = state.xdg.shm.as_ref().expect("wl_shm not bound").clone();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "LAYER_SIZE is a small f64 constant from lotta_layers_common"
    )]
    let layer_px = LAYER_SIZE as u32;
    let layer_bytes = layer_px as usize * layer_px as usize * 4 * total_children;
    let mut layer_pool = ShmPool::new(&shm, layer_bytes, &qh);

    // Buffers must be kept alive so the wl_buffer proxies remain valid.
    #[allow(
        clippy::collection_is_never_read,
        reason = "keeps wl_buffer proxies alive"
    )]
    let mut layer_buffers: Vec<ShmBuffer> = Vec::with_capacity(total_children);
    for (i, &child_id) in child_ids.iter().enumerate() {
        let [r, g, b] = layer_color(i);
        let buf = layer_pool
            .alloc_solid(layer_px, layer_px, r, g, b, 230, &qh)
            .expect("pool exhausted");

        if let Some(surface) = presenter.get_surface(child_id.index()) {
            surface.attach(Some(&buf.buffer), 0, 0);
            surface.damage_buffer(0, 0, layer_px as i32, layer_px as i32);
            surface.commit();
        }
        layer_buffers.push(buf);
    }

    // --- Root surface background buffer (re-created on resize) ---
    let (mut _bg_pool, mut _bg_buf) = attach_background(&state, &shm, &window.surface, &qh);

    // --- Animation state ---
    let start_nanos = subduction_backend_wayland::now().ticks();
    let mut scheduler = Scheduler::new(SchedulerConfig::wayland());
    let (mut win_w, mut win_h) = effective_size(&state);

    // Initial commit to map the xdg toplevel and request the first frame.
    state
        .wayland
        .request_frame(&qh)
        .expect("request_frame failed");
    window.surface.commit();
    connection.flush().expect("flush failed");

    // --- Main event loop ---
    while state.running {
        event_queue
            .blocking_dispatch(&mut state)
            .expect("dispatch failed");

        // Handle resize: re-create the background buffer when the compositor
        // sends new dimensions.
        if state.xdg.needs_redraw {
            state.xdg.needs_redraw = false;
            let (new_w, new_h) = effective_size(&state);
            if new_w != win_w || new_h != win_h {
                win_w = new_w;
                win_h = new_h;
                (_bg_pool, _bg_buf) = attach_background(&state, &shm, &window.surface, &qh);
            }
        }

        while let Some(tick) = state.wayland.poll_tick() {
            let build_start = subduction_backend_wayland::now();

            let safety = Duration(scheduler.safety_margin_ticks());
            let hints = subduction_backend_wayland::compute_present_hints(&tick, safety);
            let plan = scheduler.plan(&tick, &hints);

            let elapsed_nanos = plan.semantic_time.ticks().saturating_sub(start_nanos);
            let t = elapsed_nanos as f64 / 1_000_000_000.0;

            lotta_layers_common::animate_groups(
                &mut store,
                &group_ids,
                &child_ids,
                NUM_GROUPS,
                LAYERS_PER_GROUP,
                f64::from(win_w) / 2.0,
                f64::from(win_h) / 2.0,
                t,
            );

            let changes = store.evaluate();
            presenter.apply(&store, &changes);

            let _id = state
                .wayland
                .commit_frame(&qh, &connection)
                .expect("commit_frame failed");

            let submitted_at = subduction_backend_wayland::now();
            let feedback =
                PresentFeedback::new(&hints, build_start, submitted_at, tick.prev_actual_present);
            scheduler.observe(&feedback);
        }
    }
}

/// Creates and attaches a dark background buffer to the root surface.
fn attach_background(
    state: &ExampleState,
    shm: &wayland_client::protocol::wl_shm::WlShm,
    surface: &wayland_client::protocol::wl_surface::WlSurface,
    qh: &wayland_client::QueueHandle<ExampleState>,
) -> (ShmPool, ShmBuffer) {
    let (w, h) = effective_size(state);
    let (pool, buf) = wayland_example_common::create_background(shm, w, h, qh);
    surface.attach(Some(&buf.buffer), 0, 0);
    surface.damage_buffer(0, 0, w as i32, h as i32);
    (pool, buf)
}
