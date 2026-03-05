// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland example: animated subsurface layers driven by
//! `subduction_backend_wayland`.
//!
//! Creates an xdg toplevel window with five colored subsurfaces that orbit
//! and translate, demonstrating the Wayland backend's building blocks:
//! frame callbacks for pacing, [`WaylandPresenter`] for subsurface
//! management, and [`Scheduler`] for frame planning.
//!
//! Run with: `cargo run -p wayland_layers`
//!
//! [`WaylandPresenter`]: subduction_backend_wayland::WaylandPresenter
//! [`Scheduler`]: subduction_core::scheduler::Scheduler

use wayland_client::Connection;

use subduction_backend_wayland::{Presenter as _, WaylandPresenter, WaylandPresenterConfig};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::Duration;
use subduction_core::timing::PresentFeedback;
use subduction_core::transform::Transform3d;

use wayland_example_common::{ExampleState, ShmBuffer, ShmPool};

const DEFAULT_W: u32 = 800;
const DEFAULT_H: u32 = 600;
const NUM_LAYERS: usize = 5;
const LAYER_SIZE: u32 = 80;

/// ARGB colors for the five layers.
const COLORS: [[u8; 4]; NUM_LAYERS] = [
    [242, 67, 54, 230],   // red
    [33, 150, 243, 230],  // blue
    [77, 176, 80, 230],   // green
    [255, 194, 8, 230],   // amber
    [156, 39, 176, 230],  // purple
];

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
        "Subduction · Wayland Layers",
        DEFAULT_W as i32,
        DEFAULT_H as i32,
    );

    // Roundtrip to receive the initial configure.
    event_queue
        .roundtrip(&mut state)
        .expect("configure roundtrip failed");

    // --- Build the subduction layer tree ---
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut layer_ids: Vec<LayerId> = Vec::with_capacity(NUM_LAYERS);
    for _ in 0..NUM_LAYERS {
        let id = store.create_layer();
        store.add_child(root_id, id);
        layer_ids.push(id);
    }

    // Initial evaluate to produce `added` entries for the presenter.
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

    // --- Allocate solid-color buffers for child layers ---
    let shm = state.xdg.shm.as_ref().expect("wl_shm not bound").clone();
    let layer_bytes = LAYER_SIZE as usize * LAYER_SIZE as usize * 4 * NUM_LAYERS;
    let mut layer_pool = ShmPool::new(&shm, layer_bytes, &qh);

    // Buffers must be kept alive so the wl_buffer proxies remain valid.
    #[allow(clippy::collection_is_never_read, reason = "keeps wl_buffer proxies alive")]
    let mut layer_buffers: Vec<ShmBuffer> = Vec::with_capacity(NUM_LAYERS);
    for (i, &layer_id) in layer_ids.iter().enumerate() {
        let [r, g, b, a] = COLORS[i];
        let buf = layer_pool
            .alloc_solid(LAYER_SIZE, LAYER_SIZE, r, g, b, a, &qh)
            .expect("pool exhausted");

        if let Some(surface) = presenter.get_surface(layer_id.index()) {
            surface.attach(Some(&buf.buffer), 0, 0);
            surface.damage_buffer(0, 0, LAYER_SIZE as i32, LAYER_SIZE as i32);
            surface.commit();
        }
        layer_buffers.push(buf);
    }

    // --- Root surface background buffer (re-created on resize) ---
    let (mut _bg_pool, mut _bg_buf) = attach_background(&state, &shm, &window.surface, &qh);

    // --- Animation state ---
    let start_nanos = subduction_backend_wayland::now().ticks();
    let mut scheduler = Scheduler::new(SchedulerConfig::wayland());
    let mut win_w = effective_width(&state);
    let mut win_h = effective_height(&state);

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
            let new_w = effective_width(&state);
            let new_h = effective_height(&state);
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

            animate_transforms(&mut store, &layer_ids, f64::from(win_w), f64::from(win_h), t);

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

fn effective_width(state: &ExampleState) -> u32 {
    if state.xdg.width > 0 {
        state.xdg.width
    } else {
        DEFAULT_W
    }
}

fn effective_height(state: &ExampleState) -> u32 {
    if state.xdg.height > 0 {
        state.xdg.height
    } else {
        DEFAULT_H
    }
}

/// Creates and attaches a dark background buffer to the root surface.
fn attach_background(
    state: &ExampleState,
    shm: &wayland_client::protocol::wl_shm::WlShm,
    surface: &wayland_client::protocol::wl_surface::WlSurface,
    qh: &wayland_client::QueueHandle<ExampleState>,
) -> (ShmPool, ShmBuffer) {
    let w = effective_width(state);
    let h = effective_height(state);
    let (pool, buf) = wayland_example_common::create_background(shm, w, h, qh);
    surface.attach(Some(&buf.buffer), 0, 0);
    surface.damage_buffer(0, 0, w as i32, h as i32);
    (pool, buf)
}

fn animate_transforms(
    store: &mut LayerStore,
    layer_ids: &[LayerId],
    win_w: f64,
    win_h: f64,
    t: f64,
) {
    let cx = win_w / 2.0;
    let cy = win_h / 2.0;
    let half = f64::from(LAYER_SIZE) / 2.0;

    for (i, &layer_id) in layer_ids.iter().enumerate() {
        let phase = i as f64 * core::f64::consts::TAU / NUM_LAYERS as f64;
        let radius = 150.0 + 50.0 * (t * 0.5 + phase).sin();
        let angle = t * (0.6 + i as f64 * 0.1) + phase;

        let x = cx + radius * angle.cos() - half;
        let y = cy + radius * angle.sin() - half;

        let transform = Transform3d::from_translation(x, y, 0.0);
        store.set_transform(layer_id, transform);
    }
}
