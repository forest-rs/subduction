// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Windows example: animated DirectComposition layers driven by
//! `subduction_backend_windows`.
//!
//! Creates a window with five layers that orbit and pulse opacity,
//! demonstrating that subduction's layer tree can drive DirectComposition
//! visuals with the standard Plan → Animate → Evaluate → Render frame
//! loop.
//!
//! Run with: `cargo run -p windows_layers`

#![expect(unsafe_code, reason = "Win32 windowing example requires unsafe code")]

use std::cell::UnsafeCell;

use subduction_backend_windows::{
    self as backend, compute_hints, make_tick, DCompPresenter, Presenter as _, TickSource,
    WM_APP_TICK,
};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::HostTime;
use subduction_core::timing::PendingFeedback;
use subduction_core::trace::{
    FramePlanEvent, FrameSummaryBuilder, FrameTickEvent, PhaseBeginEvent, PhaseEndEvent, PhaseKind,
    PresentFeedbackEvent, SubmitEvent, TraceSink as _,
};
use subduction_core::transform::Transform3d;

use kurbo::Size;
use subduction_debug::recorder::RecorderSink;

use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::DirectComposition::IDCompositionDevice;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows_core::Interface;

const WINDOW_W: f64 = 800.0;
const WINDOW_H: f64 = 600.0;
const NUM_LAYERS: usize = 5;
const LAYER_SIZE: f64 = 80.0;

/// 60 Hz refresh interval in nanoseconds.
const REFRESH_NS: u64 = 16_666_667;

/// Safety margin for present hints (2 ms).
const SAFETY_MARGIN_NS: u64 = 2_000_000;

/// RGBA colors for each layer.
const COLORS: [[f32; 4]; NUM_LAYERS] = [
    [0.95, 0.26, 0.21, 0.9], // red
    [0.13, 0.59, 0.95, 0.9], // blue
    [0.30, 0.69, 0.31, 0.9], // green
    [1.00, 0.76, 0.03, 0.9], // amber
    [0.61, 0.15, 0.69, 0.9], // purple
];

/// All mutable state for the frame loop.
struct AnimState {
    store: LayerStore,
    presenter: DCompPresenter,
    scheduler: Scheduler,
    sub_ids: Vec<LayerId>,
    start_ticks: u64,
    timebase: subduction_core::time::Timebase,
    frame_index: u64,
    prev_present_time: Option<HostTime>,
    pending_feedback: Option<PendingFeedback>,
    recorder: RecorderSink,
    _tick_source: TickSource,
}

/// Single-threaded global state. Only accessed from the message pump thread.
struct GlobalState(UnsafeCell<Option<AnimState>>);

// SAFETY: Only accessed from the single message-pump thread.
unsafe impl Sync for GlobalState {}

static STATE: GlobalState = GlobalState(UnsafeCell::new(None));

fn state_mut() -> Option<&'static mut AnimState> {
    // SAFETY: Only called from the main (message pump) thread.
    // No concurrent access is possible because Win32 message dispatch
    // is single-threaded.
    unsafe { (*STATE.0.get()).as_mut() }
}

fn state_ref() -> Option<&'static AnimState> {
    // SAFETY: Same single-threaded access guarantee.
    unsafe { (*STATE.0.get()).as_ref() }
}

fn main() {
    // SAFETY: Win32 window creation and message loop.
    unsafe { run().expect("failed to run example") };
}

/// # Safety
///
/// Must be called from the main thread. Creates Win32 resources via FFI.
unsafe fn run() -> windows::core::Result<()> {
    let instance = unsafe { GetModuleHandleW(None)? };

    let class_name = windows::core::w!("SubductionLayers");
    let wc = WNDCLASSEXW {
        cbSize: size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wnd_proc),
        hInstance: instance.into(),
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW)? },
        lpszClassName: class_name,
        ..Default::default()
    };
    unsafe { RegisterClassExW(&wc) };

    let hwnd = unsafe {
        CreateWindowExW(
            WS_EX_NOREDIRECTIONBITMAP,
            class_name,
            windows::core::w!("Subduction \u{00b7} Animated Layers (Windows)"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            WINDOW_W as i32,
            WINDOW_H as i32,
            None,
            None,
            Some(instance.into()),
            None,
        )?
    };

    // Create D3D11 device for DirectComposition.
    let mut device: Option<ID3D11Device> = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )?;
    }
    let device = device.unwrap();
    let dxgi_device: IDXGIDevice2 = device.cast()?;

    // --- Build the subduction layer tree ---
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut sub_ids: Vec<LayerId> = Vec::new();
    for _ in 0..NUM_LAYERS {
        let layer_id = store.create_layer();
        store.add_child(root_id, layer_id);
        store.set_bounds(layer_id, Size::new(LAYER_SIZE, LAYER_SIZE));
        sub_ids.push(layer_id);
    }

    // Create the DComp presenter.
    let composition = backend::CompositionManager::with_device(&dxgi_device, hwnd)?;
    let mut presenter = DCompPresenter::new(composition);

    // Initial evaluate — produces `added` entries that the presenter needs.
    let changes = store.evaluate();
    presenter.apply(&store, &changes);

    // --- Fill each layer with a solid color via IDCompositionSurface ---
    let dcomp_device = presenter.composition().device();
    let size = LAYER_SIZE as u32;
    for (i, &layer_id) in sub_ids.iter().enumerate() {
        fill_layer_color(
            dcomp_device,
            &device,
            presenter.visual_for(layer_id.index()).unwrap(),
            size,
            size,
            COLORS[i],
        )?;
    }
    // Commit the content changes.
    presenter.commit()?;

    // --- Scheduler ---
    let timebase = backend::timebase();
    let start_ticks = backend::now().ticks();
    let scheduler = Scheduler::new(SchedulerConfig::windows());

    // --- Start tick source ---
    let tick_source = TickSource::start(hwnd);

    // SAFETY: Single-threaded access before message pump starts.
    unsafe {
        *STATE.0.get() = Some(AnimState {
            store,
            presenter,
            scheduler,
            sub_ids,
            start_ticks,
            timebase,
            frame_index: 0,
            prev_present_time: None,
            pending_feedback: None,
            recorder: RecorderSink::new(),
            _tick_source: tick_source,
        });
    }

    // Message pump.
    let mut msg = MSG::default();
    while unsafe { GetMessageW(&mut msg, None, 0, 0) }.into() {
        unsafe { DispatchMessageW(&msg) };
    }

    // Flush trace on exit.
    flush_trace();

    // SAFETY: Single-threaded cleanup after message pump exits.
    unsafe { *STATE.0.get() = None };
    Ok(())
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_DESTROY => {
            // SAFETY: Called in response to WM_DESTROY.
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        m if m == WM_APP_TICK => {
            on_tick();
            LRESULT(0)
        }
        // SAFETY: Default window procedure for unhandled messages.
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn on_tick() {
    let Some(s) = state_mut() else { return };

    s.frame_index += 1;
    let tick = make_tick(REFRESH_NS, s.frame_index, s.prev_present_time);
    let frame_index = tick.frame_index;

    // Resolve previous frame's feedback.
    if let Some(pending) = s.pending_feedback.take() {
        let feedback = pending.resolve(tick.prev_actual_present);
        s.scheduler.observe(&feedback);
        s.recorder.on_present_feedback(&PresentFeedbackEvent {
            frame_index: frame_index.saturating_sub(1),
            actual_present: tick.prev_actual_present,
            missed_deadline: feedback.missed_deadline,
        });
    }

    let tick_event = FrameTickEvent::from(&tick);
    s.recorder.on_frame_tick(&tick_event);

    // --- Plan phase ---
    let plan_start = backend::now();
    s.recorder.on_phase_begin(&PhaseBeginEvent {
        frame_index,
        phase: PhaseKind::Plan,
        timestamp: plan_start,
    });

    let hints = compute_hints(&tick, SAFETY_MARGIN_NS);
    let plan = s.scheduler.plan(&tick, &hints);

    let plan_end = backend::now();
    s.recorder.on_phase_end(&PhaseEndEvent {
        frame_index,
        phase: PhaseKind::Plan,
        timestamp: plan_end,
    });

    let plan_event = FramePlanEvent::new(&plan, s.scheduler.safety_margin_ticks());
    s.recorder.on_frame_plan(&plan_event);

    let mut summary = FrameSummaryBuilder::new(&tick_event, &plan_event);
    summary.phase_begin(PhaseKind::Plan, plan_start);
    summary.phase_end(PhaseKind::Plan, plan_end);

    // Convert semantic_time to elapsed seconds.
    let elapsed_ticks = plan.semantic_time.ticks().saturating_sub(s.start_ticks);
    let elapsed_nanos = s.timebase.ticks_to_nanos(elapsed_ticks);
    let t = elapsed_nanos as f64 / 1_000_000_000.0;

    // Animate.
    animate_transforms(&mut s.store, &s.sub_ids, t);

    // --- Evaluate phase ---
    let eval_start = backend::now();
    summary.phase_begin(PhaseKind::Evaluate, eval_start);
    s.recorder.on_phase_begin(&PhaseBeginEvent {
        frame_index,
        phase: PhaseKind::Evaluate,
        timestamp: eval_start,
    });
    let changes = s.store.evaluate();
    let eval_end = backend::now();
    summary.phase_end(PhaseKind::Evaluate, eval_end);
    s.recorder.on_phase_end(&PhaseEndEvent {
        frame_index,
        phase: PhaseKind::Evaluate,
        timestamp: eval_end,
    });

    // --- Render (apply) phase ---
    let render_start = backend::now();
    summary.phase_begin(PhaseKind::Render, render_start);
    s.recorder.on_phase_begin(&PhaseBeginEvent {
        frame_index,
        phase: PhaseKind::Render,
        timestamp: render_start,
    });
    s.presenter.apply(&s.store, &changes);
    let render_end = backend::now();
    summary.phase_end(PhaseKind::Render, render_end);
    s.recorder.on_phase_end(&PhaseEndEvent {
        frame_index,
        phase: PhaseKind::Render,
        timestamp: render_end,
    });

    // --- Submit phase ---
    let submit_start = backend::now();
    s.recorder.on_phase_begin(&PhaseBeginEvent {
        frame_index,
        phase: PhaseKind::Submit,
        timestamp: submit_start,
    });
    s.recorder.on_submit(&SubmitEvent {
        frame_index,
        submitted_at: submit_start,
        expected_present: hints.desired_present,
    });
    let submit_end = backend::now();
    s.recorder.on_phase_end(&PhaseEndEvent {
        frame_index,
        phase: PhaseKind::Submit,
        timestamp: submit_end,
    });

    summary.phase_begin(PhaseKind::Submit, submit_start);
    summary.phase_end(PhaseKind::Submit, submit_end);
    summary.set_missed_deadline(submit_end > plan.commit_deadline);
    s.recorder.on_frame_summary(&summary.finish());

    // Store pending feedback for next tick.
    s.prev_present_time = Some(tick.now);
    s.pending_feedback = Some(PendingFeedback {
        hints,
        build_start: plan_start,
        submitted_at: submit_start,
    });
}

fn animate_transforms(store: &mut LayerStore, sub_ids: &[LayerId], t: f64) {
    let cx = WINDOW_W / 2.0;
    let cy = WINDOW_H / 2.0;

    for (i, &layer_id) in sub_ids.iter().enumerate() {
        let phase = i as f64 * core::f64::consts::TAU / NUM_LAYERS as f64;
        let radius = 150.0 + 50.0 * (t * 0.5 + phase).sin();
        let angle = t * (0.6 + i as f64 * 0.1) + phase;

        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();

        let rotation = t * 2.0 + phase;
        let transform =
            Transform3d::from_translation(x, y, 0.0) * Transform3d::from_rotation_z(rotation);
        store.set_transform(layer_id, transform);

        #[expect(
            clippy::cast_possible_truncation,
            reason = "Opacity is intentionally narrowed from bounded [0,1] f64 to f32"
        )]
        let opacity = (0.5 + 0.5 * (t * 1.5 + phase).sin()) as f32;
        store.set_opacity(layer_id, opacity);

        // Animate bounds on the last layer — pulsing size.
        if i == NUM_LAYERS - 1 {
            let scale = 0.75 + 0.5 * (t * 1.2 + phase).sin();
            let size = LAYER_SIZE * scale;
            store.set_bounds(layer_id, Size::new(size, size));
        }
    }
}

/// Create an `IDCompositionSurface`, fill it with a solid color, and
/// set it as the visual's content.
fn fill_layer_color(
    dcomp_device: &IDCompositionDevice,
    d3d_device: &ID3D11Device,
    visual: &windows::Win32::Graphics::DirectComposition::IDCompositionVisual,
    width: u32,
    height: u32,
    color: [f32; 4],
) -> windows::core::Result<()> {
    // SAFETY: COM calls to create surface, draw into it, and set content.
    unsafe {
        let surface = dcomp_device.CreateSurface(
            width,
            height,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_ALPHA_MODE_PREMULTIPLIED,
        )?;

        let mut offset = POINT::default();
        let dxgi_surface: IDXGISurface = surface.BeginDraw(None, &mut offset)?;
        let texture: ID3D11Texture2D = dxgi_surface.cast()?;

        let mut rtv: Option<ID3D11RenderTargetView> = None;
        d3d_device.CreateRenderTargetView(&texture, None, Some(&mut rtv))?;
        let rtv = rtv.unwrap();

        let context = d3d_device.GetImmediateContext()?;
        context.ClearRenderTargetView(&rtv, &color);

        surface.EndDraw()?;
        visual.SetContent(&surface)?;
    }
    Ok(())
}

fn flush_trace() {
    let Some(s) = state_ref() else { return };
    let bytes = s.recorder.as_bytes();
    if bytes.is_empty() {
        return;
    }
    let path = "trace.json";
    match std::fs::File::create(path) {
        Ok(mut file) => {
            if let Err(e) = subduction_debug::chrome::export(bytes, s.timebase, &mut file) {
                eprintln!("Failed to write {path}: {e}");
            } else {
                eprintln!("Wrote {path} ({} bytes recorded)", bytes.len());
            }
        }
        Err(e) => eprintln!("Failed to create {path}: {e}"),
    }
}
