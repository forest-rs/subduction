// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Windows stress-test: 1000+ animated DirectComposition layers in
//! grouped orbits.
//!
//! Creates a window with 10 group layers, each containing 100 child
//! layers. Groups orbit the window center; children orbit their group.
//! Uses `lotta_layers_common` for the shared animation logic.
//!
//! Run with: `cargo run -p windows_lotta_layers`

#![expect(unsafe_code, reason = "Win32 windowing example requires unsafe code")]

use std::cell::UnsafeCell;

use lotta_layers_common::LAYER_SIZE;
use subduction_backend_windows::{
    self as backend, DCompPresenter, Presenter as _, TickSource, WM_APP_TICK, compute_hints,
    make_tick,
};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::HostTime;
use subduction_core::timing::PendingFeedback;
use subduction_core::trace::{
    FramePlanEvent, FrameSummaryBuilder, FrameTickEvent, PhaseBeginEvent, PhaseEndEvent, PhaseKind,
    PresentFeedbackEvent, SubmitEvent, TraceSink as _,
};

use kurbo::Size;
use subduction_debug::recorder::RecorderSink;

use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::DirectComposition::IDCompositionDevice;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows_core::Interface;

const WINDOW_W: f64 = 1024.0;
const WINDOW_H: f64 = 768.0;
const NUM_GROUPS: usize = 10;
const LAYERS_PER_GROUP: usize = 100;

/// 60 Hz refresh interval in nanoseconds.
const REFRESH_NS: u64 = 16_666_667;

/// Safety margin for present hints (2 ms).
const SAFETY_MARGIN_NS: u64 = 2_000_000;

struct AnimState {
    store: LayerStore,
    presenter: DCompPresenter,
    scheduler: Scheduler,
    group_ids: Vec<LayerId>,
    child_ids: Vec<LayerId>,
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

    let class_name = windows::core::w!("SubductionLottaLayers");
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
            windows::core::w!("Subduction \u{00b7} Lotta Layers (1000+) (Windows)"),
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

    let mut group_ids = Vec::with_capacity(NUM_GROUPS);
    let mut child_ids = Vec::with_capacity(NUM_GROUPS * LAYERS_PER_GROUP);

    for _ in 0..NUM_GROUPS {
        let group = store.create_layer();
        store.add_child(root_id, group);
        group_ids.push(group);

        for _ in 0..LAYERS_PER_GROUP {
            let child = store.create_layer();
            store.add_child(group, child);
            store.set_bounds(child, Size::new(LAYER_SIZE, LAYER_SIZE));
            child_ids.push(child);
        }
    }

    // Create the DComp presenter.
    let composition = backend::CompositionManager::with_device(&dxgi_device, hwnd)?;
    let mut presenter = DCompPresenter::new(composition);

    // Initial evaluate and present.
    let changes = store.evaluate();
    presenter.apply(&store, &changes);

    // --- Fill each child layer with a solid color (golden-angle hue spacing) ---
    let dcomp_device = presenter.composition().device();
    let size = LAYER_SIZE as u32;
    for (i, &child_id) in child_ids.iter().enumerate() {
        let [r, g, b] = lotta_layers_common::hsl_to_rgb((i as f64 * 137.508) % 360.0, 0.7, 0.6);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Color values are in [0,1] range"
        )]
        let color = [r as f32, g as f32, b as f32, 0.9];
        fill_layer_color(
            dcomp_device,
            &device,
            presenter.visual_for(child_id.index()).unwrap(),
            size,
            size,
            color,
        )?;
    }
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
            group_ids,
            child_ids,
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

    let elapsed_ticks = plan.semantic_time.ticks().saturating_sub(s.start_ticks);
    let elapsed_nanos = s.timebase.ticks_to_nanos(elapsed_ticks);
    let t = elapsed_nanos as f64 / 1_000_000_000.0;

    // Animate using the shared lotta_layers_common logic.
    lotta_layers_common::animate_groups(
        &mut s.store,
        &s.group_ids,
        &s.child_ids,
        NUM_GROUPS,
        LAYERS_PER_GROUP,
        WINDOW_W / 2.0,
        WINDOW_H / 2.0,
        t,
    );

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
