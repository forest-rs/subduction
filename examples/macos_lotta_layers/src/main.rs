// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! macOS stress-test: 1000+ animated `CALayer`s in grouped orbits.
//!
//! Creates a window with 10 group layers, each containing 100 child layers.
//! Groups orbit the window center; children orbit their group.
//!
//! Run with: `cargo run -p macos-lotta-layers`

#![expect(unsafe_code, reason = "FFI example requires unsafe code")]

use core::cell::RefCell;

use lotta_layers_common::LAYER_SIZE;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
    NSWindow, NSWindowDelegate, NSWindowStyleMask,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_core_graphics::CGColor;
use objc2_foundation::{NSNotification, NSObject, NSObjectProtocol, NSString};
use objc2_quartz_core::CALayer;
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
use subduction_backend_apple::TickForwarder;
use subduction_backend_apple::{
    DisplayLink, LayerPresenter, LayerRoot, Presenter as _, compute_present_hints,
};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::output::{Color, OutputId};
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::Duration;
use subduction_core::timing::{FrameTick, PendingFeedback};
use subduction_core::trace::{
    FramePlanEvent, FrameSummaryBuilder, FrameTickEvent, PhaseBeginEvent, PhaseEndEvent, PhaseKind,
    PresentFeedbackEvent, SubmitEvent, TraceSink as _,
};
use subduction_debug::recorder::RecorderSink;

const WINDOW_W: f64 = 1024.0;
const WINDOW_H: f64 = 768.0;
const NUM_GROUPS: usize = 10;
const LAYERS_PER_GROUP: usize = 100;

define_class! {
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AppDelegate"]
    #[ivars = ()]
    struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn app_did_finish_launching(&self, _notification: &NSNotification) {
            setup_window(MainThreadMarker::from(self));
        }

        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn should_terminate_after_last_window_closed(
            &self,
            _app: &NSApplication,
        ) -> bool {
            true
        }

        #[unsafe(method(applicationWillTerminate:))]
        fn app_will_terminate(&self, _notification: &NSNotification) {
            flush_trace();
        }
    }
}

impl AppDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

define_class! {
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "LottaWindowDelegate"]
    #[ivars = ()]
    struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        #[unsafe(method(windowDidBecomeKey:))]
        fn window_did_become_key(&self, _notification: &NSNotification) {
            start_display_link(MainThreadMarker::from(self));
        }
    }
}

impl WindowDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

fn start_display_link(_mtm: MainThreadMarker) {
    KEEP_ALIVE.with(|cell| {
        if cell.borrow().is_some() {
            return;
        }

        #[cfg(feature = "ca-display-link")]
        let link = {
            let l = DisplayLink::new(on_tick, OutputId(0), _mtm);
            l.start();
            l
        };

        #[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
        let link = {
            let forwarder = TickForwarder::new(on_tick);
            let l = DisplayLink::new(forwarder.sender(), OutputId(0))
                .expect("failed to create CVDisplayLink");
            l.start().expect("failed to start CVDisplayLink");
            l
        };

        *cell.borrow_mut() = Some(link);
    });
}

fn set_layer_bg_color(layer: &CALayer, r: f64, g: f64, b: f64, a: f64) {
    let color = CGColor::new_generic_rgb(r, g, b, a);
    layer.setBackgroundColor(Some(&color));
}

/// Returns an `[r, g, b]` triple in 0.0–1.0 for a given index using
/// golden-angle hue spacing in HSL space.
fn layer_color(index: usize) -> [f64; 3] {
    let hue = (index as f64 * 137.508) % 360.0;
    lotta_layers_common::hsl_to_rgb(hue, 0.7, 0.6)
}

struct AnimState {
    store: LayerStore,
    presenter: LayerPresenter,
    scheduler: Scheduler,
    /// `group_ids[g]` is the `LayerId` for group `g`.
    group_ids: Vec<LayerId>,
    /// `child_ids[g * LAYERS_PER_GROUP + c]` is the `LayerId` for child `c`
    /// of group `g`.
    child_ids: Vec<LayerId>,
    start_ticks: u64,
    timebase: subduction_core::time::Timebase,
    pending_feedback: Option<PendingFeedback>,
    recorder: RecorderSink,
}

thread_local! {
    static ANIM_STATE: RefCell<Option<AnimState>> = const { RefCell::new(None) };
    static KEEP_ALIVE: RefCell<Option<DisplayLink>> = const { RefCell::new(None) };
    static WIN_DELEGATE: RefCell<Option<Retained<WindowDelegate>>> = const { RefCell::new(None) };
}

fn setup_window(mtm: MainThreadMarker) {
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Resizable
        | NSWindowStyleMask::Miniaturizable;

    let frame = CGRect::new(CGPoint::new(100.0, 100.0), CGSize::new(WINDOW_W, WINDOW_H));

    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc::<NSWindow>(),
            frame,
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    unsafe { window.setReleasedWhenClosed(false) };
    window.setTitle(&NSString::from_str("Subduction · Lotta Layers (1000+)"));

    let content_view = window.contentView().expect("window has content view");
    content_view.setWantsLayer(true);

    let root_layer = content_view.layer().expect("content view has no layer");

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
            child_ids.push(child);
        }
    }

    // --- Initial evaluate and present ---
    let backdrop_color = Color::from_rgba8(0x1a, 0x1a, 0x24, 0xff);
    let root = LayerRoot::new(root_layer).with_backdrop_color(backdrop_color);
    let mut presenter = LayerPresenter::new(root);
    let changes = store.evaluate();
    presenter.apply(&store, &changes);

    // --- Style each child layer ---
    for (i, &child_id) in child_ids.iter().enumerate() {
        let [r, g, b] = layer_color(i);
        if let Some(ca) = presenter.get_layer(child_id.index()) {
            set_layer_bg_color(ca, r, g, b, 0.9);
            ca.setCornerRadius(2.0);
            ca.setBounds(CGRect::new(
                CGPoint::new(0.0, 0.0),
                CGSize::new(LAYER_SIZE, LAYER_SIZE),
            ));
        }
    }

    // Group layers are invisible containers (no background).

    window.center();
    window.makeKeyAndOrderFront(None);
    let app = NSApplication::sharedApplication(mtm);
    #[expect(deprecated, reason = "explicit foreground activation for demo startup")]
    app.activateIgnoringOtherApps(true);

    // --- Animation setup ---
    let timebase = DisplayLink::timebase();
    let start_ticks = DisplayLink::now().ticks();
    let scheduler = Scheduler::new(SchedulerConfig::macos());

    ANIM_STATE.with(|cell| {
        *cell.borrow_mut() = Some(AnimState {
            store,
            presenter,
            scheduler,
            group_ids,
            child_ids,
            start_ticks,
            timebase,
            pending_feedback: None,
            recorder: RecorderSink::new(),
        });
    });

    // Defer DisplayLink creation to windowDidBecomeKey: so the display is
    // fully attached.
    let win_delegate = WindowDelegate::new(mtm);
    window.setDelegate(Some(ProtocolObject::from_ref(&*win_delegate)));
    WIN_DELEGATE.with(|cell| {
        *cell.borrow_mut() = Some(win_delegate);
    });
}

fn on_tick(tick: FrameTick) {
    ANIM_STATE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let Some(s) = borrow.as_mut() else { return };

        let frame_index = tick.frame_index;

        // Resolve previous frame's feedback with actual_present from this tick.
        if let Some(pending) = s.pending_feedback.take() {
            let feedback = pending.resolve(tick.prev_actual_present);
            s.scheduler.observe(&feedback);
            s.recorder.on_present_feedback(&PresentFeedbackEvent {
                frame_index: frame_index.saturating_sub(1),
                actual_present: tick.prev_actual_present,
                missed_deadline: feedback.missed_deadline,
            });
        }

        // Record the tick event.
        let tick_event = FrameTickEvent::from(&tick);
        s.recorder.on_frame_tick(&tick_event);

        // --- Plan phase ---
        let plan_start = DisplayLink::now();
        s.recorder.on_phase_begin(&PhaseBeginEvent {
            frame_index,
            phase: PhaseKind::Plan,
            timestamp: plan_start,
        });

        let safety = Duration(s.scheduler.safety_margin_ticks());
        let hints = compute_present_hints(&tick, safety);
        let plan = s.scheduler.plan(&tick, &hints);

        let plan_end = DisplayLink::now();
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

        // Animate.
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
        let eval_start = DisplayLink::now();
        summary.phase_begin(PhaseKind::Evaluate, eval_start);
        s.recorder.on_phase_begin(&PhaseBeginEvent {
            frame_index,
            phase: PhaseKind::Evaluate,
            timestamp: eval_start,
        });
        let changes = s.store.evaluate();
        let eval_end = DisplayLink::now();
        summary.phase_end(PhaseKind::Evaluate, eval_end);
        s.recorder.on_phase_end(&PhaseEndEvent {
            frame_index,
            phase: PhaseKind::Evaluate,
            timestamp: eval_end,
        });

        // --- Render (apply) phase ---
        let render_start = DisplayLink::now();
        summary.phase_begin(PhaseKind::Render, render_start);
        s.recorder.on_phase_begin(&PhaseBeginEvent {
            frame_index,
            phase: PhaseKind::Render,
            timestamp: render_start,
        });
        s.presenter.apply(&s.store, &changes);
        let render_end = DisplayLink::now();
        summary.phase_end(PhaseKind::Render, render_end);
        s.recorder.on_phase_end(&PhaseEndEvent {
            frame_index,
            phase: PhaseKind::Render,
            timestamp: render_end,
        });

        // --- Submit phase ---
        let submit_start = DisplayLink::now();
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
        let submit_end = DisplayLink::now();
        s.recorder.on_phase_end(&PhaseEndEvent {
            frame_index,
            phase: PhaseKind::Submit,
            timestamp: submit_end,
        });

        summary.phase_begin(PhaseKind::Submit, submit_start);
        summary.phase_end(PhaseKind::Submit, submit_end);
        summary.set_missed_deadline(submit_end > plan.commit_deadline);
        s.recorder.on_frame_summary(&summary.finish());

        // Store pending feedback for resolution on next tick.
        s.pending_feedback = Some(PendingFeedback {
            hints,
            build_start: plan_start,
            submitted_at: submit_start,
        });
    });
}

fn flush_trace() {
    ANIM_STATE.with(|cell| {
        let borrow = cell.borrow();
        let Some(s) = borrow.as_ref() else { return };
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
    });
}

fn main() {
    let mtm = MainThreadMarker::new().expect("must be on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    let delegate = AppDelegate::new(mtm);
    let delegate_proto = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(delegate_proto));

    app.run();
}
