// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! macOS example: animated `CALayer`s and `AppKit` widgets driven by
//! `subduction_backend_apple`.
//!
//! Creates a window with two colored layers that orbit and pulse opacity,
//! plus three embedded `AppKit` widgets (button, text field, spinner) that
//! orbit alongside them — demonstrating that subduction's layer tree can
//! host real, interactive platform controls.
//!
//! Run with: `cargo run -p macos-layers`

#![expect(unsafe_code, reason = "FFI example requires unsafe code")]

use core::cell::RefCell;

use frameclock::scheduler::DegradationPolicy;
use frameclock::time::Timebase;
use frameclock::{
    ActiveFrame, DisplayTiming, Duration, FrameBeginResult, FrameDemand, FrameTick, OutputId,
    SchedulerConfig,
};
#[cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))]
use frameclock_apple::TickForwarder;
use frameclock_apple::{AppleFeedbackMode, AppleFrameClock, DisplayLink};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
    NSButton, NSProgressIndicator, NSProgressIndicatorStyle, NSTextField, NSWindow,
    NSWindowDelegate, NSWindowStyleMask,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_core_graphics::CGColor;
use objc2_foundation::{NSNotification, NSObject, NSObjectProtocol, NSString};
use objc2_quartz_core::CALayer;
use subduction_backend_apple::{LayerPresenter, LayerRoot, Presenter as _};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::output::Color;
use subduction_core::transform::Transform3d;

use kurbo::Size;

const WINDOW_W: f64 = 800.0;
const WINDOW_H: f64 = 600.0;
const NUM_LAYERS: usize = 5;

/// Layers `0..NUM_WIDGETS` are backed by `AppKit` widgets; the rest are
/// `CALayer` circles.
const NUM_WIDGETS: usize = 3;

/// Colors for the circle layers (indices `NUM_WIDGETS..NUM_LAYERS`).
const COLORS: [[f64; 4]; NUM_LAYERS - NUM_WIDGETS] = [
    [0.95, 0.26, 0.21, 0.9], // red
    [0.13, 0.59, 0.95, 0.9], // blue
];

const LAYER_SIZE: f64 = 80.0;
const FALLBACK_REFRESH_INTERVAL_NANOS: u64 = 16_666_667;

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
    #[name = "WindowDelegate"]
    #[ivars = ()]
    struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        #[unsafe(method(windowDidBecomeKey:))]
        fn window_did_become_key(&self, _notification: &NSNotification) {
            // Create the DisplayLink only once, deferred until the window is
            // on screen. This avoids a NULL return from CADisplayLink on
            // machines where the display isn't ready at launch (e.g. Mac mini
            // with an external monitor).
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

/// Creates and starts the `DisplayLink`, storing it in `KEEP_ALIVE`.
///
/// Called once from `windowDidBecomeKey:` so the display is fully attached.
/// No-op if already created.
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

/// All mutable state lives in a main-thread-only `thread_local!`.
///
/// The `CADisplayLink` callback runs directly on the main thread, accessing
/// this state through the thread-local.
struct AnimState {
    store: LayerStore,
    presenter: LayerPresenter,
    frame_clock: AppleFrameClock,
    sub_ids: Vec<LayerId>,
    start_ticks: u64,
    timebase: Timebase,
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

    let frame = CGRect::new(CGPoint::new(200.0, 200.0), CGSize::new(WINDOW_W, WINDOW_H));

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
    window.setTitle(&NSString::from_str(
        "Subduction · Animated Layers + AppKit Widgets",
    ));

    // Dark background via CGColor on the content view's layer.
    let content_view = window.contentView().expect("window has content view");
    content_view.setWantsLayer(true);

    let root_layer = content_view.layer().expect("content view has no layer");

    // --- Build the subduction layer tree ---
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut sub_ids: Vec<LayerId> = Vec::new();

    // Create and style the scene root, then hand it to the presenter.
    let backdrop_color = Color::from_rgba8(0x1f, 0x1f, 0x26, 0xff);
    let root = LayerRoot::new(root_layer).with_backdrop_color(backdrop_color);
    let mut presenter = LayerPresenter::new(root);

    for _ in 0..NUM_LAYERS {
        let layer_id = store.create_layer();
        store.add_child(root_id, layer_id);
        sub_ids.push(layer_id);
    }

    // Initial evaluate — produces `added` entries that the presenter needs.
    let changes = store.evaluate();
    presenter.apply(&store, &changes);

    // --- Create `AppKit` widgets for layers 0–2 and attach to presenter ---

    // Index 0: NSButton
    let button = unsafe {
        NSButton::buttonWithTitle_target_action(&NSString::from_str("Click Me"), None, None, mtm)
    };
    button.sizeToFit();
    button.setWantsLayer(true);
    content_view.addSubview(&button);
    presenter.attach_view(
        sub_ids[0].index(),
        Retained::into_super(Retained::into_super(button)),
    );

    // Index 1: NSTextField
    let field = NSTextField::textFieldWithString(&NSString::from_str("Type here\u{2026}"), mtm);
    field.setEditable(true);
    field.setBezeled(true);
    field.sizeToFit();
    let field_h = field.frame().size.height;
    field.setFrameSize(CGSize::new(150.0, field_h));
    field.setWantsLayer(true);
    content_view.addSubview(&field);
    presenter.attach_view(
        sub_ids[1].index(),
        Retained::into_super(Retained::into_super(field)),
    );

    // Index 2: NSProgressIndicator (spinner)
    let spinner = NSProgressIndicator::initWithFrame(
        mtm.alloc::<NSProgressIndicator>(),
        CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(32.0, 32.0)),
    );
    spinner.setStyle(NSProgressIndicatorStyle::Spinning);
    spinner.setIndeterminate(true);
    unsafe { spinner.startAnimation(None) };
    spinner.setWantsLayer(true);
    content_view.addSubview(&spinner);
    spinner.sizeToFit();
    presenter.attach_view(sub_ids[2].index(), Retained::into_super(spinner));

    // --- Style the remaining circle layers (indices 3–4) ---
    for (ci, &layer_id) in sub_ids[NUM_WIDGETS..].iter().enumerate() {
        let [r, g, b, a] = COLORS[ci];
        if let Some(ca) = presenter.get_layer(layer_id.index()) {
            set_layer_bg_color(ca, r, g, b, a);
            ca.setCornerRadius(12.0);
            ca.setBounds(CGRect::new(
                CGPoint::new(0.0, 0.0),
                CGSize::new(LAYER_SIZE, LAYER_SIZE),
            ));
        }
    }

    window.center();
    window.makeKeyAndOrderFront(None);
    let app = NSApplication::sharedApplication(mtm);
    #[expect(deprecated, reason = "explicit foreground activation for demo startup")]
    app.activateIgnoringOtherApps(true);

    // --- CADisplayLink-driven animation ---
    let timebase = DisplayLink::timebase();
    let start_ticks = DisplayLink::now().ticks();
    let fallback_interval = Duration(timebase.nanos_to_ticks(FALLBACK_REFRESH_INTERVAL_NANOS));
    let mut config = SchedulerConfig::predictive();
    // This example uses the display-link callback itself as the frame-start
    // wake, so keep pipeline depth fixed at 1. Hosts that mirror `WaitUntil`
    // into their own timer queue can leave adaptive depth enabled.
    config.min_depth = 1;
    config.max_depth = 1;
    config.initial_depth = 1;
    config.degradation_policy = DegradationPolicy::Fixed;
    config.minimum_frame_start_margin = fallback_interval;
    let frame_clock = AppleFrameClock::new_with_feedback_mode(
        config,
        DisplayTiming::fixed(fallback_interval),
        apple_feedback_mode(),
    );

    // Store all main-thread state in the thread-local.
    ANIM_STATE.with(|cell| {
        *cell.borrow_mut() = Some(AnimState {
            store,
            presenter,
            frame_clock,
            sub_ids,
            start_ticks,
            timebase,
        });
    });

    // Defer DisplayLink creation to windowDidBecomeKey: so the display is
    // fully attached. This avoids a NULL CADisplayLink on Mac minis with
    // external monitors.
    let win_delegate = WindowDelegate::new(mtm);
    window.setDelegate(Some(ProtocolObject::from_ref(&*win_delegate)));
    WIN_DELEGATE.with(|cell| {
        *cell.borrow_mut() = Some(win_delegate);
    });
}

fn apple_feedback_mode() -> AppleFeedbackMode {
    #[cfg(feature = "ca-display-link")]
    {
        AppleFeedbackMode::DeferredActualPresent
    }
    #[cfg(not(feature = "ca-display-link"))]
    {
        AppleFeedbackMode::CommitOnly
    }
}

fn on_tick(tick: FrameTick) {
    ANIM_STATE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let Some(s) = borrow.as_mut() else { return };

        s.frame_clock.request(FrameDemand::ANIMATION);
        let begin = s.frame_clock.begin_frame(tick);

        match begin.result {
            FrameBeginResult::Ready(frame) => render_frame(s, frame),
            FrameBeginResult::WaitUntil(_) | FrameBeginResult::Idle => {}
            FrameBeginResult::Expired(_) => {
                s.frame_clock.request(FrameDemand::ANIMATION);
            }
        }
    });
}

fn render_frame(s: &mut AnimState, frame: ActiveFrame) {
    let plan = frame.plan();
    apply_preferred_frame_interval(s, &frame);

    // Convert sample_time to elapsed seconds for the animation.
    let elapsed_ticks = plan.sample_time.ticks().saturating_sub(s.start_ticks);
    let elapsed_nanos = s.timebase.ticks_to_nanos(elapsed_ticks);
    let t = elapsed_nanos as f64 / 1_000_000_000.0;

    // Animate.
    animate_transforms(&mut s.store, &s.sub_ids, t);

    let changes = s.store.evaluate();
    s.presenter.apply(&s.store, &changes);
    let _submit = s.frame_clock.submit_frame_now(frame);
}

fn apply_preferred_frame_interval(s: &AnimState, frame: &ActiveFrame) {
    #[cfg(feature = "ca-display-link")]
    {
        let Some(range) = s.frame_clock.preferred_frame_rate_range(frame) else {
            return;
        };
        KEEP_ALIVE.with(|cell| {
            let borrow = cell.borrow();
            let Some(link) = borrow.as_ref() else {
                return;
            };
            link.set_preferred_frame_rate_range(range);
        });
    }
    #[cfg(not(feature = "ca-display-link"))]
    {
        _ = (s, frame);
    }
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

        if i < NUM_WIDGETS {
            // Widget layers: translate only (no rotation) so that AppKit
            // hit-testing and text editing work correctly.
            let transform = Transform3d::from_translation(x, y, 0.0);
            store.set_transform(layer_id, transform);
        } else {
            // Circle layers: orbit + rotation + opacity pulsing.
            let rotation = t * 2.0 + phase;
            let transform =
                Transform3d::from_translation(x, y, 0.0) * Transform3d::from_rotation_z(rotation);
            store.set_transform(layer_id, transform);

            #[expect(
                clippy::cast_possible_truncation,
                reason = "Opacity is intentionally narrowed from bounded [0, 1] f64 to f32 for layer API"
            )]
            let opacity = (0.5 + 0.5 * (t * 1.5 + phase).sin()) as f32;
            store.set_opacity(layer_id, opacity);

            // Animate bounds on the last circle layer (blue) — pulsing size.
            if i == NUM_LAYERS - 1 {
                let scale = 0.75 + 0.5 * (t * 1.2 + phase).sin();
                let size = LAYER_SIZE * scale;
                store.set_bounds(layer_id, Size::new(size, size));
            }
        }
    }
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
