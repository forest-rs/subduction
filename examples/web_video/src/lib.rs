// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Web audio-video sync torture test with beat-locked visual patterns.
//!
//! This example renders a real `<video>` element (native controls disabled),
//! then overlays beat-locked patterns and generated click audio so sync drift is
//! both visible and audible.
//!
//! Build with: `wasm-pack build --target web examples/web-video`
//! Then serve `examples/web-video/` and open `index.html`.

#![no_std]
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "this crate only runs in the browser")
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "f64→f32 opacity casts and u64→usize casts are intentional"
)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString as _};
use core::cell::RefCell;
use core::f64::consts::TAU;

use subduction_backend_web::RafLoop;
use subduction_backend_web::{DomPresenter, LayerRoot, Presenter as _};
use subduction_core::clock::AffineClock;
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::output::OutputId;
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::{Duration, HostTime, Timebase};
use subduction_core::timing::{FrameTick, PresentFeedback, TimingConfidence};
use subduction_core::transform::Transform3d;
use subduction_sync_harness::{PathologyToggles, SyncSample, SyncTracker};
use wasm_bindgen::JsCast as _;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use web_sys::{
    AudioContext, Document, Event, GainNode, HtmlButtonElement, HtmlElement, HtmlInputElement,
    HtmlVideoElement, OscillatorType,
};

const VIDEO_W: f64 = 848.0;
const VIDEO_H: f64 = 480.0;
const VIDEO_URL: &str = "https://github.com/vidanov/video/raw/master/test_files/1080p50.mp4";
const BEAT_HZ: f64 = 2.0;
const GRAPH_SAMPLES: usize = 60;
const TIMER_JITTER_MAX_S: f64 = 0.010;
const DECODE_JITTER_MAX_MS: f64 = 20.0;
const DECODE_JITTER_SPIKE_MS: f64 = 35.0;
const GPU_STALL_MS: f64 = 14.0;

struct PathologyUi {
    decode_jitter: HtmlInputElement,
    gpu_stall: HtmlInputElement,
    timer_jitter: HtmlInputElement,
    vary_refresh: HtmlInputElement,
}

struct VideoUi {
    play_button: HtmlButtonElement,
    seek: HtmlInputElement,
    progress_fill: HtmlElement,
    metrics: HtmlElement,
    hud: HtmlElement,
    timecode: HtmlElement,
    graph: HtmlElement,
    sync_grade: HtmlElement,
    tooltip_element: HtmlElement,
    pathologies: PathologyUi,
}

struct VideoState {
    store: LayerStore,
    presenter: DomPresenter,
    scheduler: Scheduler,
    timebase: Timebase,
    app_start: HostTime,
    media_clock: AffineClock,
    video: HtmlVideoElement,
    ui: VideoUi,
    sweep_id: LayerId,
    flash_id: LayerId,
    hand_id: LayerId,
    tooltip_id: LayerId,
    audio: Option<AudioContext>,
    last_beat: u64,
    sync: SyncTracker<GRAPH_SAMPLES>,
    prev_tick_us: Option<u64>,
    dropped_frames: u64,
    duplicated_frames: u64,
    rng_state: u64,
    last_timer_jitter_ms: f64,
    last_decode_jitter_ms: f64,
    last_gpu_stall_ms: f64,
}

fn seconds_per_tick(timebase: Timebase) -> f64 {
    f64::from(timebase.numer) / f64::from(timebase.denom) / 1e9
}

fn reanchor_media_clock(clock: &mut AffineClock, timebase: Timebase, host: HostTime, media: f64) {
    *clock = AffineClock::new(seconds_per_tick(timebase), 0.08, 0.08);
    clock.update(host.ticks(), media);
}

/// Entry point for the web-video demo.
#[cfg_attr(all(target_arch = "wasm32", not(test)), wasm_bindgen(start))]
pub fn main() -> Result<(), JsValue> {
    let document = web_sys::window()
        .expect("window")
        .document()
        .expect("document");

    let shell = create_shell(&document)?;
    document.body().expect("body").append_child(&shell)?;

    let video_wrap = element(&document, "div")?;
    style(
        &video_wrap,
        "position: relative; width: 848px; height: 480px; border-radius: 14px; overflow: hidden; box-shadow: 0 24px 50px rgba(7,18,44,0.28);",
    )?;

    let video: HtmlVideoElement = document.create_element("video")?.unchecked_into();
    video.set_src(VIDEO_URL);
    video.set_controls(false);
    video.set_loop(true);
    video.set_muted(true);
    video.set_autoplay(false);
    video.set_preload("auto");
    video.set_attribute("playsinline", "")?;
    video.set_width(VIDEO_W as u32);
    video.set_height(VIDEO_H as u32);
    style(
        &video,
        "width: 100%; height: 100%; object-fit: cover; background: #10121f; display: block;",
    )?;
    video_wrap.append_child(&video)?;

    let overlay_host = element(&document, "div")?;
    style(
        &overlay_host,
        "position: absolute; inset: 0; pointer-events: none;",
    )?;
    video_wrap.append_child(&overlay_host)?;

    let controls = element(&document, "div")?;
    style(
        &controls,
        "display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: center; width: 848px;",
    )?;

    let play_button: HtmlButtonElement = document.create_element("button")?.unchecked_into();
    play_button.set_text_content(Some("Play + Arm Audio"));
    style(
        &play_button,
        "border: 0; border-radius: 999px; padding: 10px 18px; background: #0f5d71; color: #eff8ff; font-weight: 600; cursor: pointer;",
    )?;
    controls.append_child(&play_button)?;

    let seek: HtmlInputElement = document.create_element("input")?.unchecked_into();
    seek.set_type("range");
    seek.set_min("0");
    seek.set_max("1000");
    seek.set_value("0");
    style(&seek, "width: 100%;")?;
    controls.append_child(&seek)?;

    let progress_track = element(&document, "div")?;
    style(
        &progress_track,
        "width: 848px; height: 10px; border-radius: 999px; background: rgba(16,21,45,0.16); overflow: hidden;",
    )?;
    let progress_fill = element(&document, "div")?;
    style(
        &progress_fill,
        "width: 0%; height: 100%; background: linear-gradient(90deg, #0f5d71, #22b4b9);",
    )?;
    progress_track.append_child(&progress_fill)?;

    let toggles_panel = element(&document, "div")?;
    style(
        &toggles_panel,
        "width: 848px; display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 6px 14px; font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; color: #17304a;",
    )?;
    let decode_jitter = add_checkbox(
        &document,
        &toggles_panel,
        "decode jitter (0-12ms)",
        "Inject random CPU stalls before media sync to simulate decode jitter. Includes periodic spikes.",
        false,
    )?;
    let gpu_stall = add_checkbox(
        &document,
        &toggles_panel,
        "gpu stall (8ms busy spin)",
        "Inject extra work in the frame critical path to simulate expensive render passes.",
        false,
    )?;
    let timer_jitter = add_checkbox(
        &document,
        &toggles_panel,
        "timer jitter (+/-4ms)",
        "Perturb semantic time in pacing-only mode to emulate timer noise.",
        false,
    )?;
    let vary_refresh = add_checkbox(
        &document,
        &toggles_panel,
        "emulate refresh 60<->120",
        "Toggle effective frame budget between 60 Hz and 120 Hz every 5s.",
        false,
    )?;

    let timecode = element(&document, "pre")?;
    style(
        &timecode,
        "width: 848px; margin: 0; padding: 6px 8px; border-radius: 8px; background: rgba(8,16,32,0.88); color: #d7edff; font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;",
    )?;
    timecode.set_attribute(
        "title",
        "Truth strip: frame index, intended present bucket, beat index, and timing confidence.",
    )?;

    let hud = element(&document, "pre")?;
    style(
        &hud,
        "width: 848px; margin: 0; padding: 6px 8px; border-radius: 8px; background: rgba(241,248,255,0.86); color: #1c3147; font: 12px/1.25 ui-monospace, SFMono-Regular, Menlo, monospace;",
    )?;
    hud.set_attribute(
        "title",
        "Scheduler HUD: timing confidence, Ts/Tp, pipeline depth, and deadline misses.",
    )?;

    let graph = element(&document, "pre")?;
    style(
        &graph,
        "width: 848px; margin: 0; padding: 6px 8px; border-radius: 8px; background: rgba(19,24,37,0.9); color: #9eeab4; font: 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: 0.4px;",
    )?;
    graph.set_attribute(
        "title",
        "Sparkline of last 60 frame deltas (ms). Flatter is steadier pacing.",
    )?;

    let sync_grade = element(&document, "div")?;
    style(
        &sync_grade,
        "width: 848px; padding: 6px 8px; border-radius: 8px; background: rgba(232,244,255,0.9); font: 700 13px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; color: #134b1b;",
    )?;
    sync_grade.set_attribute(
        "title",
        "Composite KPI from phase error and miss rate. A is tight sync, D is unstable.",
    )?;

    let metrics = element(&document, "div")?;
    style(
        &metrics,
        "width: 848px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: #29415c;",
    )?;
    metrics.set_attribute(
        "title",
        "Live counters: media position, drift, A/V phase delta, and intentional drop/dup actions.",
    )?;
    metrics.set_text_content(Some("press Play to arm audio click track"));

    shell.append_child(&video_wrap)?;
    shell.append_child(&controls)?;
    shell.append_child(&progress_track)?;
    shell.append_child(&toggles_panel)?;
    shell.append_child(&timecode)?;
    shell.append_child(&hud)?;
    shell.append_child(&graph)?;
    shell.append_child(&sync_grade)?;
    shell.append_child(&metrics)?;

    let mut store = LayerStore::new();
    let root = store.create_layer();
    let sweep_id = store.create_layer();
    let flash_id = store.create_layer();
    let hand_id = store.create_layer();
    let tooltip_id = store.create_layer();
    store.add_child(root, sweep_id);
    store.add_child(root, flash_id);
    store.add_child(root, hand_id);
    store.add_child(root, tooltip_id);

    let initial = store.evaluate();
    let root = LayerRoot::new(overlay_host);
    let mut presenter = DomPresenter::new(root);
    presenter.apply(&store, &initial);

    if let Some(sweep_el) = presenter.get_element(sweep_id.index()) {
        style(
            sweep_el,
            "position: absolute; width: 2px; height: 100%; background: rgba(255,255,255,0.88); box-shadow: 0 0 0 1px rgba(6,12,23,0.2);",
        )?;
    }
    if let Some(flash_el) = presenter.get_element(flash_id.index()) {
        style(
            flash_el,
            "position: absolute; width: 44px; height: 44px; border-radius: 10px; background: rgba(255,210,82,0.95);",
        )?;
    }
    if let Some(hand_el) = presenter.get_element(hand_id.index()) {
        style(
            hand_el,
            "position: absolute; width: 52px; height: 4px; border-radius: 999px; background: rgba(122,230,255,0.95);",
        )?;
    }
    let tooltip_element = presenter
        .get_element(tooltip_id.index())
        .expect("tooltip element")
        .clone();
    style(
        &tooltip_element,
        "position: absolute; min-width: 220px; padding: 6px 8px; border-radius: 8px; background: rgba(9,15,30,0.78); color: #f2f6ff; font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;",
    )?;

    let timebase = subduction_backend_web::timebase();
    let app_start = subduction_backend_web::now();
    let mut media_clock = AffineClock::new(seconds_per_tick(timebase), 0.08, 0.08);
    media_clock.update(app_start.ticks(), 0.0);

    let mut scheduler_cfg = SchedulerConfig::web();
    scheduler_cfg.nominal_latency = Duration::ZERO;

    let state = Rc::new(RefCell::new(VideoState {
        store,
        presenter,
        scheduler: Scheduler::new(scheduler_cfg),
        timebase,
        app_start,
        media_clock,
        video,
        ui: VideoUi {
            play_button,
            seek,
            progress_fill,
            metrics,
            hud,
            timecode,
            graph,
            sync_grade,
            tooltip_element,
            pathologies: PathologyUi {
                decode_jitter,
                gpu_stall,
                timer_jitter,
                vary_refresh,
            },
        },
        sweep_id,
        flash_id,
        hand_id,
        tooltip_id,
        audio: None,
        last_beat: 0,
        sync: SyncTracker::new(16.67),
        prev_tick_us: None,
        dropped_frames: 0,
        duplicated_frames: 0,
        rng_state: 0x8f2f_3d29_11ab_9121,
        last_timer_jitter_ms: 0.0,
        last_decode_jitter_ms: 0.0,
        last_gpu_stall_ms: 0.0,
    }));

    bind_controls(&state)?;

    let state_cb = Rc::clone(&state);
    let raf = RafLoop::new(move |tick| on_tick(&state_cb, tick), OutputId(0));
    raf.start();
    core::mem::forget(raf);

    Ok(())
}

fn bind_controls(state: &Rc<RefCell<VideoState>>) -> Result<(), JsValue> {
    let play_state = Rc::clone(state);
    let play_cb = Closure::wrap(Box::new(move |_event: Event| {
        let mut s = play_state.borrow_mut();
        ensure_audio_armed(&mut s);

        if s.video.paused() {
            if let Some(audio) = s.audio.as_ref() {
                let _ = audio.resume();
            }
            let _ = s.video.play();
        } else {
            let _ = s.video.pause();
            if let Some(audio) = s.audio.as_ref() {
                let _ = audio.suspend();
            }
        }
    }) as Box<dyn FnMut(_)>);
    state
        .borrow()
        .ui
        .play_button
        .add_event_listener_with_callback("click", play_cb.as_ref().unchecked_ref())?;
    play_cb.forget();

    let seek_state = Rc::clone(state);
    let seek_cb = Closure::wrap(Box::new(move |_event: Event| {
        let mut s = seek_state.borrow_mut();
        let dur = s.video.duration();
        if dur.is_finite() && dur > 0.0 {
            let normalized = s.ui.seek.value_as_number().clamp(0.0, 1000.0) / 1000.0;
            let next_time = dur * normalized;
            s.video.set_current_time(next_time);
            let host = subduction_backend_web::now();
            let timebase = s.timebase;
            reanchor_media_clock(&mut s.media_clock, timebase, host, next_time);
        }
    }) as Box<dyn FnMut(_)>);
    state
        .borrow()
        .ui
        .seek
        .add_event_listener_with_callback("input", seek_cb.as_ref().unchecked_ref())?;
    seek_cb.forget();

    let ended_state = Rc::clone(state);
    let ended_cb = Closure::wrap(Box::new(move |_event: Event| {
        let mut s = ended_state.borrow_mut();
        s.video.set_current_time(0.0);
        let _ = s.video.play();
        let timebase = s.timebase;
        reanchor_media_clock(
            &mut s.media_clock,
            timebase,
            subduction_backend_web::now(),
            0.0,
        );
    }) as Box<dyn FnMut(_)>);
    state
        .borrow()
        .video
        .add_event_listener_with_callback("ended", ended_cb.as_ref().unchecked_ref())?;
    ended_cb.forget();

    Ok(())
}

fn on_tick(state: &Rc<RefCell<VideoState>>, tick: FrameTick) {
    let mut s = state.borrow_mut();
    let frame_delta_ms = if let Some(prev) = s.prev_tick_us {
        (tick.now.ticks().saturating_sub(prev)) as f64 / 1000.0
    } else {
        16.67
    };
    s.prev_tick_us = Some(tick.now.ticks());

    let build_start = subduction_backend_web::now();
    let safety = Duration(s.scheduler.safety_margin_ticks());
    let hints = subduction_backend_web::compute_present_hints(&tick, safety);
    let plan = s.scheduler.plan(&tick, &hints);

    let mut semantic_seconds = ticks_to_secs(
        s.timebase,
        plan.semantic_time
            .ticks()
            .saturating_sub(s.app_start.ticks()),
    );

    let pathologies = PathologyToggles {
        decode_jitter: s.ui.pathologies.decode_jitter.checked(),
        gpu_stall: s.ui.pathologies.gpu_stall.checked(),
        timer_jitter: s.ui.pathologies.timer_jitter.checked(),
        vary_refresh: s.ui.pathologies.vary_refresh.checked(),
    };

    if pathologies.timer_jitter {
        let jitter = rand_range(&mut s.rng_state, -TIMER_JITTER_MAX_S, TIMER_JITTER_MAX_S);
        semantic_seconds = (semantic_seconds + jitter).max(0.0);
        s.last_timer_jitter_ms = jitter * 1000.0;
    } else {
        s.last_timer_jitter_ms = 0.0;
    }

    let target_time = plan.present_time.unwrap_or(plan.semantic_time);
    let target_present_seconds = ticks_to_secs(
        s.timebase,
        target_time.ticks().saturating_sub(s.app_start.ticks()),
    );

    if pathologies.decode_jitter {
        let mut stall = rand_range(&mut s.rng_state, 0.0, DECODE_JITTER_MAX_MS);
        if rand_unit(&mut s.rng_state) < 0.08 {
            stall += DECODE_JITTER_SPIKE_MS;
        }
        busy_wait_ms(stall);
        s.last_decode_jitter_ms = stall;
    } else {
        s.last_decode_jitter_ms = 0.0;
    }

    let observed_media = s.video.current_time();
    let duration = s.video.duration();
    let has_duration = duration.is_finite() && duration > 0.0;

    if s.video.paused() {
        let timebase = s.timebase;
        reanchor_media_clock(
            &mut s.media_clock,
            timebase,
            plan.semantic_time,
            observed_media,
        );
    } else {
        s.media_clock
            .update(plan.semantic_time.ticks(), observed_media);
    }
    let expected_media = s
        .media_clock
        .media_time_at(plan.semantic_time.ticks())
        .unwrap_or(observed_media);

    let emu_refresh_hz = if pathologies.vary_refresh {
        let idx = ((semantic_seconds / 5.0).floor() as i64).rem_euclid(2);
        if idx == 0 { 60.0 } else { 120.0 }
    } else {
        60.0
    };
    let frame_dur = 1.0 / emu_refresh_hz;

    // Intentional frame policy: if video lags too far behind target, drop
    // forward one frame; if it runs too far ahead, duplicate by pinning to
    // frame boundary.
    if !s.video.paused() {
        let policy_target = if plan.present_time.is_some() {
            target_present_seconds
        } else {
            expected_media
        };
        let delta = observed_media - policy_target;
        if delta < -(frame_dur * 0.85) {
            let mut next = observed_media + frame_dur;
            if has_duration && next >= duration {
                next = 0.0;
            }
            s.video.set_current_time(next);
            s.dropped_frames = s.dropped_frames.saturating_add(1);
        } else if delta > frame_dur {
            let mut hold = (observed_media / frame_dur).floor() * frame_dur;
            if has_duration && hold >= duration {
                hold = 0.0;
            }
            s.video.set_current_time(hold.max(0.0));
            s.duplicated_frames = s.duplicated_frames.saturating_add(1);
        }
    }

    let phase = fract(semantic_seconds * BEAT_HZ);
    let beat_idx = (semantic_seconds * BEAT_HZ).floor().max(0.0) as u64;

    if s.video.paused() {
        s.last_beat = beat_idx;
    } else if beat_idx > s.last_beat {
        s.last_beat = beat_idx;
        if let Some(audio) = s.audio.as_ref() {
            play_click(audio);
        }
    }

    let media_drift_ms = (observed_media - expected_media) * 1000.0;
    let phase_target = if plan.present_time.is_some() {
        target_present_seconds
    } else {
        expected_media
    };
    let phase_error_ms = (observed_media - phase_target) * 1000.0;

    let progress = if duration.is_finite() && duration > 0.0 {
        (observed_media / duration).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let sweep_x = phase * VIDEO_W;
    let flash_alpha = if phase < 0.1 {
        1.0 - (phase / 0.1)
    } else {
        0.1
    };

    let dial_center_x = VIDEO_W - 82.0;
    let dial_center_y = VIDEO_H - 58.0;
    let hand_angle = phase * TAU - TAU * 0.25;

    let sweep_id = s.sweep_id;
    let flash_id = s.flash_id;
    let hand_id = s.hand_id;
    let tooltip_id = s.tooltip_id;

    s.store.set_transform(
        sweep_id,
        Transform3d::from_translation(sweep_x - 1.0, 0.0, 0.0),
    );
    s.store.set_opacity(sweep_id, 1.0);

    s.store.set_transform(
        flash_id,
        Transform3d::from_translation(VIDEO_W - 56.0, 12.0, 0.0),
    );
    s.store.set_opacity(flash_id, flash_alpha as f32);

    s.store.set_transform(
        hand_id,
        Transform3d::from_translation(dial_center_x, dial_center_y, 0.0)
            * Transform3d::from_rotation_z(hand_angle)
            * Transform3d::from_translation(0.0, -2.0, 0.0),
    );
    s.store.set_opacity(hand_id, 1.0);

    s.store
        .set_transform(tooltip_id, Transform3d::from_translation(10.0, 10.0, 0.0));
    s.store.set_opacity(tooltip_id, 1.0);

    let changes = s.store.evaluate();
    let VideoState {
        ref store,
        ref mut presenter,
        ..
    } = *s;
    presenter.apply(store, &changes);

    if pathologies.gpu_stall {
        busy_wait_ms(GPU_STALL_MS);
        s.last_gpu_stall_ms = GPU_STALL_MS;
    } else {
        s.last_gpu_stall_ms = 0.0;
    }
    let submitted_at = subduction_backend_web::now();

    let feedback = PresentFeedback::new(&hints, build_start, submitted_at, None);
    s.scheduler.observe(&feedback);

    let build_ms = submitted_at.ticks().saturating_sub(build_start.ticks()) as f64 / 1000.0;
    let frame_budget_ms = frame_dur * 1000.0;
    let hard_miss =
        plan.present_time.is_some() && submitted_at.ticks() > hints.latest_commit.ticks();
    let soft_miss = build_ms > frame_budget_ms * 1.20;

    let audio_delta_ms = if let Some(audio) = s.audio.as_ref() {
        let audio_phase = fract(audio.current_time() * BEAT_HZ);
        phase_delta_ms(phase, audio_phase)
    } else {
        f64::NAN
    };

    s.ui.tooltip_element.set_text_content(Some(&format!(
        "phase err {:+.2}ms | media drift {:+.2}ms",
        phase_error_ms, media_drift_ms
    )));

    let pct = progress * 100.0;
    let _ =
        s.ui.progress_fill
            .style()
            .set_property("width", &format!("{pct:.2}%"));
    s.ui.seek.set_value(&format!("{:.0}", progress * 1000.0));

    let play_label = if s.video.paused() { "Play" } else { "Pause" };
    s.ui.play_button.set_text_content(Some(play_label));

    let confidence = confidence_label(tick.confidence);
    let ts_ms = ticks_to_secs(s.timebase, plan.semantic_time.ticks()) * 1000.0;
    let tp_label = if let Some(tp) = plan.present_time {
        format!("{:.3}ms", ticks_to_secs(s.timebase, tp.ticks()) * 1000.0)
    } else {
        "none".to_string()
    };

    let present_bucket = (phase_target * emu_refresh_hz).floor().max(0.0) as u64;
    let timecode_text = format!(
        "F {:06} | PT_BUCKET {:08} | beat {:05} | conf {}",
        tick.frame_index, present_bucket, beat_idx, confidence
    );
    s.ui.timecode.set_text_content(Some(&timecode_text));

    let av = if audio_delta_ms.is_finite() {
        format!("{:+.2}ms", audio_delta_ms)
    } else {
        "n/a".to_string()
    };

    let report = s.sync.observe(SyncSample {
        confidence: tick.confidence,
        phase_error_ms,
        hard_miss,
        soft_miss,
        frame_delta_ms,
    });
    let grade = report.grade.as_str();
    let miss_rate = report.miss_rate_per_1000;
    let color = grade_color(report.grade.as_str());

    s.ui.sync_grade.set_text_content(Some(&format!(
        "Sync Grade {} | phase error {:+.2} ms | miss rate {:.2}/1000",
        grade, phase_error_ms, miss_rate
    )));
    let _ = s.ui.sync_grade.style().set_property("color", color);

    s.ui.hud.set_text_content(Some(&format!(
        "TimingConfidence: {confidence}\nTs: {ts_ms:.3}ms\nTp: {tp_label}\npipeline depth: {}\nmissed deadlines: {}\ninj(timer/decode/gpu): {:+.2} / {:.2} / {:.2} ms",
        s.scheduler.pipeline_depth(),
        report.missed_frames,
        s.last_timer_jitter_ms,
        s.last_decode_jitter_ms,
        s.last_gpu_stall_ms
    )));

    let graph_str = s.sync.sparkline_ascii(8.0, 25.0);
    s.ui.graph.set_text_content(Some(&format!(
        "frame dt (last {GRAPH_SAMPLES}): {graph_str}"
    )));

    let status = if duration.is_finite() && duration > 0.0 {
        format!(
            "media {} / {} | media drift {:+.2} ms | A-V phase delta {} | drop {} dup {}",
            fmt_secs(observed_media),
            fmt_secs(duration),
            media_drift_ms,
            av,
            s.dropped_frames,
            s.duplicated_frames
        )
    } else {
        format!(
            "media {} | media drift {:+.2} ms | A-V phase delta {} | drop {} dup {}",
            fmt_secs(observed_media),
            media_drift_ms,
            av,
            s.dropped_frames,
            s.duplicated_frames
        )
    };
    s.ui.metrics.set_text_content(Some(&status));
}

fn ensure_audio_armed(state: &mut VideoState) {
    if state.audio.is_some() {
        return;
    }

    if let Ok(ctx) = AudioContext::new() {
        let _ = ctx.resume();
        state.audio = Some(ctx);
    }
}

fn play_click(ctx: &AudioContext) {
    let now = ctx.current_time();

    let Ok(osc) = ctx.create_oscillator() else {
        return;
    };
    let Ok(gain): Result<GainNode, _> = ctx.create_gain() else {
        return;
    };

    osc.set_type(OscillatorType::Square);
    osc.frequency().set_value(1100.0);

    gain.gain().set_value_at_time(0.0001, now).ok();
    gain.gain()
        .linear_ramp_to_value_at_time(0.25, now + 0.002)
        .ok();
    gain.gain()
        .exponential_ramp_to_value_at_time(0.0001, now + 0.04)
        .ok();

    osc.connect_with_audio_node(&gain).ok();
    gain.connect_with_audio_node(&ctx.destination()).ok();

    osc.start().ok();
    osc.stop_with_when(now + 0.045).ok();
}

fn confidence_label(c: TimingConfidence) -> &'static str {
    match c {
        TimingConfidence::Predictive => "Predictive",
        TimingConfidence::Estimated => "Estimated",
        TimingConfidence::PacingOnly => "PacingOnly",
    }
}

fn grade_color(grade: &str) -> &'static str {
    match grade {
        "A" => "#134b1b",
        "B" => "#7a5b00",
        "C" => "#7a3d00",
        _ => "#7f1a1a",
    }
}

fn rand_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn rand_range(state: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rand_unit(state)
}

fn busy_wait_ms(ms: f64) {
    if ms <= 0.0 {
        return;
    }
    let start = subduction_backend_web::now().ticks();
    let target = start.saturating_add((ms * 1000.0) as u64);
    while subduction_backend_web::now().ticks() < target {}
}

fn phase_delta_ms(a_phase: f64, b_phase: f64) -> f64 {
    let wrapped = (a_phase - b_phase + 0.5).rem_euclid(1.0) - 0.5;
    wrapped * (1000.0 / BEAT_HZ)
}

fn fract(v: f64) -> f64 {
    v.rem_euclid(1.0)
}

fn element(doc: &Document, tag: &str) -> Result<HtmlElement, JsValue> {
    Ok(doc.create_element(tag)?.unchecked_into())
}

fn add_checkbox(
    doc: &Document,
    host: &HtmlElement,
    label: &str,
    title: &str,
    checked: bool,
) -> Result<HtmlInputElement, JsValue> {
    let row = element(doc, "label")?;
    style(&row, "display: inline-flex; gap: 8px; align-items: center;")?;
    row.set_attribute("title", title)?;
    let input: HtmlInputElement = doc.create_element("input")?.unchecked_into();
    input.set_type("checkbox");
    input.set_checked(checked);
    let text = element(doc, "span")?;
    text.set_text_content(Some(label));
    row.append_child(&input)?;
    row.append_child(&text)?;
    host.append_child(&row)?;
    Ok(input)
}

fn create_shell(doc: &Document) -> Result<HtmlElement, JsValue> {
    let shell = element(doc, "section")?;
    style(
        &shell,
        "width: 960px; padding: 24px 28px 20px; border-radius: 20px; background: rgba(255,255,255,0.82); border: 1px solid rgba(22,44,65,0.15); box-shadow: 0 24px 70px rgba(26,43,64,0.2); display: grid; gap: 14px; justify-items: center;",
    )?;
    Ok(shell)
}

fn style(el: &web_sys::Element, css: &str) -> Result<(), JsValue> {
    el.set_attribute("style", css)
}

fn ticks_to_secs(timebase: Timebase, ticks: u64) -> f64 {
    timebase.ticks_to_nanos(ticks) as f64 / 1e9
}

fn fmt_secs(seconds: f64) -> String {
    let clamped = if seconds.is_finite() && seconds >= 0.0 {
        seconds
    } else {
        0.0
    };
    let mins = (clamped / 60.0).floor() as u64;
    let secs = clamped - mins as f64 * 60.0;
    format!("{mins:02}:{secs:06.3}")
}
