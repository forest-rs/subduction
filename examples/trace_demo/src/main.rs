// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simulated frame loop that exercises the tracing and diagnostics pipeline.
//!
//! Runs 60 synthetic frames through the scheduler, recording events to both a
//! [`PrettyPrintSink`] and a
//! [`RecorderSink`], then exports a
//! Chrome trace JSON file.

use std::fs::File;
use std::io::BufWriter;

use subduction_core::output::OutputId;
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::{HostTime, Timebase};
use subduction_core::timing::{FrameTick, PresentFeedback, PresentHints, TimingConfidence};
use subduction_core::trace::{
    FramePlanEvent, FrameSummaryBuilder, FrameTickEvent, PhaseBeginEvent, PhaseEndEvent, PhaseKind,
    PresentFeedbackEvent, SubmitEvent, TraceSink, Tracer,
};

use subduction_debug::pretty::PrettyPrintSink;
use subduction_debug::recorder::RecorderSink;

const FRAME_COUNT: u64 = 60;
/// 16.6ms refresh interval in nanoseconds (≈60 Hz).
const REFRESH_INTERVAL_NS: u64 = 16_666_667;

fn main() {
    let timebase = Timebase::NANOS;
    let refresh_interval = REFRESH_INTERVAL_NS;

    // -- sinks -------------------------------------------------------------
    let mut pretty = PrettyPrintSink::new(Box::new(std::io::stdout()), timebase);
    let mut recorder = RecorderSink::new();

    // -- scheduler ---------------------------------------------------------
    let config = SchedulerConfig::macos();
    let mut scheduler = Scheduler::new(config);

    // -- simulated loop ----------------------------------------------------
    let mut now_ticks: u64 = 1_000_000_000; // start at 1s

    for frame_index in 0..FRAME_COUNT {
        // 1. Tick
        let tick = FrameTick {
            now: HostTime(now_ticks),
            predicted_present: Some(HostTime(now_ticks + refresh_interval)),
            refresh_interval: Some(refresh_interval),
            confidence: TimingConfidence::Predictive,
            frame_index,
            output: OutputId(0),
            prev_actual_present: if frame_index > 0 {
                // Previous frame presented on time.
                Some(HostTime(now_ticks - refresh_interval))
            } else {
                None
            },
        };

        let tick_event = FrameTickEvent::from(&tick);
        pretty.on_frame_tick(&tick_event);
        recorder.on_frame_tick(&tick_event);

        // 2. Hints + plan
        let hints = PresentHints {
            desired_present: tick.predicted_present,
            latest_commit: HostTime(now_ticks + refresh_interval - 2_000_000),
        };

        let plan_start = HostTime(now_ticks + 50_000);
        emit_phase_begin(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Plan,
            plan_start,
        );

        let plan = scheduler.plan(&tick, &hints);
        let plan_end = HostTime(now_ticks + 100_000);

        let plan_event = FramePlanEvent::new(&plan, scheduler.safety_margin_ticks());
        pretty.on_frame_plan(&plan_event);
        recorder.on_frame_plan(&plan_event);

        emit_phase_end(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Plan,
            plan_end,
        );

        // 3. Evaluate (simulated)
        let eval_start = plan_end;
        emit_phase_begin(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Evaluate,
            eval_start,
        );
        let eval_end = HostTime(eval_start.ticks() + 500_000);
        emit_phase_end(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Evaluate,
            eval_end,
        );

        // 4. Render (simulated)
        let render_start = eval_end;
        emit_phase_begin(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Render,
            render_start,
        );
        let render_end = HostTime(render_start.ticks() + 2_000_000);
        emit_phase_end(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Render,
            render_end,
        );

        // 5. Submit
        let submit_start = render_end;
        emit_phase_begin(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Submit,
            submit_start,
        );
        let submit_end = HostTime(submit_start.ticks() + 100_000);
        emit_phase_end(
            &mut pretty,
            &mut recorder,
            frame_index,
            PhaseKind::Submit,
            submit_end,
        );

        let submit_event = SubmitEvent {
            frame_index,
            submitted_at: submit_end,
            expected_present: plan.present_time,
        };
        pretty.on_submit(&submit_event);
        recorder.on_submit(&submit_event);

        // 6. Feedback
        let missed = submit_end > plan.commit_deadline;
        let feedback = PresentFeedback {
            submitted_at: submit_end,
            build_start: plan_start,
            expected_present: plan.present_time,
            actual_present: plan.present_time,
            missed_deadline: Some(missed),
            pacing_overrun: None,
        };
        scheduler.observe(&feedback);

        let feedback_event = PresentFeedbackEvent {
            frame_index,
            actual_present: plan.present_time,
            missed_deadline: Some(missed),
        };
        pretty.on_present_feedback(&feedback_event);
        recorder.on_present_feedback(&feedback_event);

        // 7. Summary
        let mut builder = FrameSummaryBuilder::new(&tick_event, &plan_event);
        builder.phase_begin(PhaseKind::Plan, plan_start);
        builder.phase_end(PhaseKind::Plan, plan_end);
        builder.phase_begin(PhaseKind::Evaluate, eval_start);
        builder.phase_end(PhaseKind::Evaluate, eval_end);
        builder.phase_begin(PhaseKind::Render, render_start);
        builder.phase_end(PhaseKind::Render, render_end);
        builder.phase_begin(PhaseKind::Submit, submit_start);
        builder.phase_end(PhaseKind::Submit, submit_end);
        builder.set_missed_deadline(missed);
        let summary = builder.finish();

        pretty.on_frame_summary(&summary);
        recorder.on_frame_summary(&summary);

        // Also exercise Tracer wrapper (just to prove it compiles and dispatches).
        if frame_index == 0 {
            let mut tracer = Tracer::new(&mut pretty);
            tracer.frame_tick(&tick_event);
        }

        // Advance time.
        now_ticks += refresh_interval;
    }

    // -- export Chrome trace -----------------------------------------------
    let path = "trace.json";
    let file = File::create(path).expect("failed to create trace.json");
    let mut writer = BufWriter::new(file);
    subduction_debug::chrome::export(recorder.as_bytes(), timebase, &mut writer)
        .expect("failed to write Chrome trace");

    println!("Wrote {path} ({FRAME_COUNT} frames)");
}

fn emit_phase_begin(
    pretty: &mut PrettyPrintSink,
    recorder: &mut RecorderSink,
    frame_index: u64,
    phase: PhaseKind,
    timestamp: HostTime,
) {
    let e = PhaseBeginEvent {
        frame_index,
        phase,
        timestamp,
    };
    pretty.on_phase_begin(&e);
    recorder.on_phase_begin(&e);
}

fn emit_phase_end(
    pretty: &mut PrettyPrintSink,
    recorder: &mut RecorderSink,
    frame_index: u64,
    phase: PhaseKind,
    timestamp: HostTime,
) {
    let e = PhaseEndEvent {
        frame_index,
        phase,
        timestamp,
    };
    pretty.on_phase_end(&e);
    recorder.on_phase_end(&e);
}
