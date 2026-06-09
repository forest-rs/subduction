// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic frameclock scheduler and diagnostics example.
//!
//! This example drives [`frameclock`] without a window, renderer, or Subduction
//! layer tree. It demonstrates the core loop: build a tick, plan the frame,
//! emit diagnostics, submit synthetic work, and feed presentation feedback back
//! into the scheduler.

use frameclock::{
    Diagnostics, DiagnosticsSink, Duration, FramePlanEvent, FrameTick, FrameTickEvent, HostTime,
    OutputId, PresentFeedback, PresentFeedbackEvent, PresentHints, Scheduler, SchedulerConfig,
    SchedulerStateEvent, SubmitEvent, TimingConfidence,
};

const FRAME_COUNT: u64 = 90;
const REFRESH_INTERVAL: Duration = Duration(16_666_667);
const SAFETY_MARGIN: Duration = Duration(2_000_000);
const START_TIME: HostTime = HostTime(1_000_000_000);

#[derive(Debug, Default)]
struct SummarySink {
    ticks: u64,
    plans: u64,
    submits: u64,
    feedback: u64,
    misses: u64,
    overruns: u64,
    final_depth: u8,
    final_safety_margin: u64,
}

impl DiagnosticsSink for SummarySink {
    fn on_frame_tick(&mut self, _event: &FrameTickEvent) {
        self.ticks += 1;
    }

    fn on_frame_plan(&mut self, _event: &FramePlanEvent) {
        self.plans += 1;
    }

    fn on_submit(&mut self, _event: &SubmitEvent) {
        self.submits += 1;
    }

    fn on_present_feedback(&mut self, event: &PresentFeedbackEvent) {
        self.feedback += 1;
        if event.missed_deadline == Some(true) {
            self.misses += 1;
        }
        if event.pacing_overrun == Some(true) {
            self.overruns += 1;
        }
    }

    fn on_scheduler_state(&mut self, event: &SchedulerStateEvent) {
        self.final_depth = event.state.pipeline_depth;
        self.final_safety_margin = event.state.safety_margin_ticks;
    }
}

fn main() {
    let mut scheduler = Scheduler::new(SchedulerConfig::predictive());
    let mut sink = SummarySink::default();

    {
        let mut diagnostics = Diagnostics::new(&mut sink);
        let output = OutputId(0);

        for frame_index in 0..FRAME_COUNT {
            let now = START_TIME + Duration(frame_index * REFRESH_INTERVAL.ticks());
            let predicted_present = now + REFRESH_INTERVAL;
            let tick = FrameTick {
                now,
                predicted_present: Some(predicted_present),
                refresh_interval: Some(REFRESH_INTERVAL.ticks()),
                confidence: TimingConfidence::Predictive,
                frame_index,
                output,
                prev_actual_present: if frame_index > 0 { Some(now) } else { None },
            };
            diagnostics.frame_tick(&FrameTickEvent::from(&tick));

            let hints = PresentHints {
                desired_present: tick.predicted_present,
                latest_commit: predicted_present
                    .checked_sub(SAFETY_MARGIN)
                    .unwrap_or(tick.now),
            };

            let plan = scheduler.plan(&tick, &hints);
            diagnostics.frame_plan(&FramePlanEvent::new(&plan, scheduler.safety_margin_ticks()));

            let build_start = tick.now + Duration(250_000);
            let build_cost = if frame_index % 23 == 11 {
                Duration(16_000_000)
            } else {
                Duration(850_000)
            };
            let submitted_at = build_start + build_cost;
            let feedback = PresentFeedback::new(&hints, build_start, submitted_at, None);

            diagnostics.submit(&SubmitEvent {
                frame_index,
                submitted_at,
                expected_present: feedback.expected_present,
            });
            diagnostics.present_feedback(&PresentFeedbackEvent::new(frame_index, &feedback));

            scheduler.observe(&feedback);
            diagnostics.scheduler_state(&SchedulerStateEvent {
                state: scheduler.state(),
            });
        }
    }

    println!("frameclock simulated run");
    println!("frames: {}", sink.ticks);
    println!("plans: {}", sink.plans);
    println!("submits: {}", sink.submits);
    println!("feedback events: {}", sink.feedback);
    println!("misses: {}", sink.misses);
    println!("pacing overruns: {}", sink.overruns);
    println!("final pipeline depth: {}", sink.final_depth);
    println!("final safety margin ticks: {}", sink.final_safety_margin);
}
