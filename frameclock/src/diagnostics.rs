// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Diagnostics hooks for frame timing and scheduling.
//!
//! The diagnostics surface is intentionally sink-oriented and backend-neutral.
//! `frameclock` does not depend on Tracy, Spoor, or any runtime collector.
//! Adapter crates can implement [`DiagnosticsSink`] and map these events into
//! their preferred instrumentation system.

use crate::output::OutputId;
use crate::scheduler::SchedulerState;
use crate::time::{Duration, HostTime};
use crate::timing::{FrameDemand, FramePlan, FrameTick, PresentFeedback, TimingConfidence};

/// Emitted when a platform adapter receives or constructs a display-frame tick.
#[derive(Clone, Copy, Debug)]
pub struct FrameTickEvent {
    /// Monotonic frame counter.
    pub frame_index: u64,
    /// Target output for this tick.
    pub output: OutputId,
    /// Host time when the tick was generated or received.
    pub now: HostTime,
    /// Predicted presentation time, if the platform exposes one.
    pub predicted_present: Option<HostTime>,
    /// Refresh interval in host-time ticks, if known.
    pub refresh_interval: Option<u64>,
    /// Timing confidence for this tick.
    pub confidence: TimingConfidence,
}

impl From<&FrameTick> for FrameTickEvent {
    fn from(tick: &FrameTick) -> Self {
        Self {
            frame_index: tick.frame_index,
            output: tick.output,
            now: tick.now,
            predicted_present: tick.predicted_present,
            refresh_interval: tick.refresh_interval,
            confidence: tick.confidence,
        }
    }
}

/// Emitted after the scheduler produces a [`FramePlan`].
#[derive(Clone, Copy, Debug)]
pub struct FramePlanEvent {
    /// Monotonic frame counter.
    pub frame_index: u64,
    /// Target output for this plan.
    pub output: OutputId,
    /// Demand that selected this frame.
    pub demand: FrameDemand,
    /// Scheduler-selected delivery interval for this frame.
    pub frame_interval: Duration,
    /// Time applications should wake or start app-side frame work.
    pub frame_start: HostTime,
    /// Time applications should sample animations and simulation state for.
    pub sample_time: HostTime,
    /// Intended presentation time, if known.
    pub target_present: Option<HostTime>,
    /// Latest known commit/submission deadline.
    pub commit_deadline: HostTime,
    /// Current scheduler pipeline depth.
    pub pipeline_depth: u8,
    /// Current scheduler safety margin in host-time ticks.
    pub safety_margin_ticks: u64,
}

impl FramePlanEvent {
    /// Creates a diagnostics event from a [`FramePlan`] and safety margin.
    #[must_use]
    pub fn new(plan: &FramePlan, safety_margin_ticks: u64) -> Self {
        Self {
            frame_index: plan.frame_index,
            output: plan.output,
            demand: plan.demand,
            frame_interval: plan.frame_interval,
            frame_start: plan.frame_start,
            sample_time: plan.sample_time,
            target_present: plan.target_present,
            commit_deadline: plan.commit_deadline,
            pipeline_depth: plan.pipeline_depth,
            safety_margin_ticks,
        }
    }
}

/// Emitted when a frame is submitted to the display pipeline.
#[derive(Clone, Copy, Debug)]
pub struct SubmitEvent {
    /// Monotonic frame counter.
    pub frame_index: u64,
    /// Host time of submission.
    pub submitted_at: HostTime,
    /// Expected presentation time at submission, if known.
    pub expected_present: Option<HostTime>,
}

/// Emitted when presentation feedback is resolved.
#[derive(Clone, Copy, Debug)]
pub struct PresentFeedbackEvent {
    /// Monotonic frame counter.
    pub frame_index: u64,
    /// Actual presentation time, if reported by the platform.
    pub actual_present: Option<HostTime>,
    /// Whether the frame missed a real presentation deadline, if determinable.
    pub missed_deadline: Option<bool>,
    /// Whether frame building overran the pacing boundary exposed by the
    /// platform, if determinable.
    pub pacing_overrun: Option<bool>,
}

impl PresentFeedbackEvent {
    /// Creates a feedback diagnostics event.
    #[must_use]
    pub fn new(frame_index: u64, feedback: &PresentFeedback) -> Self {
        Self {
            frame_index,
            actual_present: feedback.actual_present,
            missed_deadline: feedback.missed_deadline,
            pacing_overrun: feedback.pacing_overrun,
        }
    }
}

/// Emitted when scheduler adaptation state is sampled.
#[derive(Clone, Copy, Debug)]
pub struct SchedulerStateEvent {
    /// Current scheduler state.
    pub state: SchedulerState,
}

/// Per-frame summary of frameclock-owned timing and pacing facts.
///
/// This is a compact aggregate for logs, counters, trace rows, and diagnostics
/// UIs that do not need a host-specific frame-loop trace. It intentionally does
/// not include app/render phases, layer changes, damage rectangles, or other
/// renderer-owned concepts.
#[derive(Clone, Copy, Debug)]
pub struct FrameTimingSummary {
    /// Monotonic frame counter.
    pub frame_index: u64,
    /// Target output for this frame.
    pub output: OutputId,
    /// Timing confidence for the originating frame tick.
    pub confidence: TimingConfidence,
    /// Host time when the frame tick was generated or received.
    pub tick_time: HostTime,
    /// Predicted presentation time from the originating tick, if known.
    pub predicted_present: Option<HostTime>,
    /// Refresh interval from the originating tick, if known.
    pub refresh_interval: Option<u64>,
    /// Demand that selected this frame.
    pub demand: FrameDemand,
    /// Scheduler-selected delivery interval for this frame.
    pub frame_interval: Duration,
    /// Time applications should wake or start app-side frame work.
    pub frame_start: HostTime,
    /// Time applications should sample animations and simulation state for.
    pub sample_time: HostTime,
    /// Intended presentation time, if known.
    pub target_present: Option<HostTime>,
    /// Latest known commit/submission deadline.
    pub commit_deadline: HostTime,
    /// Host time when the frame was submitted, if recorded.
    pub submitted_at: Option<HostTime>,
    /// Expected presentation time at submission, if recorded.
    pub expected_present: Option<HostTime>,
    /// Actual presentation time, if reported by the platform.
    pub actual_present: Option<HostTime>,
    /// Whether the frame missed a real presentation deadline, if determinable.
    pub missed_deadline: Option<bool>,
    /// Whether frame building overran a pacing boundary, if determinable.
    pub pacing_overrun: Option<bool>,
    /// Scheduler pipeline depth used for the plan.
    pub pipeline_depth: u8,
    /// Scheduler safety margin used for the plan, in host-time ticks.
    pub safety_margin_ticks: u64,
    /// Scheduler adaptation state sampled for this frame, if recorded.
    pub scheduler_state: Option<SchedulerState>,
}

/// Builds a [`FrameTimingSummary`] from frameclock diagnostics events.
///
/// Create one builder per planned frame, feed it the events observed for that
/// frame, then call [`finish`](Self::finish). Event order is not significant.
/// The builder returns `None` if the required tick and plan are missing or if
/// they do not describe the same frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimingSummaryBuilder {
    tick: Option<FrameTickEvent>,
    plan: Option<FramePlanEvent>,
    submit: Option<SubmitEvent>,
    feedback: Option<PresentFeedbackEvent>,
    scheduler_state: Option<SchedulerStateEvent>,
}

impl FrameTimingSummaryBuilder {
    /// Creates an empty summary builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tick: None,
            plan: None,
            submit: None,
            feedback: None,
            scheduler_state: None,
        }
    }

    /// Starts a builder with a tick and plan.
    #[must_use]
    pub const fn from_tick_and_plan(tick: &FrameTickEvent, plan: &FramePlanEvent) -> Self {
        Self {
            tick: Some(*tick),
            plan: Some(*plan),
            submit: None,
            feedback: None,
            scheduler_state: None,
        }
    }

    /// Records a frame tick event.
    pub fn record_frame_tick(&mut self, event: &FrameTickEvent) -> &mut Self {
        self.tick = Some(*event);
        self
    }

    /// Records a frame plan event.
    pub fn record_frame_plan(&mut self, event: &FramePlanEvent) -> &mut Self {
        self.plan = Some(*event);
        self
    }

    /// Records a submit event.
    pub fn record_submit(&mut self, event: &SubmitEvent) -> &mut Self {
        self.submit = Some(*event);
        self
    }

    /// Records a presentation feedback event.
    pub fn record_present_feedback(&mut self, event: &PresentFeedbackEvent) -> &mut Self {
        self.feedback = Some(*event);
        self
    }

    /// Records a scheduler state event.
    pub fn record_scheduler_state(&mut self, event: &SchedulerStateEvent) -> &mut Self {
        self.scheduler_state = Some(*event);
        self
    }

    /// Produces a frame timing summary.
    ///
    /// Returns `None` when the required tick and plan are missing or refer to
    /// different frames. Optional submit and feedback events are included only
    /// when their frame index matches the plan.
    #[must_use]
    pub fn finish(self) -> Option<FrameTimingSummary> {
        let tick = self.tick?;
        let plan = self.plan?;
        if tick.frame_index != plan.frame_index || tick.output != plan.output {
            return None;
        }

        let submit = self
            .submit
            .filter(|submit| submit.frame_index == plan.frame_index);
        let feedback = self
            .feedback
            .filter(|feedback| feedback.frame_index == plan.frame_index);
        let scheduler_state = self.scheduler_state.map(|event| event.state);

        Some(FrameTimingSummary {
            frame_index: plan.frame_index,
            output: plan.output,
            confidence: tick.confidence,
            tick_time: tick.now,
            predicted_present: tick.predicted_present,
            refresh_interval: tick.refresh_interval,
            demand: plan.demand,
            frame_interval: plan.frame_interval,
            frame_start: plan.frame_start,
            sample_time: plan.sample_time,
            target_present: plan.target_present,
            commit_deadline: plan.commit_deadline,
            submitted_at: submit.map(|submit| submit.submitted_at),
            expected_present: submit.and_then(|submit| submit.expected_present),
            actual_present: feedback.and_then(|feedback| feedback.actual_present),
            missed_deadline: feedback.and_then(|feedback| feedback.missed_deadline),
            pacing_overrun: feedback.and_then(|feedback| feedback.pacing_overrun),
            pipeline_depth: plan.pipeline_depth,
            safety_margin_ticks: plan.safety_margin_ticks,
            scheduler_state,
        })
    }
}

/// Receives frameclock diagnostics events.
///
/// All methods have default no-op implementations so sinks can implement only
/// the events they need.
pub trait DiagnosticsSink {
    /// Called when a display-frame tick is available.
    fn on_frame_tick(&mut self, event: &FrameTickEvent) {
        _ = event;
    }

    /// Called when a frame plan is produced.
    fn on_frame_plan(&mut self, event: &FramePlanEvent) {
        _ = event;
    }

    /// Called when a frame is submitted.
    fn on_submit(&mut self, event: &SubmitEvent) {
        _ = event;
    }

    /// Called when presentation feedback is resolved.
    fn on_present_feedback(&mut self, event: &PresentFeedbackEvent) {
        _ = event;
    }

    /// Called when scheduler adaptation state is sampled.
    fn on_scheduler_state(&mut self, event: &SchedulerStateEvent) {
        _ = event;
    }

    /// Called when a per-frame timing summary is available.
    fn on_frame_timing_summary(&mut self, event: &FrameTimingSummary) {
        _ = event;
    }
}

impl DiagnosticsSink for FrameTimingSummaryBuilder {
    fn on_frame_tick(&mut self, event: &FrameTickEvent) {
        self.record_frame_tick(event);
    }

    fn on_frame_plan(&mut self, event: &FramePlanEvent) {
        self.record_frame_plan(event);
    }

    fn on_submit(&mut self, event: &SubmitEvent) {
        self.record_submit(event);
    }

    fn on_present_feedback(&mut self, event: &PresentFeedbackEvent) {
        self.record_present_feedback(event);
    }

    fn on_scheduler_state(&mut self, event: &SchedulerStateEvent) {
        self.record_scheduler_state(event);
    }
}

/// A diagnostics sink that discards all events.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopDiagnostics;

impl DiagnosticsSink for NoopDiagnostics {}

/// Thin wrapper around an optional diagnostics sink.
pub struct Diagnostics<'a> {
    sink: Option<&'a mut dyn DiagnosticsSink>,
}

impl core::fmt::Debug for Diagnostics<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Diagnostics").finish_non_exhaustive()
    }
}

impl<'a> Diagnostics<'a> {
    /// Creates diagnostics that dispatch to `sink`.
    #[must_use]
    pub fn new(sink: &'a mut dyn DiagnosticsSink) -> Self {
        Self { sink: Some(sink) }
    }

    /// Creates diagnostics that discard all events.
    #[must_use]
    pub const fn none() -> Self {
        Self { sink: None }
    }

    /// Emits a frame tick event.
    pub fn frame_tick(&mut self, event: &FrameTickEvent) {
        if let Some(sink) = &mut self.sink {
            sink.on_frame_tick(event);
        }
    }

    /// Emits a frame plan event.
    pub fn frame_plan(&mut self, event: &FramePlanEvent) {
        if let Some(sink) = &mut self.sink {
            sink.on_frame_plan(event);
        }
    }

    /// Emits a submit event.
    pub fn submit(&mut self, event: &SubmitEvent) {
        if let Some(sink) = &mut self.sink {
            sink.on_submit(event);
        }
    }

    /// Emits a presentation feedback event.
    pub fn present_feedback(&mut self, event: &PresentFeedbackEvent) {
        if let Some(sink) = &mut self.sink {
            sink.on_present_feedback(event);
        }
    }

    /// Emits a scheduler state event.
    pub fn scheduler_state(&mut self, event: &SchedulerStateEvent) {
        if let Some(sink) = &mut self.sink {
            sink.on_scheduler_state(event);
        }
    }

    /// Emits a frame timing summary.
    pub fn frame_timing_summary(&mut self, event: &FrameTimingSummary) {
        if let Some(sink) = &mut self.sink {
            sink.on_frame_timing_summary(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tick() -> FrameTickEvent {
        FrameTickEvent {
            frame_index: 7,
            output: OutputId(2),
            now: HostTime(1_000),
            predicted_present: Some(HostTime(2_000)),
            refresh_interval: Some(16_666_667),
            confidence: TimingConfidence::Predictive,
        }
    }

    fn sample_plan() -> FramePlanEvent {
        FramePlanEvent {
            frame_index: 7,
            output: OutputId(2),
            demand: FrameDemand::ANIMATION,
            frame_interval: Duration(16_666_667),
            frame_start: HostTime(1_500),
            sample_time: HostTime(2_000),
            target_present: Some(HostTime(2_000)),
            commit_deadline: HostTime(1_900),
            pipeline_depth: 1,
            safety_margin_ticks: 500,
        }
    }

    #[test]
    fn timing_summary_requires_tick_and_plan() {
        assert!(FrameTimingSummaryBuilder::new().finish().is_none());

        let mut builder = FrameTimingSummaryBuilder::new();
        builder.record_frame_tick(&sample_tick());
        assert!(builder.finish().is_none());
    }

    #[test]
    fn timing_summary_rejects_mismatched_tick_and_plan() {
        let mut tick = sample_tick();
        tick.frame_index = 8;

        let summary = FrameTimingSummaryBuilder::from_tick_and_plan(&tick, &sample_plan()).finish();

        assert!(summary.is_none());
    }

    #[test]
    fn timing_summary_aggregates_matching_events() {
        let tick = sample_tick();
        let plan = sample_plan();
        let submit = SubmitEvent {
            frame_index: 7,
            submitted_at: HostTime(1_850),
            expected_present: Some(HostTime(2_000)),
        };
        let feedback = PresentFeedbackEvent {
            frame_index: 7,
            actual_present: Some(HostTime(2_050)),
            missed_deadline: Some(true),
            pacing_overrun: Some(false),
        };
        let state = SchedulerStateEvent {
            state: SchedulerState {
                pipeline_depth: 1,
                safety_margin_ticks: 600,
                consecutive_misses: 1,
                consecutive_hits: 0,
            },
        };

        let mut builder = FrameTimingSummaryBuilder::from_tick_and_plan(&tick, &plan);
        builder
            .record_submit(&submit)
            .record_present_feedback(&feedback)
            .record_scheduler_state(&state);
        let summary = builder
            .finish()
            .expect("matching tick and plan should produce a summary");

        assert_eq!(summary.frame_index, 7);
        assert_eq!(summary.output, OutputId(2));
        assert_eq!(summary.tick_time, HostTime(1_000));
        assert_eq!(summary.submitted_at, Some(HostTime(1_850)));
        assert_eq!(summary.actual_present, Some(HostTime(2_050)));
        assert_eq!(summary.missed_deadline, Some(true));
        assert_eq!(
            summary
                .scheduler_state
                .map(|state| state.safety_margin_ticks),
            Some(600)
        );
    }

    #[test]
    fn timing_summary_ignores_mismatched_optional_events() {
        let mismatched_submit = SubmitEvent {
            frame_index: 8,
            submitted_at: HostTime(1_850),
            expected_present: Some(HostTime(2_000)),
        };
        let mismatched_feedback = PresentFeedbackEvent {
            frame_index: 9,
            actual_present: Some(HostTime(2_050)),
            missed_deadline: Some(true),
            pacing_overrun: Some(true),
        };

        let mut builder =
            FrameTimingSummaryBuilder::from_tick_and_plan(&sample_tick(), &sample_plan());
        builder
            .record_submit(&mismatched_submit)
            .record_present_feedback(&mismatched_feedback);
        let summary = builder
            .finish()
            .expect("mismatched optional events should not reject the frame");

        assert_eq!(summary.submitted_at, None);
        assert_eq!(summary.actual_present, None);
        assert_eq!(summary.missed_deadline, None);
        assert_eq!(summary.pacing_overrun, None);
    }
}
