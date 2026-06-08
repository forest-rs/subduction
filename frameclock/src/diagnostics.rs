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
use crate::time::HostTime;
use crate::timing::{FramePlan, FrameTick, PresentFeedback, TimingConfidence};

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
}
