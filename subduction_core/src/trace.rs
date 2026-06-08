// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tracing and diagnostics for the frame loop.
//!
//! This module provides a [`TraceSink`] trait with per-event methods that
//! frame-loop instrumentation calls at each stage. All method bodies default to
//! no-ops, so implementing only the events you care about is fine.
//!
//! [`Tracer`] wraps an optional `&mut dyn TraceSink`. When the `trace` feature
//! is **off**, every `Tracer` method compiles to nothing (zero overhead). When
//! **on**, each method performs a single `Option` branch before dispatching.
//!
//! [`FrameSummaryBuilder`] is a convenience helper that collects phase
//! timestamps during a frame and produces a [`FrameSummary`] at the end.
//!
//! # Crate features
//!
//! - `trace` — enables the `Tracer` method bodies (one branch per call).
//! - `trace-rich` (implies `trace`) — gates [`LayerChange`] and [`DamageRect`]
//!   events plus the corresponding `TraceSink` methods.

use crate::output::OutputId;
use crate::time::HostTime;
use crate::timing::{FramePlan, FrameTick, TimingConfidence};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Which phase of the frame loop is being measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseKind {
    /// Scene planning (scheduler → frame plan).
    Plan,
    /// Layer tree evaluation (dirty propagation, world transforms).
    Evaluate,
    /// Backend rendering / compositing.
    Render,
    /// Submitting the frame to the display pipeline.
    Submit,
}

/// Which property of a layer changed.
#[cfg(feature = "trace-rich")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerField {
    /// Local or world transform.
    Transform,
    /// Opacity value.
    Opacity,
    /// Clip region.
    Clip,
    /// Content (surface, texture, etc.).
    Content,
    /// Layer flags.
    Flags,
    /// Topology (parent/child relationships).
    Topology,
}

// ---------------------------------------------------------------------------
// Event structs
// ---------------------------------------------------------------------------

/// Emitted when the backend delivers a display-link tick.
#[derive(Clone, Copy, Debug)]
pub struct FrameTickEvent {
    /// Monotonic frame counter.
    pub frame_index: u64,
    /// Which output this tick targets.
    pub output: OutputId,
    /// Host time when the tick was generated.
    pub now: HostTime,
    /// Predicted present time, if known.
    pub predicted_present: Option<HostTime>,
    /// Refresh interval in ticks, if known.
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

/// Emitted after the scheduler produces a frame plan.
#[derive(Clone, Copy, Debug)]
pub struct FramePlanEvent {
    /// Frame counter.
    pub frame_index: u64,
    /// Which output this plan targets.
    pub output: OutputId,
    /// Semantic (animation) time.
    pub sample_time: HostTime,
    /// Intended display time, if known.
    pub target_present: Option<HostTime>,
    /// Latest commit time.
    pub commit_deadline: HostTime,
    /// Current pipeline depth.
    pub pipeline_depth: u8,
    /// Current scheduler safety margin in ticks.
    pub safety_margin_ticks: u64,
}

impl FramePlanEvent {
    /// Creates a `FramePlanEvent` from a [`FramePlan`] plus the scheduler's
    /// current safety margin (which the plan itself does not carry).
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

/// Marks the beginning of a frame-loop phase.
#[derive(Clone, Copy, Debug)]
pub struct PhaseBeginEvent {
    /// Frame counter.
    pub frame_index: u64,
    /// Which phase is starting.
    pub phase: PhaseKind,
    /// Host time at the start of the phase.
    pub timestamp: HostTime,
}

/// Marks the end of a frame-loop phase.
#[derive(Clone, Copy, Debug)]
pub struct PhaseEndEvent {
    /// Frame counter.
    pub frame_index: u64,
    /// Which phase is ending.
    pub phase: PhaseKind,
    /// Host time at the end of the phase.
    pub timestamp: HostTime,
}

/// Emitted when a frame is submitted to the display pipeline.
#[derive(Clone, Copy, Debug)]
pub struct SubmitEvent {
    /// Frame counter.
    pub frame_index: u64,
    /// Host time of submission.
    pub submitted_at: HostTime,
    /// Expected present time at submission, if known.
    pub expected_present: Option<HostTime>,
}

/// Emitted when actual presentation feedback arrives.
#[derive(Clone, Copy, Debug)]
pub struct PresentFeedbackEvent {
    /// Frame counter.
    pub frame_index: u64,
    /// Actual present time, if reported by the platform.
    pub actual_present: Option<HostTime>,
    /// Whether the deadline was missed, if determinable.
    pub missed_deadline: Option<bool>,
}

/// Per-frame timing summary produced by [`FrameSummaryBuilder`].
#[derive(Clone, Copy, Debug)]
pub struct FrameSummary {
    /// Frame counter.
    pub frame_index: u64,
    /// Which output.
    pub output: OutputId,
    /// Timing confidence.
    pub confidence: TimingConfidence,
    /// Host time when the tick was generated.
    pub now: HostTime,
    /// Intended present time, if known.
    pub target_present: Option<HostTime>,
    /// Semantic (animation) time.
    pub sample_time: HostTime,
    /// Commit deadline.
    pub deadline: HostTime,
    /// Pipeline depth for this frame.
    pub pipeline_depth: u8,
    /// Plan phase duration in ticks (0 if not measured).
    pub plan_ticks: u64,
    /// Evaluate phase duration in ticks (0 if not measured).
    pub eval_ticks: u64,
    /// Render phase duration in ticks (0 if not measured).
    pub render_ticks: u64,
    /// Submit phase duration in ticks (0 if not measured).
    pub submit_ticks: u64,
    /// Whether the deadline was missed.
    pub missed_deadline: bool,
}

/// A per-frame layer change record.
#[cfg(feature = "trace-rich")]
#[derive(Clone, Copy, Debug)]
pub struct LayerChange {
    /// Index of the layer that changed.
    pub layer_index: u32,
    /// Which field changed.
    pub field: LayerField,
}

/// An axis-aligned damage rectangle.
#[cfg(feature = "trace-rich")]
#[derive(Clone, Copy, Debug)]
pub struct DamageRect {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

// ---------------------------------------------------------------------------
// TraceSink trait
// ---------------------------------------------------------------------------

/// Receives trace events from the frame loop.
///
/// All methods have default no-op implementations, so you only need to
/// override the events you care about.
pub trait TraceSink {
    /// Called when a display-link tick is received.
    fn on_frame_tick(&mut self, e: &FrameTickEvent) {
        _ = e;
    }

    /// Called after the scheduler produces a frame plan.
    fn on_frame_plan(&mut self, e: &FramePlanEvent) {
        _ = e;
    }

    /// Called at the beginning of a frame-loop phase.
    fn on_phase_begin(&mut self, e: &PhaseBeginEvent) {
        _ = e;
    }

    /// Called at the end of a frame-loop phase.
    fn on_phase_end(&mut self, e: &PhaseEndEvent) {
        _ = e;
    }

    /// Called when a frame is submitted.
    fn on_submit(&mut self, e: &SubmitEvent) {
        _ = e;
    }

    /// Called when presentation feedback arrives.
    fn on_present_feedback(&mut self, e: &PresentFeedbackEvent) {
        _ = e;
    }

    /// Called with a per-frame timing summary.
    fn on_frame_summary(&mut self, s: &FrameSummary) {
        _ = s;
    }

    /// Called with per-frame layer changes (requires `trace-rich` feature).
    #[cfg(feature = "trace-rich")]
    fn on_layer_changes(&mut self, frame_index: u64, changes: &[LayerChange]) {
        _ = (frame_index, changes);
    }

    /// Called with per-frame damage rectangles (requires `trace-rich` feature).
    #[cfg(feature = "trace-rich")]
    fn on_damage_rects(&mut self, frame_index: u64, rects: &[DamageRect]) {
        _ = (frame_index, rects);
    }
}

// ---------------------------------------------------------------------------
// NoopSink
// ---------------------------------------------------------------------------

/// A [`TraceSink`] that discards all events.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopSink;

impl TraceSink for NoopSink {}

// ---------------------------------------------------------------------------
// Tracer wrapper
// ---------------------------------------------------------------------------

/// Thin wrapper around an optional [`TraceSink`].
///
/// When the `trace` feature is **off**, every method compiles to nothing. When
/// **on**, each method checks the inner `Option` (one branch) before
/// dispatching to the sink.
pub struct Tracer<'a> {
    #[cfg(feature = "trace")]
    sink: Option<&'a mut dyn TraceSink>,
    #[cfg(not(feature = "trace"))]
    _marker: core::marker::PhantomData<&'a mut dyn TraceSink>,
}

impl core::fmt::Debug for Tracer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tracer").finish_non_exhaustive()
    }
}

impl<'a> Tracer<'a> {
    /// Creates a tracer that dispatches to the given sink.
    #[inline]
    #[must_use]
    pub fn new(sink: &'a mut dyn TraceSink) -> Self {
        #[cfg(feature = "trace")]
        {
            Self { sink: Some(sink) }
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = sink;
            Self {
                _marker: core::marker::PhantomData,
            }
        }
    }

    /// Creates a tracer that discards all events.
    #[inline]
    #[must_use]
    pub fn none() -> Self {
        #[cfg(feature = "trace")]
        {
            Self { sink: None }
        }
        #[cfg(not(feature = "trace"))]
        {
            Self {
                _marker: core::marker::PhantomData,
            }
        }
    }

    /// Emits a [`FrameTickEvent`].
    #[inline]
    pub fn frame_tick(&mut self, e: &FrameTickEvent) {
        #[cfg(feature = "trace")]
        if let Some(s) = &mut self.sink {
            s.on_frame_tick(e);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = e;
        }
    }

    /// Emits a [`FramePlanEvent`].
    #[inline]
    pub fn frame_plan(&mut self, e: &FramePlanEvent) {
        #[cfg(feature = "trace")]
        if let Some(s) = &mut self.sink {
            s.on_frame_plan(e);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = e;
        }
    }

    /// Emits a [`PhaseBeginEvent`].
    #[inline]
    pub fn phase_begin(&mut self, e: &PhaseBeginEvent) {
        #[cfg(feature = "trace")]
        if let Some(s) = &mut self.sink {
            s.on_phase_begin(e);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = e;
        }
    }

    /// Emits a [`PhaseEndEvent`].
    #[inline]
    pub fn phase_end(&mut self, e: &PhaseEndEvent) {
        #[cfg(feature = "trace")]
        if let Some(s) = &mut self.sink {
            s.on_phase_end(e);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = e;
        }
    }

    /// Emits a [`SubmitEvent`].
    #[inline]
    pub fn submit(&mut self, e: &SubmitEvent) {
        #[cfg(feature = "trace")]
        if let Some(s) = &mut self.sink {
            s.on_submit(e);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = e;
        }
    }

    /// Emits a [`PresentFeedbackEvent`].
    #[inline]
    pub fn present_feedback(&mut self, e: &PresentFeedbackEvent) {
        #[cfg(feature = "trace")]
        if let Some(s) = &mut self.sink {
            s.on_present_feedback(e);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = e;
        }
    }

    /// Emits a [`FrameSummary`].
    #[inline]
    pub fn frame_summary(&mut self, s: &FrameSummary) {
        #[cfg(feature = "trace")]
        if let Some(sink) = &mut self.sink {
            sink.on_frame_summary(s);
        }
        #[cfg(not(feature = "trace"))]
        {
            _ = s;
        }
    }

    /// Emits layer changes (requires `trace-rich` feature).
    #[cfg(feature = "trace-rich")]
    #[inline]
    pub fn layer_changes(&mut self, frame_index: u64, changes: &[LayerChange]) {
        if let Some(s) = &mut self.sink {
            s.on_layer_changes(frame_index, changes);
        }
    }

    /// Emits damage rectangles (requires `trace-rich` feature).
    #[cfg(feature = "trace-rich")]
    #[inline]
    pub fn damage_rects(&mut self, frame_index: u64, rects: &[DamageRect]) {
        if let Some(s) = &mut self.sink {
            s.on_damage_rects(frame_index, rects);
        }
    }
}

// ---------------------------------------------------------------------------
// FrameSummaryBuilder
// ---------------------------------------------------------------------------

/// Collects phase timestamps during a frame and produces a [`FrameSummary`].
#[derive(Debug)]
pub struct FrameSummaryBuilder {
    tick: FrameTickEvent,
    plan: FramePlanEvent,
    phase_starts: [Option<HostTime>; 4],
    phase_ends: [Option<HostTime>; 4],
    missed_deadline: bool,
}

impl FrameSummaryBuilder {
    /// Starts building a summary for the given tick and plan.
    #[must_use]
    pub fn new(tick: &FrameTickEvent, plan: &FramePlanEvent) -> Self {
        Self {
            tick: *tick,
            plan: *plan,
            phase_starts: [None; 4],
            phase_ends: [None; 4],
            missed_deadline: false,
        }
    }

    /// Records the start of a phase.
    pub fn phase_begin(&mut self, phase: PhaseKind, t: HostTime) {
        self.phase_starts[phase_index(phase)] = Some(t);
    }

    /// Records the end of a phase.
    pub fn phase_end(&mut self, phase: PhaseKind, t: HostTime) {
        self.phase_ends[phase_index(phase)] = Some(t);
    }

    /// Sets whether the deadline was missed.
    pub fn set_missed_deadline(&mut self, missed: bool) {
        self.missed_deadline = missed;
    }

    /// Consumes the builder and produces the final [`FrameSummary`].
    #[must_use]
    pub fn finish(self) -> FrameSummary {
        FrameSummary {
            frame_index: self.tick.frame_index,
            output: self.tick.output,
            confidence: self.tick.confidence,
            now: self.tick.now,
            target_present: self.plan.target_present,
            sample_time: self.plan.sample_time,
            deadline: self.plan.commit_deadline,
            pipeline_depth: self.plan.pipeline_depth,
            plan_ticks: self.phase_duration(PhaseKind::Plan),
            eval_ticks: self.phase_duration(PhaseKind::Evaluate),
            render_ticks: self.phase_duration(PhaseKind::Render),
            submit_ticks: self.phase_duration(PhaseKind::Submit),
            missed_deadline: self.missed_deadline,
        }
    }

    fn phase_duration(&self, phase: PhaseKind) -> u64 {
        let idx = phase_index(phase);
        match (self.phase_starts[idx], self.phase_ends[idx]) {
            (Some(start), Some(end)) => end.saturating_duration_since(start).ticks(),
            _ => 0,
        }
    }
}

/// Maps a [`PhaseKind`] to an array index.
const fn phase_index(phase: PhaseKind) -> usize {
    match phase {
        PhaseKind::Plan => 0,
        PhaseKind::Evaluate => 1,
        PhaseKind::Render => 2,
        PhaseKind::Submit => 3,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::OutputId;
    use crate::time::HostTime;
    use crate::timing::TimingConfidence;

    fn sample_tick() -> FrameTickEvent {
        FrameTickEvent {
            frame_index: 42,
            output: OutputId(0),
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(1_016_667)),
            refresh_interval: Some(16_666_667),
            confidence: TimingConfidence::Predictive,
        }
    }

    fn sample_plan() -> FramePlanEvent {
        FramePlanEvent {
            frame_index: 42,
            output: OutputId(0),
            sample_time: HostTime(1_016_667),
            target_present: Some(HostTime(1_016_667)),
            commit_deadline: HostTime(1_014_000),
            pipeline_depth: 2,
            safety_margin_ticks: 500,
        }
    }

    #[test]
    fn frame_tick_event_from_frame_tick() {
        let tick = FrameTick {
            now: HostTime(100),
            predicted_present: Some(HostTime(200)),
            refresh_interval: Some(16_666_667),
            confidence: TimingConfidence::Predictive,
            frame_index: 7,
            output: OutputId(1),
            prev_actual_present: None,
        };
        let evt = FrameTickEvent::from(&tick);
        assert_eq!(evt.frame_index, 7);
        assert_eq!(evt.output, OutputId(1));
        assert_eq!(evt.now, HostTime(100));
        assert_eq!(evt.predicted_present, Some(HostTime(200)));
    }

    #[test]
    fn frame_plan_event_new() {
        let plan = FramePlan {
            sample_time: HostTime(1000),
            target_present: Some(HostTime(1000)),
            commit_deadline: HostTime(900),
            pipeline_depth: 2,
            output: OutputId(0),
            frame_index: 5,
        };
        let evt = FramePlanEvent::new(&plan, 123);
        assert_eq!(evt.frame_index, 5);
        assert_eq!(evt.safety_margin_ticks, 123);
        assert_eq!(evt.pipeline_depth, 2);
    }

    #[test]
    fn noop_sink_compiles() {
        let mut sink = NoopSink;
        sink.on_frame_tick(&sample_tick());
        sink.on_frame_plan(&sample_plan());
        sink.on_frame_summary(&FrameSummary {
            frame_index: 0,
            output: OutputId(0),
            confidence: TimingConfidence::PacingOnly,
            now: HostTime(0),
            target_present: None,
            sample_time: HostTime(0),
            deadline: HostTime(0),
            pipeline_depth: 1,
            plan_ticks: 0,
            eval_ticks: 0,
            render_ticks: 0,
            submit_ticks: 0,
            missed_deadline: false,
        });
    }

    #[test]
    fn tracer_none_does_nothing() {
        let mut tracer = Tracer::none();
        tracer.frame_tick(&sample_tick());
        tracer.frame_plan(&sample_plan());
    }

    #[test]
    fn summary_builder_computes_durations() {
        let tick = sample_tick();
        let plan = sample_plan();
        let mut builder = FrameSummaryBuilder::new(&tick, &plan);

        builder.phase_begin(PhaseKind::Plan, HostTime(1_000_000));
        builder.phase_end(PhaseKind::Plan, HostTime(1_000_100));
        builder.phase_begin(PhaseKind::Evaluate, HostTime(1_000_100));
        builder.phase_end(PhaseKind::Evaluate, HostTime(1_000_500));
        builder.phase_begin(PhaseKind::Render, HostTime(1_000_500));
        builder.phase_end(PhaseKind::Render, HostTime(1_002_000));
        builder.phase_begin(PhaseKind::Submit, HostTime(1_002_000));
        builder.phase_end(PhaseKind::Submit, HostTime(1_002_050));
        builder.set_missed_deadline(false);

        let summary = builder.finish();
        assert_eq!(summary.plan_ticks, 100);
        assert_eq!(summary.eval_ticks, 400);
        assert_eq!(summary.render_ticks, 1500);
        assert_eq!(summary.submit_ticks, 50);
        assert!(!summary.missed_deadline);
        assert_eq!(summary.frame_index, 42);
    }

    #[test]
    fn summary_builder_missing_phases_are_zero() {
        let tick = sample_tick();
        let plan = sample_plan();
        let builder = FrameSummaryBuilder::new(&tick, &plan);
        let summary = builder.finish();
        assert_eq!(summary.plan_ticks, 0);
        assert_eq!(summary.eval_ticks, 0);
        assert_eq!(summary.render_ticks, 0);
        assert_eq!(summary.submit_ticks, 0);
    }

    #[cfg(feature = "trace")]
    #[test]
    fn tracer_dispatches_to_sink() {
        use alloc::vec::Vec;

        struct RecordingSink {
            ticks: Vec<u64>,
        }
        impl TraceSink for RecordingSink {
            fn on_frame_tick(&mut self, e: &FrameTickEvent) {
                self.ticks.push(e.frame_index);
            }
        }

        let mut sink = RecordingSink { ticks: Vec::new() };
        {
            let mut tracer = Tracer::new(&mut sink);
            tracer.frame_tick(&sample_tick());
        }
        // Access sink after tracer is dropped.
        assert_eq!(sink.ticks, &[42]);
    }
}
