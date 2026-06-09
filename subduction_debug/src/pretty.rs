// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Human-readable trace output.
//!
//! [`PrettyPrintSink`] implements [`TraceSink`] and writes one line per event
//! to a [`Write`] destination (default: stderr). Timestamps
//! are converted to microseconds using a [`Timebase`].

use std::io::Write;

use frameclock::Timebase;
use subduction_core::trace::{
    DamageRect, FramePlanEvent, FrameSummary, FrameTickEvent, LayerChange, PhaseBeginEvent,
    PhaseEndEvent, PhaseKind, PresentFeedbackEvent, SubmitEvent, TraceSink,
};

/// Writes human-readable trace lines to a [`Write`] destination.
pub struct PrettyPrintSink<W: Write = Box<dyn Write>> {
    writer: W,
    timebase: Timebase,
}

impl<W: Write> std::fmt::Debug for PrettyPrintSink<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrettyPrintSink")
            .field("timebase", &self.timebase)
            .finish_non_exhaustive()
    }
}

impl PrettyPrintSink {
    /// Creates a sink that writes to stderr.
    #[must_use]
    pub fn stderr(timebase: Timebase) -> Self {
        Self {
            writer: Box::new(std::io::stderr()),
            timebase,
        }
    }

    /// Creates a sink that writes to a boxed writer.
    #[must_use]
    pub fn new(writer: Box<dyn Write>, timebase: Timebase) -> Self {
        Self { writer, timebase }
    }
}

impl<W: Write> PrettyPrintSink<W> {
    /// Creates a sink that writes to the given destination.
    #[must_use]
    pub fn with_writer(writer: W, timebase: Timebase) -> Self {
        Self { writer, timebase }
    }

    fn ticks_to_us(&self, ticks: u64) -> f64 {
        self.timebase.ticks_to_nanos(ticks) as f64 / 1000.0
    }

    fn host_us(&self, t: frameclock::HostTime) -> f64 {
        self.ticks_to_us(t.ticks())
    }
}

fn phase_name(phase: PhaseKind) -> &'static str {
    match phase {
        PhaseKind::Plan => "plan",
        PhaseKind::Evaluate => "eval",
        PhaseKind::Render => "render",
        PhaseKind::Submit => "submit",
    }
}

impl<W: Write> TraceSink for PrettyPrintSink<W> {
    fn on_frame_tick(&mut self, e: &FrameTickEvent) {
        let _ = writeln!(
            self.writer,
            "[tick] frame={} output={} now={:.1}µs confidence={:?}",
            e.frame_index,
            e.output.0,
            self.host_us(e.now),
            e.confidence,
        );
    }

    fn on_frame_plan(&mut self, e: &FramePlanEvent) {
        let _ = writeln!(
            self.writer,
            "[plan] frame={} demand={:?} interval={:.1}µs start={:.1}µs deadline={:.1}µs depth={} margin={}t",
            e.frame_index,
            e.demand,
            self.ticks_to_us(e.frame_interval.ticks()),
            self.host_us(e.frame_start),
            self.host_us(e.commit_deadline),
            e.pipeline_depth,
            e.safety_margin_ticks,
        );
    }

    fn on_phase_begin(&mut self, e: &PhaseBeginEvent) {
        let _ = writeln!(
            self.writer,
            "[phase:begin] frame={} {} at {:.1}µs",
            e.frame_index,
            phase_name(e.phase),
            self.host_us(e.timestamp),
        );
    }

    fn on_phase_end(&mut self, e: &PhaseEndEvent) {
        let _ = writeln!(
            self.writer,
            "[phase:end] frame={} {} at {:.1}µs",
            e.frame_index,
            phase_name(e.phase),
            self.host_us(e.timestamp),
        );
    }

    fn on_submit(&mut self, e: &SubmitEvent) {
        let _ = writeln!(
            self.writer,
            "[submit] frame={} at {:.1}µs",
            e.frame_index,
            self.host_us(e.submitted_at),
        );
    }

    fn on_present_feedback(&mut self, e: &PresentFeedbackEvent) {
        let missed = match e.missed_deadline {
            Some(true) => "MISSED",
            Some(false) => "ok",
            None => "?",
        };
        let overrun = match e.pacing_overrun {
            Some(true) => "overrun",
            Some(false) => "ok",
            None => "?",
        };
        let _ = writeln!(
            self.writer,
            "[feedback] frame={} missed={missed} pacing={overrun}",
            e.frame_index,
        );
    }

    fn on_frame_summary(&mut self, s: &FrameSummary) {
        let missed = if s.missed_deadline { "MISSED" } else { "ok" };
        let _ = writeln!(
            self.writer,
            "[summary] frame={} depth={} plan={:.1}µs eval={:.1}µs \
             render={:.1}µs submit={:.1}µs deadline={missed}",
            s.frame_index,
            s.pipeline_depth,
            self.ticks_to_us(s.plan_ticks),
            self.ticks_to_us(s.eval_ticks),
            self.ticks_to_us(s.render_ticks),
            self.ticks_to_us(s.submit_ticks),
        );
    }

    fn on_layer_changes(&mut self, frame_index: u64, changes: &[LayerChange]) {
        let _ = writeln!(
            self.writer,
            "[layers] frame={frame_index} changes={}",
            changes.len(),
        );
    }

    fn on_damage_rects(&mut self, frame_index: u64, rects: &[DamageRect]) {
        let _ = writeln!(
            self.writer,
            "[damage] frame={frame_index} rects={}",
            rects.len(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameclock::HostTime;
    use frameclock::OutputId;
    use frameclock::TimingConfidence;

    #[test]
    fn pretty_print_tick() {
        let mut sink = PrettyPrintSink::with_writer(Vec::<u8>::new(), Timebase::NANOS);
        sink.on_frame_tick(&FrameTickEvent {
            frame_index: 1,
            output: OutputId(0),
            now: HostTime(1_000_000),
            predicted_present: None,
            refresh_interval: None,
            confidence: TimingConfidence::PacingOnly,
        });
        let output = String::from_utf8(sink.writer).unwrap();
        assert!(output.contains("[tick]"), "got: {output}");
        assert!(output.contains("frame=1"), "got: {output}");
    }
}
