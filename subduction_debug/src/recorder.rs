// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compact binary event recording and decoding.
//!
//! [`RecorderSink`] implements [`TraceSink`] and encodes events into a
//! `Vec<u8>` as fixed-size little-endian records. [`decode`] reads them back
//! as an iterator of [`RecordedEvent`].
//!
//! Rich events ([`on_layer_changes`](TraceSink::on_layer_changes),
//! [`on_damage_rects`](TraceSink::on_damage_rects)) store only the count.

use frameclock::timing::PresentationTiming;
use frameclock::{Duration, FrameDemand, HostTime, OutputId};
use subduction_core::trace::{
    DamageRect, FramePlanEvent, FrameSummary, FrameTickEvent, LayerChange, PhaseBeginEvent,
    PhaseEndEvent, PhaseKind, PresentFeedbackEvent, SubmitEvent, TraceSink,
};

// ---------------------------------------------------------------------------
// Event type discriminants
// ---------------------------------------------------------------------------

const TAG_FRAME_TICK: u8 = 1;
const TAG_FRAME_PLAN: u8 = 2;
const TAG_PHASE_BEGIN: u8 = 3;
const TAG_PHASE_END: u8 = 4;
const TAG_SUBMIT: u8 = 5;
const TAG_PRESENT_FEEDBACK: u8 = 6;
const TAG_FRAME_SUMMARY: u8 = 7;
const TAG_LAYER_CHANGES_COUNT: u8 = 8;
const TAG_DAMAGE_RECTS_COUNT: u8 = 9;

// ---------------------------------------------------------------------------
// RecorderSink
// ---------------------------------------------------------------------------

/// A [`TraceSink`] that encodes events into a compact binary buffer.
#[derive(Debug, Default)]
pub struct RecorderSink {
    buf: Vec<u8>,
}

impl RecorderSink {
    /// Creates an empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a view of the recorded bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Consumes the recorder and returns the recorded bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    // -- encoding helpers --------------------------------------------------

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_option_u64(&mut self, v: Option<u64>) {
        match v {
            Some(val) => {
                self.write_u8(1);
                self.write_u64(val);
            }
            None => {
                self.write_u8(0);
                self.write_u64(0);
            }
        }
    }

    fn write_option_bool(&mut self, v: Option<bool>) {
        match v {
            Some(true) => self.write_u8(2),
            Some(false) => self.write_u8(1),
            None => self.write_u8(0),
        }
    }

    fn write_presentation_timing(&mut self, presentation_timing: PresentationTiming) {
        self.write_u8(match presentation_timing {
            PresentationTiming::Predictive => 0,
            PresentationTiming::Estimated => 1,
            PresentationTiming::PacingOnly => 2,
        });
    }

    fn write_phase(&mut self, p: PhaseKind) {
        self.write_u8(match p {
            PhaseKind::Plan => 0,
            PhaseKind::Evaluate => 1,
            PhaseKind::Render => 2,
            PhaseKind::Submit => 3,
        });
    }
}

impl TraceSink for RecorderSink {
    fn on_frame_tick(&mut self, e: &FrameTickEvent) {
        self.write_u8(TAG_FRAME_TICK);
        self.write_u64(e.frame_index);
        self.write_u32(e.output.0);
        self.write_u64(e.now.ticks());
        self.write_option_u64(e.predicted_present.map(|t| t.ticks()));
        self.write_option_u64(e.refresh_interval);
    }

    fn on_frame_plan(&mut self, e: &FramePlanEvent) {
        self.write_u8(TAG_FRAME_PLAN);
        self.write_u64(e.frame_index);
        self.write_u32(e.output.0);
        self.write_u8(e.demand.bits());
        self.write_u64(e.frame_interval.ticks());
        self.write_u64(e.frame_start.ticks());
        self.write_u64(e.sample_time.ticks());
        self.write_option_u64(e.target_present.map(|t| t.ticks()));
        self.write_presentation_timing(e.presentation_timing);
        self.write_u64(e.commit_deadline.ticks());
        self.write_u8(e.pipeline_depth);
        self.write_u64(e.safety_margin_ticks);
    }

    fn on_phase_begin(&mut self, e: &PhaseBeginEvent) {
        self.write_u8(TAG_PHASE_BEGIN);
        self.write_u64(e.frame_index);
        self.write_phase(e.phase);
        self.write_u64(e.timestamp.ticks());
    }

    fn on_phase_end(&mut self, e: &PhaseEndEvent) {
        self.write_u8(TAG_PHASE_END);
        self.write_u64(e.frame_index);
        self.write_phase(e.phase);
        self.write_u64(e.timestamp.ticks());
    }

    fn on_submit(&mut self, e: &SubmitEvent) {
        self.write_u8(TAG_SUBMIT);
        self.write_u64(e.frame_index);
        self.write_u64(e.submitted_at.ticks());
        self.write_option_u64(e.expected_present.map(|t| t.ticks()));
    }

    fn on_present_feedback(&mut self, e: &PresentFeedbackEvent) {
        self.write_u8(TAG_PRESENT_FEEDBACK);
        self.write_u64(e.frame_index);
        self.write_option_u64(e.actual_present.map(|t| t.ticks()));
        self.write_option_bool(e.missed_deadline);
        self.write_option_bool(e.pacing_overrun);
    }

    fn on_frame_summary(&mut self, s: &FrameSummary) {
        self.write_u8(TAG_FRAME_SUMMARY);
        self.write_u64(s.frame_index);
        self.write_u32(s.output.0);
        self.write_presentation_timing(s.presentation_timing);
        self.write_u64(s.now.ticks());
        self.write_option_u64(s.target_present.map(|t| t.ticks()));
        self.write_u64(s.sample_time.ticks());
        self.write_u64(s.deadline.ticks());
        self.write_u8(s.pipeline_depth);
        self.write_u64(s.plan_ticks);
        self.write_u64(s.eval_ticks);
        self.write_u64(s.render_ticks);
        self.write_u64(s.submit_ticks);
        self.write_u8(u8::from(s.missed_deadline));
    }

    fn on_layer_changes(&mut self, frame_index: u64, changes: &[LayerChange]) {
        self.write_u8(TAG_LAYER_CHANGES_COUNT);
        self.write_u64(frame_index);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "layer change count capped at u32::MAX for recording"
        )]
        self.write_u32(changes.len().min(u32::MAX as usize) as u32);
    }

    fn on_damage_rects(&mut self, frame_index: u64, rects: &[DamageRect]) {
        self.write_u8(TAG_DAMAGE_RECTS_COUNT);
        self.write_u64(frame_index);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "damage rect count capped at u32::MAX for recording"
        )]
        self.write_u32(rects.len().min(u32::MAX as usize) as u32);
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// A decoded event from a binary recording.
#[derive(Clone, Debug)]
pub enum RecordedEvent {
    /// A [`FrameTickEvent`].
    FrameTick(FrameTickEvent),
    /// A [`FramePlanEvent`].
    FramePlan(FramePlanEvent),
    /// A [`PhaseBeginEvent`].
    PhaseBegin(PhaseBeginEvent),
    /// A [`PhaseEndEvent`].
    PhaseEnd(PhaseEndEvent),
    /// A [`SubmitEvent`].
    Submit(SubmitEvent),
    /// A [`PresentFeedbackEvent`].
    PresentFeedback(PresentFeedbackEvent),
    /// A [`FrameSummary`].
    FrameSummary(FrameSummary),
    /// Layer-change count for a frame.
    LayerChangesCount {
        /// Frame counter.
        frame_index: u64,
        /// Number of layer changes.
        count: u32,
    },
    /// Damage-rect count for a frame.
    DamageRectsCount {
        /// Frame counter.
        frame_index: u64,
        /// Number of damage rects.
        count: u32,
    },
}

/// Decodes a byte slice produced by [`RecorderSink`] into an iterator of
/// [`RecordedEvent`].
pub fn decode(bytes: &[u8]) -> DecodeIter<'_> {
    DecodeIter {
        data: bytes,
        pos: 0,
    }
}

/// Iterator over decoded events.
#[derive(Debug)]
pub struct DecodeIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl DecodeIter<'_> {
    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_u8(&mut self) -> Option<u8> {
        if self.remaining() < 1 {
            return None;
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Some(v)
    }

    fn read_u32(&mut self) -> Option<u32> {
        if self.remaining() < 4 {
            return None;
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().ok()?);
        self.pos += 4;
        Some(v)
    }

    fn read_u64(&mut self) -> Option<u64> {
        if self.remaining() < 8 {
            return None;
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().ok()?);
        self.pos += 8;
        Some(v)
    }

    fn read_option_u64(&mut self) -> Option<Option<u64>> {
        let present = self.read_u8()?;
        let val = self.read_u64()?;
        Some(if present != 0 { Some(val) } else { None })
    }

    fn read_option_bool(&mut self) -> Option<Option<bool>> {
        let v = self.read_u8()?;
        Some(match v {
            0 => None,
            1 => Some(false),
            _ => Some(true),
        })
    }

    fn read_presentation_timing(&mut self) -> Option<PresentationTiming> {
        Some(match self.read_u8()? {
            0 => PresentationTiming::Predictive,
            1 => PresentationTiming::Estimated,
            _ => PresentationTiming::PacingOnly,
        })
    }

    fn read_phase(&mut self) -> Option<PhaseKind> {
        Some(match self.read_u8()? {
            0 => PhaseKind::Plan,
            1 => PhaseKind::Evaluate,
            2 => PhaseKind::Render,
            _ => PhaseKind::Submit,
        })
    }

    fn decode_frame_tick(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::FrameTick(FrameTickEvent {
            frame_index: self.read_u64()?,
            output: OutputId(self.read_u32()?),
            now: HostTime(self.read_u64()?),
            predicted_present: self.read_option_u64()?.map(HostTime),
            refresh_interval: self.read_option_u64()?,
        }))
    }

    fn decode_frame_plan(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::FramePlan(FramePlanEvent {
            frame_index: self.read_u64()?,
            output: OutputId(self.read_u32()?),
            demand: FrameDemand::from_bits_truncate(self.read_u8()?),
            frame_interval: Duration(self.read_u64()?),
            frame_start: HostTime(self.read_u64()?),
            sample_time: HostTime(self.read_u64()?),
            target_present: self.read_option_u64()?.map(HostTime),
            presentation_timing: self.read_presentation_timing()?,
            commit_deadline: HostTime(self.read_u64()?),
            pipeline_depth: self.read_u8()?,
            safety_margin_ticks: self.read_u64()?,
        }))
    }

    fn decode_phase_begin(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::PhaseBegin(PhaseBeginEvent {
            frame_index: self.read_u64()?,
            phase: self.read_phase()?,
            timestamp: HostTime(self.read_u64()?),
        }))
    }

    fn decode_phase_end(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::PhaseEnd(PhaseEndEvent {
            frame_index: self.read_u64()?,
            phase: self.read_phase()?,
            timestamp: HostTime(self.read_u64()?),
        }))
    }

    fn decode_submit(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::Submit(SubmitEvent {
            frame_index: self.read_u64()?,
            submitted_at: HostTime(self.read_u64()?),
            expected_present: self.read_option_u64()?.map(HostTime),
        }))
    }

    fn decode_present_feedback(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::PresentFeedback(PresentFeedbackEvent {
            frame_index: self.read_u64()?,
            actual_present: self.read_option_u64()?.map(HostTime),
            missed_deadline: self.read_option_bool()?,
            pacing_overrun: self.read_option_bool()?,
        }))
    }

    fn decode_frame_summary(&mut self) -> Option<RecordedEvent> {
        Some(RecordedEvent::FrameSummary(FrameSummary {
            frame_index: self.read_u64()?,
            output: OutputId(self.read_u32()?),
            presentation_timing: self.read_presentation_timing()?,
            now: HostTime(self.read_u64()?),
            target_present: self.read_option_u64()?.map(HostTime),
            sample_time: HostTime(self.read_u64()?),
            deadline: HostTime(self.read_u64()?),
            pipeline_depth: self.read_u8()?,
            plan_ticks: self.read_u64()?,
            eval_ticks: self.read_u64()?,
            render_ticks: self.read_u64()?,
            submit_ticks: self.read_u64()?,
            missed_deadline: self.read_u8()? != 0,
        }))
    }

    fn decode_layer_changes_count(&mut self) -> Option<RecordedEvent> {
        let frame_index = self.read_u64()?;
        let count = self.read_u32()?;
        Some(RecordedEvent::LayerChangesCount { frame_index, count })
    }

    fn decode_damage_rects_count(&mut self) -> Option<RecordedEvent> {
        let frame_index = self.read_u64()?;
        let count = self.read_u32()?;
        Some(RecordedEvent::DamageRectsCount { frame_index, count })
    }
}

impl Iterator for DecodeIter<'_> {
    type Item = RecordedEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let tag = self.read_u8()?;
        match tag {
            TAG_FRAME_TICK => self.decode_frame_tick(),
            TAG_FRAME_PLAN => self.decode_frame_plan(),
            TAG_PHASE_BEGIN => self.decode_phase_begin(),
            TAG_PHASE_END => self.decode_phase_end(),
            TAG_SUBMIT => self.decode_submit(),
            TAG_PRESENT_FEEDBACK => self.decode_present_feedback(),
            TAG_FRAME_SUMMARY => self.decode_frame_summary(),
            TAG_LAYER_CHANGES_COUNT => self.decode_layer_changes_count(),
            TAG_DAMAGE_RECTS_COUNT => self.decode_damage_rects_count(),
            _ => None, // unknown tag → stop iteration
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tick_event() -> FrameTickEvent {
        FrameTickEvent {
            frame_index: 7,
            output: OutputId(1),
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(1_016_667)),
            refresh_interval: Some(16_666_667),
        }
    }

    fn sample_plan_event() -> FramePlanEvent {
        FramePlanEvent {
            frame_index: 7,
            output: OutputId(1),
            demand: FrameDemand::ANIMATION,
            frame_interval: Duration(16_666_667),
            frame_start: HostTime(1_013_500),
            sample_time: HostTime(1_016_667),
            target_present: Some(HostTime(1_016_667)),
            presentation_timing: PresentationTiming::Predictive,
            commit_deadline: HostTime(1_014_000),
            pipeline_depth: 2,
            safety_margin_ticks: 500,
        }
    }

    fn sample_summary() -> FrameSummary {
        FrameSummary {
            frame_index: 7,
            output: OutputId(1),
            presentation_timing: PresentationTiming::Predictive,
            now: HostTime(1_000_000),
            target_present: Some(HostTime(1_016_667)),
            sample_time: HostTime(1_016_667),
            deadline: HostTime(1_014_000),
            pipeline_depth: 2,
            plan_ticks: 100,
            eval_ticks: 400,
            render_ticks: 1500,
            submit_ticks: 50,
            missed_deadline: false,
        }
    }

    #[test]
    fn round_trip_frame_tick() {
        let mut rec = RecorderSink::new();
        let orig = sample_tick_event();
        rec.on_frame_tick(&orig);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            RecordedEvent::FrameTick(e) => {
                assert_eq!(e.frame_index, orig.frame_index);
                assert_eq!(e.output, orig.output);
                assert_eq!(e.now, orig.now);
                assert_eq!(e.predicted_present, orig.predicted_present);
                assert_eq!(e.refresh_interval, orig.refresh_interval);
            }
            other => panic!("expected FrameTick, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_frame_plan() {
        let mut rec = RecorderSink::new();
        let orig = sample_plan_event();
        rec.on_frame_plan(&orig);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            RecordedEvent::FramePlan(e) => {
                assert_eq!(e.frame_index, orig.frame_index);
                assert_eq!(e.demand, orig.demand);
                assert_eq!(e.frame_interval, orig.frame_interval);
                assert_eq!(e.frame_start, orig.frame_start);
                assert_eq!(e.presentation_timing, orig.presentation_timing);
                assert_eq!(e.pipeline_depth, orig.pipeline_depth);
                assert_eq!(e.safety_margin_ticks, orig.safety_margin_ticks);
            }
            other => panic!("expected FramePlan, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_phase_events() {
        let mut rec = RecorderSink::new();
        let begin = PhaseBeginEvent {
            frame_index: 5,
            phase: PhaseKind::Render,
            timestamp: HostTime(2000),
        };
        let end = PhaseEndEvent {
            frame_index: 5,
            phase: PhaseKind::Render,
            timestamp: HostTime(3000),
        };
        rec.on_phase_begin(&begin);
        rec.on_phase_end(&end);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 2);
        match &events[0] {
            RecordedEvent::PhaseBegin(e) => {
                assert_eq!(e.frame_index, 5);
                assert_eq!(e.phase, PhaseKind::Render);
                assert_eq!(e.timestamp, HostTime(2000));
            }
            other => panic!("expected PhaseBegin, got {other:?}"),
        }
        match &events[1] {
            RecordedEvent::PhaseEnd(e) => {
                assert_eq!(e.frame_index, 5);
                assert_eq!(e.phase, PhaseKind::Render);
                assert_eq!(e.timestamp, HostTime(3000));
            }
            other => panic!("expected PhaseEnd, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_submit() {
        let mut rec = RecorderSink::new();
        let orig = SubmitEvent {
            frame_index: 10,
            submitted_at: HostTime(5000),
            expected_present: Some(HostTime(6000)),
        };
        rec.on_submit(&orig);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            RecordedEvent::Submit(e) => {
                assert_eq!(e.frame_index, 10);
                assert_eq!(e.submitted_at, HostTime(5000));
                assert_eq!(e.expected_present, Some(HostTime(6000)));
            }
            other => panic!("expected Submit, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_present_feedback() {
        let mut rec = RecorderSink::new();
        let orig = PresentFeedbackEvent {
            frame_index: 3,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: Some(false),
        };
        rec.on_present_feedback(&orig);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            RecordedEvent::PresentFeedback(e) => {
                assert_eq!(e.frame_index, 3);
                assert_eq!(e.actual_present, None);
                assert_eq!(e.missed_deadline, Some(true));
                assert_eq!(e.pacing_overrun, Some(false));
            }
            other => panic!("expected PresentFeedback, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_frame_summary() {
        let mut rec = RecorderSink::new();
        let orig = sample_summary();
        rec.on_frame_summary(&orig);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            RecordedEvent::FrameSummary(s) => {
                assert_eq!(s.frame_index, orig.frame_index);
                assert_eq!(s.plan_ticks, orig.plan_ticks);
                assert_eq!(s.eval_ticks, orig.eval_ticks);
                assert_eq!(s.render_ticks, orig.render_ticks);
                assert_eq!(s.submit_ticks, orig.submit_ticks);
                assert_eq!(s.missed_deadline, orig.missed_deadline);
                assert_eq!(s.presentation_timing, orig.presentation_timing);
                assert_eq!(s.pipeline_depth, orig.pipeline_depth);
            }
            other => panic!("expected FrameSummary, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_multiple_events() {
        let mut rec = RecorderSink::new();
        rec.on_frame_tick(&sample_tick_event());
        rec.on_frame_plan(&sample_plan_event());
        rec.on_phase_begin(&PhaseBeginEvent {
            frame_index: 7,
            phase: PhaseKind::Plan,
            timestamp: HostTime(1000),
        });
        rec.on_frame_summary(&sample_summary());

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], RecordedEvent::FrameTick(_)));
        assert!(matches!(events[1], RecordedEvent::FramePlan(_)));
        assert!(matches!(events[2], RecordedEvent::PhaseBegin(_)));
        assert!(matches!(events[3], RecordedEvent::FrameSummary(_)));
    }

    #[test]
    fn empty_buffer_decodes_to_nothing() {
        let events: Vec<_> = decode(&[]).collect();
        assert!(events.is_empty());
    }

    #[test]
    fn layer_changes_count() {
        use subduction_core::trace::LayerField;
        let mut rec = RecorderSink::new();
        let changes = vec![
            LayerChange {
                layer_index: 0,
                field: LayerField::Transform,
            },
            LayerChange {
                layer_index: 1,
                field: LayerField::Opacity,
            },
        ];
        rec.on_layer_changes(42, &changes);

        let events: Vec<_> = decode(rec.as_bytes()).collect();
        assert_eq!(events.len(), 1);
        match &events[0] {
            RecordedEvent::LayerChangesCount { frame_index, count } => {
                assert_eq!(*frame_index, 42);
                assert_eq!(*count, 2);
            }
            other => panic!("expected LayerChangesCount, got {other:?}"),
        }
    }
}
