// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Chrome Trace Event Format exporter.
//!
//! [`export`] reads recorded bytes from a [`RecorderSink`](super::recorder::RecorderSink)
//! and writes [Chrome Trace Event Format][spec] JSON to the given writer.
//!
//! [spec]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU

use std::io::{self, Write};

use serde_json::{Value, json};

use frameclock::time::Timebase;

use crate::recorder::{RecordedEvent, decode};

/// Exports recorded events as Chrome Trace Event Format JSON.
///
/// The output is a complete JSON array of trace event objects, suitable for
/// loading into `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).
///
/// Timestamps are converted to microseconds using the provided [`Timebase`].
pub fn export(bytes: &[u8], timebase: Timebase, writer: &mut dyn Write) -> io::Result<()> {
    let mut events: Vec<Value> = Vec::new();

    for recorded in decode(bytes) {
        match recorded {
            RecordedEvent::FrameTick(e) => {
                events.push(json!({
                    "ph": "i",
                    "name": "FrameTick",
                    "cat": "Scheduler",
                    "ts": ticks_to_us(e.now.ticks(), timebase),
                    "pid": e.output.0,
                    "tid": 0,
                    "s": "g",
                    "args": {
                        "frame_index": e.frame_index,
                    }
                }));
            }
            RecordedEvent::FramePlan(e) => {
                events.push(json!({
                    "ph": "i",
                    "name": "FramePlan",
                    "cat": "Scheduler",
                    "ts": ticks_to_us(e.commit_deadline.ticks(), timebase),
                    "pid": e.output.0,
                    "tid": 0,
                    "s": "g",
                    "args": {
                        "frame_index": e.frame_index,
                        "presentation_timing": format!("{:?}", e.presentation_timing),
                        "pipeline_depth": e.pipeline_depth,
                        "safety_margin_ticks": e.safety_margin_ticks,
                    }
                }));
            }
            RecordedEvent::PhaseBegin(e) => {
                events.push(json!({
                    "ph": "B",
                    "name": format!("{:?}", e.phase),
                    "cat": "Frame",
                    "ts": ticks_to_us(e.timestamp.ticks(), timebase),
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "frame_index": e.frame_index,
                    }
                }));
            }
            RecordedEvent::PhaseEnd(e) => {
                events.push(json!({
                    "ph": "E",
                    "name": format!("{:?}", e.phase),
                    "cat": "Frame",
                    "ts": ticks_to_us(e.timestamp.ticks(), timebase),
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "frame_index": e.frame_index,
                    }
                }));
            }
            RecordedEvent::Submit(e) => {
                events.push(json!({
                    "ph": "i",
                    "name": "Submit",
                    "cat": "Frame",
                    "ts": ticks_to_us(e.submitted_at.ticks(), timebase),
                    "pid": 0,
                    "tid": 0,
                    "s": "t",
                    "args": {
                        "frame_index": e.frame_index,
                    }
                }));
            }
            RecordedEvent::PresentFeedback(e) => {
                events.push(json!({
                    "ph": "i",
                    "name": "PresentFeedback",
                    "cat": "Frame",
                    "ts": e.actual_present.map_or(0.0, |t| ticks_to_us(t.ticks(), timebase)),
                    "pid": 0,
                    "tid": 0,
                    "s": "t",
                    "args": {
                        "frame_index": e.frame_index,
                        "missed": e.missed_deadline,
                        "pacing_overrun": e.pacing_overrun,
                    }
                }));
            }
            RecordedEvent::FrameSummary(s) => {
                events.push(json!({
                    "ph": "i",
                    "name": "FrameSummary",
                    "cat": "Summary",
                    "ts": ticks_to_us(s.now.ticks(), timebase),
                    "pid": s.output.0,
                    "tid": 0,
                    "s": "g",
                    "args": {
                        "frame_index": s.frame_index,
                        "pipeline_depth": s.pipeline_depth,
                        "plan_us": ticks_to_us(s.plan_ticks, timebase),
                        "eval_us": ticks_to_us(s.eval_ticks, timebase),
                        "render_us": ticks_to_us(s.render_ticks, timebase),
                        "submit_us": ticks_to_us(s.submit_ticks, timebase),
                        "missed_deadline": s.missed_deadline,
                    }
                }));
            }
            RecordedEvent::LayerChangesCount { frame_index, count } => {
                events.push(json!({
                    "ph": "i",
                    "name": "LayerChanges",
                    "cat": "Rich",
                    "ts": 0,
                    "pid": 0,
                    "tid": 0,
                    "s": "p",
                    "args": {
                        "frame_index": frame_index,
                        "count": count,
                    }
                }));
            }
            RecordedEvent::DamageRectsCount { frame_index, count } => {
                events.push(json!({
                    "ph": "i",
                    "name": "DamageRects",
                    "cat": "Rich",
                    "ts": 0,
                    "pid": 0,
                    "tid": 0,
                    "s": "p",
                    "args": {
                        "frame_index": frame_index,
                        "count": count,
                    }
                }));
            }
        }
    }

    serde_json::to_writer_pretty(writer, &events)?;
    Ok(())
}

fn ticks_to_us(ticks: u64, timebase: Timebase) -> f64 {
    timebase.ticks_to_nanos(ticks) as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::RecorderSink;
    use frameclock::HostTime;
    use frameclock::OutputId;
    use subduction_core::trace::{
        FrameTickEvent, PhaseBeginEvent, PhaseEndEvent, PhaseKind, TraceSink,
    };

    #[test]
    fn export_produces_valid_json() {
        let mut rec = RecorderSink::new();
        rec.on_frame_tick(&FrameTickEvent {
            frame_index: 0,
            output: OutputId(0),
            now: HostTime(1_000_000),
            predicted_present: None,
            refresh_interval: Some(16_666_667),
        });
        rec.on_phase_begin(&PhaseBeginEvent {
            frame_index: 0,
            phase: PhaseKind::Plan,
            timestamp: HostTime(1_000_000),
        });
        rec.on_phase_end(&PhaseEndEvent {
            frame_index: 0,
            phase: PhaseKind::Plan,
            timestamp: HostTime(1_000_100),
        });

        let mut out = Vec::new();
        export(rec.as_bytes(), Timebase::NANOS, &mut out).unwrap();
        let json_str = String::from_utf8(out).unwrap();

        // Should parse as a JSON array.
        let parsed: Vec<Value> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.len(), 3);

        // First event is an instant FrameTick.
        assert_eq!(parsed[0]["ph"], "i");
        assert_eq!(parsed[0]["name"], "FrameTick");

        // Second is a phase begin.
        assert_eq!(parsed[1]["ph"], "B");
        assert_eq!(parsed[1]["name"], "Plan");

        // Third is a phase end.
        assert_eq!(parsed[2]["ph"], "E");
        assert_eq!(parsed[2]["name"], "Plan");
    }

    #[test]
    fn export_empty_recording() {
        let mut out = Vec::new();
        export(&[], Timebase::NANOS, &mut out).unwrap();
        let json_str = String::from_utf8(out).unwrap();
        let parsed: Vec<Value> = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_empty());
    }
}
