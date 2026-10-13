// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a `Recorder` emits, as a Footfall backend sees it.

use std::sync::{Mutex, MutexGuard, OnceLock};

use footfall_core::{
    Backend, ClockReading, ContextHandle, CounterDesc, DropReason, FramePoint, FramePointKind,
    InstrumentationLevel, MarkerKind, MarkerPhase, TickPeriod, Unit,
};
use frameclock::diagnostics::{FrameDropReason, FrameTimingBasis, FrameTimingSummary};
use frameclock::scheduler::SchedulerState;
use frameclock::time::Timebase;
use frameclock::timing::{FrameDemand, PresentationTiming};
use frameclock::{Duration, HostTime, OutputId};
use frameclock_footfall::{HOST_CLOCK, Recorder};

/// Mach absolute time on Apple silicon: 125 ns per 3 ticks.
const MACH: Timebase = Timebase::new(125, 3);

#[derive(Debug, PartialEq)]
enum Seen {
    Marker(MarkerPhase, u64),
    Sync(u64),
    Point(u64, FramePointKind, Option<u64>),
    Counter(&'static str, Unit, &'static str, u64),
}

#[derive(Default)]
struct Recording(Mutex<Vec<Seen>>);

impl Recording {
    fn push(&self, seen: Seen) {
        self.0.lock().unwrap().push(seen);
    }
}

impl Backend for Recording {
    fn numbered_marker(
        &self,
        kind: MarkerKind,
        phase: MarkerPhase,
        number: u64,
        _context: ContextHandle,
    ) {
        assert_eq!(kind, MarkerKind::Frame, "only frame markers are emitted");
        self.push(Seen::Marker(phase, number));
    }

    fn clock_sync(&self, reading: ClockReading) {
        assert!(
            core::ptr::eq(reading.clock(), &HOST_CLOCK),
            "syncs are of the host clock"
        );
        assert_eq!(
            reading.period(),
            TickPeriod::new(125, 3),
            "the Mach timebase"
        );
        self.push(Seen::Sync(reading.ticks()));
    }

    fn frame_point(&self, point: &FramePoint) {
        if let Some(time) = point.time() {
            assert!(
                core::ptr::eq(time.clock(), &HOST_CLOCK),
                "points are in the host clock"
            );
            assert_eq!(time.period(), TickPeriod::new(125, 3), "the Mach timebase");
        }
        self.push(Seen::Point(
            point.number(),
            point.kind(),
            point.time().map(|time| time.ticks()),
        ));
    }

    fn counter_add(&self, desc: &'static CounterDesc, _context: ContextHandle, value: u64) {
        self.push(Seen::Counter(desc.name(), desc.unit(), "add", value));
    }

    fn gauge_set(&self, desc: &'static CounterDesc, _context: ContextHandle, value: u64) {
        self.push(Seen::Counter(desc.name(), desc.unit(), "gauge", value));
    }

    fn histogram_record(&self, desc: &'static CounterDesc, _context: ContextHandle, value: u64) {
        self.push(Seen::Counter(desc.name(), desc.unit(), "histogram", value));
    }
}

/// Installs the recording backend once, and serializes the tests that read
/// it, since a Footfall backend is process-wide.
fn recording() -> (&'static Recording, MutexGuard<'static, ()>) {
    static BACKEND: OnceLock<&'static Recording> = OnceLock::new();
    static SERIAL: Mutex<()> = Mutex::new(());
    let guard = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let backend = *BACKEND.get_or_init(|| {
        let backend: &'static Recording = Box::leak(Box::default());
        footfall_core::install_backend(backend).expect("nothing else installs a backend");
        footfall_core::set_max_level(InstrumentationLevel::Trace);
        backend
    });
    backend.0.lock().unwrap().clear();
    (backend, guard)
}

fn taken(backend: &Recording) -> Vec<Seen> {
    core::mem::take(&mut *backend.0.lock().unwrap())
}

fn summary(frame_index: u64) -> FrameTimingSummary {
    FrameTimingSummary {
        frame_index,
        output: OutputId(0),
        timing_basis: FrameTimingBasis::ActualPresent,
        presentation_timing: PresentationTiming::Predictive,
        tick_time: HostTime(1_000),
        predicted_present: Some(HostTime(3_900)),
        refresh_interval: Some(3_000),
        demand: FrameDemand::ANIMATION,
        frame_interval: Duration(3_000),
        frame_start: HostTime(1_200),
        sample_time: HostTime(4_000),
        target_present: Some(HostTime(4_000)),
        commit_deadline: HostTime(3_250),
        submitted_at: Some(HostTime(2_900)),
        work: Some(Duration(600)),
        expected_present: Some(HostTime(4_100)),
        actual_present: Some(HostTime(7_000)),
        missed_deadline: Some(true),
        pacing_overrun: None,
        drop_reason: None,
        pipeline_depth: 2,
        safety_margin_ticks: 300,
        scheduler_state: Some(SchedulerState {
            pipeline_depth: 2,
            present_latency_ticks: 30,
            safety_margin_ticks: 300,
            consecutive_misses: 1,
            consecutive_hits: 0,
        }),
    }
}

#[test]
fn a_shown_frame_records_every_time_frameclock_knows_in_host_ticks() {
    let (backend, _serial) = recording();
    Recorder::new(MACH).record(&summary(1234));

    assert_eq!(
        taken(backend),
        [
            Seen::Point(1234, FramePointKind::Predicted, Some(3_900)),
            Seen::Point(1234, FramePointKind::Deadline, Some(3_250)),
            Seen::Point(1234, FramePointKind::Target, Some(4_000)),
            Seen::Point(1234, FramePointKind::Expected, Some(4_100)),
            Seen::Point(1234, FramePointKind::Presented, Some(7_000)),
            Seen::Counter("frameclock_pipeline_depth", Unit::Count, "gauge", 2),
            // 300 Mach ticks are 12.5 µs.
            Seen::Counter("frameclock_safety_margin", Unit::Nanos, "gauge", 12_500),
            Seen::Counter("frameclock_frame_interval", Unit::Nanos, "gauge", 125_000),
            Seen::Counter("frameclock_present_latency", Unit::Nanos, "gauge", 1_250),
            Seen::Counter(
                "frameclock_submission_work",
                Unit::Nanos,
                "histogram",
                25_000
            ),
        ]
    );
}

#[test]
fn a_frame_the_display_never_showed_is_dropped_as_replaced() {
    let (backend, _serial) = recording();
    let mut replaced = summary(7);
    replaced.actual_present = None;
    replaced.missed_deadline = None;
    replaced.work = None;
    replaced.scheduler_state = None;
    replaced.drop_reason = Some(FrameDropReason::NotPresented);
    Recorder::new(MACH).record(&replaced);

    let seen = taken(backend);
    assert_eq!(
        seen[..5],
        [
            Seen::Point(7, FramePointKind::Predicted, Some(3_900)),
            Seen::Point(7, FramePointKind::Deadline, Some(3_250)),
            Seen::Point(7, FramePointKind::Target, Some(4_000)),
            Seen::Point(7, FramePointKind::Expected, Some(4_100)),
            Seen::Point(7, FramePointKind::Dropped(DropReason::Replaced), None),
        ],
        "a replaced frame keeps its plan's times"
    );
    assert!(
        !seen
            .iter()
            .any(|seen| matches!(seen, Seen::Counter("frameclock_present_latency", ..))),
        "no scheduler state was sampled"
    );
}

#[test]
fn every_drop_reason_reaches_footfall() {
    let (backend, _serial) = recording();
    let recorder = Recorder::new(MACH);
    let reasons = [
        (FrameDropReason::Discarded, DropReason::Discarded),
        (FrameDropReason::Superseded, DropReason::Superseded),
        (FrameDropReason::MissedDeadline, DropReason::MissedDeadline),
        (
            FrameDropReason::SurfaceUnavailable,
            DropReason::SurfaceUnavailable,
        ),
        (
            FrameDropReason::OutputUnavailable,
            DropReason::OutputUnavailable,
        ),
        (FrameDropReason::NotPresented, DropReason::Replaced),
    ];
    for (number, (frameclock_reason, _)) in (0_u64..).zip(reasons) {
        let mut dropped = summary(number);
        dropped.actual_present = None;
        dropped.drop_reason = Some(frameclock_reason);
        recorder.record(&dropped);
    }

    let drops: Vec<_> = taken(backend)
        .into_iter()
        .filter_map(|seen| match seen {
            Seen::Point(number, FramePointKind::Dropped(reason), None) => Some((number, reason)),
            _ => None,
        })
        .collect();
    let expected: Vec<_> = (0_u64..)
        .zip(reasons)
        .map(|(number, (_, reason))| (number, reason))
        .collect();
    assert_eq!(drops, expected, "one drop per frame, with its reason");
}

#[test]
fn a_pacing_only_frame_records_its_deadline_alone() {
    let (backend, _serial) = recording();
    let mut paced = summary(9);
    paced.timing_basis = FrameTimingBasis::PacingOnly;
    paced.presentation_timing = PresentationTiming::PacingOnly;
    paced.target_present = None;
    paced.predicted_present = None;
    paced.expected_present = None;
    paced.actual_present = None;
    Recorder::new(MACH).record(&paced);

    let points: Vec<_> = taken(backend)
        .into_iter()
        .filter(|seen| matches!(seen, Seen::Point(..)))
        .collect();
    assert_eq!(
        points,
        [Seen::Point(9, FramePointKind::Deadline, Some(3_250))]
    );
}

#[test]
fn syncs_and_frames_are_in_frameclock_terms() {
    let (backend, _serial) = recording();
    let recorder = Recorder::new(MACH);
    let mut driver = frameclock::FrameDriver::new(frameclock::SchedulerConfig::pacing_only());
    driver.request(FrameDemand::INPUT);

    recorder.sync(HostTime(24_000_000));
    let frameclock::FrameBeginResult::Ready(frame) = driver
        .begin_frame(frameclock::FrameOpportunity::pacing_only(
            HostTime(24_000_000),
            Duration(400_000),
            OutputId(0),
        ))
        .result
    else {
        panic!("input starts a frame right away");
    };
    let number = frame.plan().frame_index;
    recorder.begin_frame(&frame);
    recorder.end_frame(&frame);

    assert_eq!(
        taken(backend),
        [
            Seen::Sync(24_000_000),
            Seen::Marker(MarkerPhase::Begin, number),
            Seen::Marker(MarkerPhase::End, number),
        ]
    );
}

#[test]
fn as_a_diagnostics_sink_it_records_summaries() {
    let (backend, _serial) = recording();
    let mut sink = Recorder::new(MACH);
    let mut diagnostics = frameclock::diagnostics::Diagnostics::new(&mut sink);
    diagnostics.frame_timing_summary(&summary(42));

    assert!(
        taken(backend).contains(&Seen::Point(42, FramePointKind::Presented, Some(7_000))),
        "the summary reached Footfall"
    );
}
