// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simulated `FrameDriver` recorded into a real Footfall runtime, judged by
//! Footfall's pacing analysis.

use footfall_analysis::{FramePacing, LateCause, Outcome, OwnedSnapshot, Support, Trace};
use footfall_core::{DropReason, FramePointKind, InstrumentationLevel, TimingBasis};
use footfall_runtime::Runtime;
use frameclock::diagnostics::{FrameDropReason, FrameTimingSummary};
use frameclock::time::Timebase;
use frameclock::timing::{DisplayTiming, FrameTick};
use frameclock::{
    Duration, FrameBeginResult, FrameDemand, FrameDriver, FrameOpportunity, FrameReport,
    FrameSubmission, HostTime, OutputId, PresentHints, PresentOutcome, SchedulerConfig,
};
use frameclock_footfall::Recorder;

/// A 60 Hz display, in nanosecond host ticks.
const REFRESH: u64 = 16_666_667;
const START: u64 = 1_000_000_000;

fn opportunity(now: u64) -> FrameOpportunity {
    let tick = FrameTick {
        now: HostTime(now),
        predicted_present: Some(HostTime(now + REFRESH)),
        refresh_interval: Some(REFRESH),
        output: OutputId(0),
        prev_actual_present: None,
    };
    FrameOpportunity::new(
        tick,
        PresentHints::predictive(HostTime(now + REFRESH), HostTime(now + REFRESH * 3 / 4)),
        DisplayTiming::fixed(Duration(REFRESH)),
    )
}

#[test]
fn pacing_judges_frames_and_predictions_from_what_frameclock_learned() {
    let runtime = Runtime::builder()
        .level(InstrumentationLevel::Trace)
        .event_capacity(4096)
        .install_global()
        .expect("nothing else installs a backend");
    let recorder = Recorder::new(Timebase::NANOS);
    let mut driver = FrameDriver::new(SchedulerConfig::predictive());

    // The simulation's host clock runs far ahead of the runtime's real one,
    // so it is synced once: Footfall then places host times at the declared
    // period from that sync, exactly. A real host syncs each frame.
    recorder.sync(HostTime(START));

    let tolerance = core::time::Duration::from_millis(1);
    let mut late = Vec::new();
    let mut summaries = Vec::new();
    for frame in 0..12_u64 {
        let now = START + frame * REFRESH;
        driver.request(FrameDemand::INPUT);
        let FrameBeginResult::Ready(active) = driver.begin_frame(opportunity(now)).result else {
            panic!("input starts frame {frame} right away");
        };
        let number = active.plan().frame_index;
        let target = active
            .plan()
            .target_present
            .expect("the plan is predictive");
        recorder.begin_frame(&active);
        recorder.end_frame(&active);

        // Frame 5 was made stale by newer work, and frame 9 found no
        // drawable: neither was submitted.
        let host_drop = match frame {
            5 => Some(FrameDropReason::Superseded),
            9 => Some(FrameDropReason::SurfaceUnavailable),
            _ => None,
        };
        let summary = if let Some(reason) = host_drop {
            driver.discard_frame_with_reason(active, reason)
        } else {
            let submit =
                driver.submit_frame(active, FrameSubmission::reported(HostTime(now + 1_000_000)));
            assert_eq!(submit.summary, None, "frame {frame} waits for its report");
            let outcome = match frame {
                3 | 7 => {
                    late.push(number);
                    PresentOutcome::Presented(target + Duration(REFRESH))
                }
                // The platform reported it was never shown, without
                // saying why.
                10 => PresentOutcome::NotPresented,
                _ => PresentOutcome::Presented(target),
            };
            driver
                .report_frame(
                    number,
                    FrameReport {
                        work: Some(Duration(2_000_000)),
                        outcome,
                    },
                )
                .expect("the frame was held for its report")
        };
        recorder.record(&summary);
        summaries.push(summary);
    }

    let trace = Trace::from_snapshot(&OwnedSnapshot::from(runtime.snapshot()));
    let pacing = trace.pacing(tolerance);
    assert_eq!(pacing.unplaced_points(), 0, "every host time was placed");
    let counts = pacing.counts();
    assert_eq!(
        (
            counts.on_time,
            counts.late,
            counts.dropped,
            counts.skipped,
            counts.submitted_late,
            counts.unknown
        ),
        (7, 2, 2, 1, 0, 0),
        "on time, late, dropped, skipped, submitted late, unknown: {counts:?}"
    );
    assert_eq!(
        counts.missed(),
        4,
        "the superseded frame is not a miss; one not presented, for a reason not known, is"
    );
    let deadlines = pacing.deadlines();
    assert_eq!(
        (deadlines.met, deadlines.missed),
        (12, 0),
        "every frame's work was done before the deadline frameclock gave it, \
         including the late ones"
    );
    let basis = pacing.presentation_basis();
    assert_eq!(
        (basis.observed, basis.estimated, basis.unknown),
        (9, 0, 0),
        "every presentation time was reported by the platform"
    );
    assert_eq!(
        pacing.wasted().frames,
        3,
        "every frame never shown is wasted work, whatever the reason"
    );
    for number in late {
        let frame = pacing.frame(number).expect("the late frame was recorded");
        assert!(
            matches!(frame.outcome(), Outcome::Late { by, .. } if by == core::time::Duration::from_nanos(REFRESH)),
            "frame {number} was shown a refresh late: {:?}",
            frame.outcome()
        );
        assert!(
            frame.frame().is_some(),
            "its CPU work is the numbered frame"
        );
        let why = frame.why().expect("a late frame says why");
        assert_eq!(
            (
                why.cause,
                why.support,
                why.submit_reported,
                why.submitted_late,
                why.gpu_recorded,
                why.behind_previous,
                why.presented,
            ),
            (
                LateCause::AfterSubmit,
                Support::Unresolved,
                true,
                None,
                false,
                false,
                TimingBasis::Observed,
            ),
            "frame {number} was submitted in time, as frameclock reported, with no GPU work recorded: {why:?}"
        );
    }
    assert_eq!(
        (
            counts.late_support.observed,
            counts.late_support.inferred,
            counts.late_support.unresolved
        ),
        (0, 0, 2),
        "the trace stops short of why either late frame was late"
    );
    let superseded = summaries[5].frame_index;
    assert_eq!(
        pacing.frame(superseded).map(FramePacing::outcome),
        Some(Outcome::Dropped {
            reason: DropReason::Superseded
        }),
        "a superseded frame is skipped, not missed"
    );
    assert_eq!(
        pacing
            .frame(summaries[9].frame_index)
            .map(FramePacing::outcome),
        Some(Outcome::Dropped {
            reason: DropReason::SurfaceUnavailable
        })
    );

    // How good frameclock's predictions were, from the points it recorded,
    // against the same judgment made straight from the summaries.
    for (kind, time) in [
        (
            FramePointKind::Predicted,
            (|summary| summary.predicted_present) as fn(&FrameTimingSummary) -> Option<HostTime>,
        ),
        (FramePointKind::Target, |summary| summary.target_present),
        (FramePointKind::Expected, |summary| summary.expected_present),
    ] {
        let (mut within, mut shown_late) = (0, 0);
        for summary in &summaries {
            let (Some(at), Some(shown)) = (time(summary), summary.actual_present) else {
                continue;
            };
            if u128::from(shown.ticks().abs_diff(at.ticks())) > tolerance.as_nanos() {
                assert!(shown > at, "no frame is shown early");
                shown_late += 1;
            } else {
                within += 1;
            }
        }
        let error = pacing.prediction_error(kind);
        assert!(
            within + shown_late > 0,
            "the summaries carry {kind:?} times"
        );
        assert_eq!(
            (error.within(), error.late.count(), error.early.count()),
            (within, shown_late, 0),
            "{kind:?} predictions"
        );
    }
}
