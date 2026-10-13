// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simulated `FrameDriver` recorded into a real Footfall runtime, judged by
//! Footfall's pacing analysis.

use footfall_analysis::{Outcome, OutcomeCounts, OwnedSnapshot, Trace};
use footfall_core::InstrumentationLevel;
use footfall_runtime::Runtime;
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
fn pacing_judges_frames_from_what_frameclock_learned() {
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

    let mut late = Vec::new();
    let mut dropped = Vec::new();
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
        let submit =
            driver.submit_frame(active, FrameSubmission::reported(HostTime(now + 1_000_000)));
        assert_eq!(submit.summary, None, "frame {frame} waits for its report");

        let outcome = match frame {
            3 | 7 => {
                late.push(number);
                PresentOutcome::Presented(target + Duration(REFRESH))
            }
            10 => {
                dropped.push(number);
                PresentOutcome::NotPresented
            }
            _ => PresentOutcome::Presented(target),
        };
        let summary = driver
            .report_frame(
                number,
                FrameReport {
                    work: Some(Duration(2_000_000)),
                    outcome,
                },
            )
            .expect("the frame was held for its report");
        recorder.record(&summary);
    }

    let trace = Trace::from_snapshot(&OwnedSnapshot::from(runtime.snapshot()));
    let pacing = trace.pacing(core::time::Duration::from_millis(1));
    assert_eq!(pacing.unplaced_points(), 0, "every host time was placed");
    assert_eq!(
        pacing.counts(),
        OutcomeCounts {
            on_time: 9,
            late: 2,
            dropped: 1,
            submitted_late: 0,
            unknown: 0,
        }
    );
    for number in late {
        let frame = pacing.frame(number).expect("the late frame was recorded");
        assert_eq!(
            frame.outcome(),
            Outcome::Late {
                by: core::time::Duration::from_nanos(REFRESH)
            },
            "frame {number} was shown a refresh late"
        );
        assert!(
            frame.frame().is_some(),
            "its CPU work is the numbered frame"
        );
    }
    for number in dropped {
        assert_eq!(
            pacing.frame(number).map(|frame| frame.outcome()),
            Some(Outcome::Dropped)
        );
    }
}
