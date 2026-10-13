// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Records `frameclock`'s frame timing in Footfall traces.
//!
//! [`Recorder`] turns what [`FrameDriver`](frameclock::FrameDriver) knows
//! about each frame into Footfall observations, so a trace can say which
//! frames missed the display, by how much, and what their work did
//! differently:
//!
//! - [`Recorder::begin_frame`] and [`Recorder::end_frame`] delimit the frame's
//!   CPU work with Footfall frame markers numbered by frameclock's frame
//!   index.
//! - [`Recorder::record`] turns each [`FrameTimingSummary`] into frame points
//!   (the commit deadline, the target present time, when the frame was shown,
//!   or that it never was) and gauges for the scheduler's adaptation.
//! - [`Recorder::sync`] records the host clock against Footfall's own, so
//!   analysis can place those times.
//!
//! Frame points stay in frameclock's host ticks on the wire, under
//! [`HOST_CLOCK`] and the platform adapter's [`Timebase`]. Footfall analysis
//! converts them, and `Trace::pacing` judges each frame from them.
//!
//! A summary often arrives frames after the frame's work, when the display
//! reports back. That is fine: frame points carry the frame's number, not the
//! time they were recorded.
//!
//! # Example
//!
//! ```
//! use frameclock::{
//!     Duration, FrameBeginResult, FrameDemand, FrameDriver, FrameOpportunity,
//!     FrameSubmission, HostTime, OutputId, SchedulerConfig, time::Timebase,
//! };
//! use frameclock_footfall::Recorder;
//!
//! // Use the platform adapter's timebase, such as `frameclock_apple::timebase()`.
//! let recorder = Recorder::new(Timebase::NANOS);
//! let mut driver = FrameDriver::new(SchedulerConfig::pacing_only());
//! driver.request(FrameDemand::ANIMATION);
//!
//! let now = HostTime(1_000_000);
//! recorder.sync(now);
//! let begin = driver.begin_frame(FrameOpportunity::pacing_only(
//!     now,
//!     Duration(16_666_667),
//!     OutputId(0),
//! ));
//! if let Some(summary) = &begin.resolved_feedback {
//!     recorder.record(summary);
//! }
//! if let FrameBeginResult::Ready(frame) = begin.result {
//!     recorder.begin_frame(&frame);
//!     // Build and submit the frame.
//!     recorder.end_frame(&frame);
//!     let submit = driver.submit_frame(frame, FrameSubmission::new(HostTime(2_000_000), None));
//!     if let Some(summary) = &submit.summary {
//!         recorder.record(summary);
//!     }
//! }
//! ```
//!
//! Without an installed Footfall backend every call does nothing.

#![no_std]

use footfall_core::{
    ClockDesc, ClockReading, ContextHandle, CounterDesc, FramePoint, FramePointKind, TickPeriod,
    Unit,
};
use frameclock::diagnostics::{DiagnosticsSink, FrameDropReason};
use frameclock::time::Timebase;
use frameclock::{ActiveFrame, FrameTimingSummary, HostTime};

/// The clock frameclock's [`HostTime`] values are read from: Mach absolute
/// time on Apple platforms, `QueryPerformanceCounter` on Windows,
/// `CLOCK_MONOTONIC` on Linux, `performance.now()` on the web.
///
/// Every frame point a [`Recorder`] emits is a reading of this clock. Use it
/// for other host-clock readings too, such as the host half of a GPU
/// timestamp pair, so Footfall converts them all through the same syncs.
pub static HOST_CLOCK: ClockDesc = ClockDesc::new("frameclock host");

static PIPELINE_DEPTH: CounterDesc = CounterDesc::count("frameclock_pipeline_depth");
static SAFETY_MARGIN: CounterDesc = CounterDesc::new("frameclock_safety_margin", Unit::Nanos);
static FRAME_INTERVAL: CounterDesc = CounterDesc::new("frameclock_frame_interval", Unit::Nanos);
static PRESENT_LATENCY: CounterDesc = CounterDesc::new("frameclock_present_latency", Unit::Nanos);
static SUBMISSION_WORK: CounterDesc = CounterDesc::new("frameclock_submission_work", Unit::Nanos);
static DISCARDED: CounterDesc = CounterDesc::count("frameclock_dropped_discarded");
static SUPERSEDED: CounterDesc = CounterDesc::count("frameclock_dropped_superseded");
static MISSED_DEADLINE: CounterDesc = CounterDesc::count("frameclock_dropped_missed_deadline");
static SURFACE_UNAVAILABLE: CounterDesc =
    CounterDesc::count("frameclock_dropped_surface_unavailable");
static OUTPUT_UNAVAILABLE: CounterDesc =
    CounterDesc::count("frameclock_dropped_output_unavailable");
static NOT_PRESENTED: CounterDesc = CounterDesc::count("frameclock_dropped_not_presented");

const fn dropped_counter(reason: FrameDropReason) -> &'static CounterDesc {
    match reason {
        FrameDropReason::Discarded => &DISCARDED,
        FrameDropReason::Superseded => &SUPERSEDED,
        FrameDropReason::MissedDeadline => &MISSED_DEADLINE,
        FrameDropReason::SurfaceUnavailable => &SURFACE_UNAVAILABLE,
        FrameDropReason::OutputUnavailable => &OUTPUT_UNAVAILABLE,
        FrameDropReason::NotPresented => &NOT_PRESENTED,
    }
}

/// Records one [`FrameDriver`](frameclock::FrameDriver)'s frames in Footfall.
///
/// Construct it with the [`Timebase`] of the platform adapter whose
/// [`HostTime`] values the driver uses. It holds no state beyond that, so one
/// recorder can be shared, or kept in a `static`.
///
/// It also implements [`DiagnosticsSink`] for integrations that call the
/// scheduler directly: it records each
/// [`on_frame_timing_summary`](DiagnosticsSink::on_frame_timing_summary).
#[derive(Clone, Copy, Debug)]
pub struct Recorder {
    timebase: Timebase,
    context: ContextHandle,
}

impl Recorder {
    /// Creates a recorder for host times in `timebase`.
    #[must_use]
    pub const fn new(timebase: Timebase) -> Self {
        Self {
            timebase,
            context: ContextHandle::ROOT,
        }
    }

    /// Returns this recorder, emitting frame markers and gauges under
    /// `context`, such as the window or surface the driver paces.
    #[must_use]
    pub const fn with_context(mut self, context: ContextHandle) -> Self {
        self.context = context;
        self
    }

    /// Returns how long one host tick lasts, as Footfall records it.
    #[must_use]
    pub const fn period(&self) -> TickPeriod {
        TickPeriod::new(self.timebase.numer, self.timebase.denom)
    }

    /// Returns the context frame markers and gauges are emitted under.
    #[must_use]
    pub const fn context(&self) -> ContextHandle {
        self.context
    }

    /// Records that the host clock reads `now`.
    ///
    /// Read the host clock and call this right away, such as at the start
    /// of each frame. Footfall converts every host time through these syncs,
    /// so frame points recorded before the first one cannot be placed until
    /// it arrives, and analysis reports them as unplaced.
    pub fn sync(&self, now: HostTime) {
        footfall_core::clock_sync(self.reading(now));
    }

    /// Marks the start of `frame`'s CPU work, numbered by its frame index.
    ///
    /// Call it once [`FrameDriver::begin_frame`](frameclock::FrameDriver::begin_frame)
    /// returns the frame ready.
    pub fn begin_frame(&self, frame: &ActiveFrame) {
        footfall_core::frame_begin_numbered(frame.plan().frame_index, self.context);
    }

    /// Marks the end of `frame`'s CPU work.
    ///
    /// Call it just before handing the frame to
    /// [`FrameDriver::submit_frame`](frameclock::FrameDriver::submit_frame) or
    /// [`FrameDriver::discard_frame`](frameclock::FrameDriver::discard_frame),
    /// so a frame whose work ends after its commit deadline shows as submitted
    /// late.
    pub fn end_frame(&self, frame: &ActiveFrame) {
        footfall_core::frame_end_numbered(frame.plan().frame_index, self.context);
    }

    /// Records what `summary` says about its frame.
    ///
    /// Pass every summary the driver returns: from
    /// [`submit_frame`](frameclock::FrameDriver::submit_frame),
    /// [`report_frame`](frameclock::FrameDriver::report_frame),
    /// [`FrameBegin::resolved_feedback`](frameclock::FrameBegin::resolved_feedback),
    /// [`FrameBeginResult::Expired`](frameclock::FrameBeginResult::Expired), and
    /// [`discard_frame`](frameclock::FrameDriver::discard_frame).
    ///
    /// A frame the display was meant to show gets its commit deadline, its
    /// target present time when frameclock has one, and either when it was
    /// shown or that it was dropped. A frame the host discarded or replaced
    /// itself was never meant to be shown, so it gets no frame points, only
    /// its drop count.
    ///
    /// Alongside the frame points it sets these counters, under
    /// [`context`](Self::context):
    ///
    /// | Counter | Kind | Unit | From |
    /// | --- | --- | --- | --- |
    /// | `frameclock_pipeline_depth` | gauge | count | `pipeline_depth` |
    /// | `frameclock_safety_margin` | gauge | nanos | `safety_margin_ticks` |
    /// | `frameclock_frame_interval` | gauge | nanos | `frame_interval` |
    /// | `frameclock_present_latency` | gauge | nanos | `scheduler_state`, when sampled |
    /// | `frameclock_submission_work` | histogram | nanos | `work`, when reported |
    /// | `frameclock_dropped_<reason>` | counter | count | `drop_reason`, in snake case |
    ///
    /// Gauges are set as of when the summary arrives, which may be frames
    /// after the frame ran.
    pub fn record(&self, summary: &FrameTimingSummary) {
        let number = summary.frame_index;
        let host_intended = matches!(
            summary.drop_reason,
            Some(FrameDropReason::Discarded | FrameDropReason::Superseded)
        );
        if !host_intended {
            self.point(number, FramePointKind::Deadline, summary.commit_deadline);
            if let Some(target) = summary.target_present {
                self.point(number, FramePointKind::Target, target);
            }
            if let Some(shown) = summary.actual_present {
                self.point(number, FramePointKind::Presented, shown);
            }
            if summary.drop_reason.is_some() {
                footfall_core::frame_point(&FramePoint::dropped(number));
            }
        }
        if let Some(reason) = summary.drop_reason {
            footfall_core::counter_add(dropped_counter(reason), self.context, 1);
        }

        footfall_core::gauge_set(
            &PIPELINE_DEPTH,
            self.context,
            u64::from(summary.pipeline_depth),
        );
        footfall_core::gauge_set(
            &SAFETY_MARGIN,
            self.context,
            self.nanos(summary.safety_margin_ticks),
        );
        footfall_core::gauge_set(
            &FRAME_INTERVAL,
            self.context,
            self.nanos(summary.frame_interval.ticks()),
        );
        if let Some(state) = summary.scheduler_state {
            footfall_core::gauge_set(
                &PRESENT_LATENCY,
                self.context,
                self.nanos(state.present_latency_ticks),
            );
        }
        if let Some(work) = summary.work {
            footfall_core::histogram_record(
                &SUBMISSION_WORK,
                self.context,
                self.nanos(work.ticks()),
            );
        }
    }

    fn point(&self, number: u64, kind: FramePointKind, at: HostTime) {
        footfall_core::frame_point(&FramePoint::at(
            number,
            kind,
            &HOST_CLOCK,
            self.period(),
            at.ticks(),
        ));
    }

    const fn reading(&self, at: HostTime) -> ClockReading {
        ClockReading::new(&HOST_CLOCK, at.ticks(), self.period())
    }

    const fn nanos(&self, ticks: u64) -> u64 {
        self.timebase.ticks_to_nanos(ticks)
    }
}

impl DiagnosticsSink for Recorder {
    fn on_frame_timing_summary(&mut self, event: &FrameTimingSummary) {
        self.record(event);
    }
}
