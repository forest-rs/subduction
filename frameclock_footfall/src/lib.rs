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
//! - [`Recorder::record`] turns each [`FrameTimingSummary`] into frame points:
//!   the platform's prediction, the commit deadline with the scheduler state
//!   the plan was made with, the target, sample, and expected present times,
//!   when the frame was submitted, and when it was shown, or that it never
//!   was and why.
//! - [`Recorder::sync`] records the host clock against Footfall's own, so
//!   analysis can place those times.
//!
//! Frame points stay in frameclock's host ticks on the wire, under
//! [`HOST_CLOCK`] and the platform adapter's [`Timebase`]. Footfall analysis
//! converts them, and `Trace::pacing` judges each frame from them.
//!
//! A summary often arrives frames after the frame's work, when the display
//! reports back. That is fine: frame points carry the frame's number, and the
//! scheduler state rides on the frame's own deadline point, so both describe
//! the frame they belong to whenever they are recorded.
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
    Args, ClockDesc, ClockReading, ContextHandle, CounterDesc, DropReason, FrameNumber, FramePoint,
    FramePointKind, FrameStream, TickPeriod, TimingBasis, Unit,
};
use frameclock::diagnostics::{DiagnosticsSink, FrameDropReason, FrameTimingBasis};
use frameclock::time::Timebase;
use frameclock::timing::PresentationTiming;
use frameclock::{ActiveFrame, FrameTimingSummary, HostTime};

/// The clock frameclock's [`HostTime`] values are read from: Mach absolute
/// time on Apple platforms, `QueryPerformanceCounter` on Windows,
/// `CLOCK_MONOTONIC` on Linux, `performance.now()` on the web.
///
/// Every frame point a [`Recorder`] emits is a reading of this clock. Use it
/// for other host-clock readings too, such as the host half of a GPU
/// timestamp pair, so Footfall converts them all through the same syncs.
pub static HOST_CLOCK: ClockDesc = ClockDesc::new("frameclock host");

static SUBMISSION_WORK: CounterDesc = CounterDesc::new("frameclock_submission_work", Unit::Nanos);
/// Footfall's name for why frameclock dropped a frame.
const fn drop_reason(reason: FrameDropReason) -> DropReason {
    match reason {
        FrameDropReason::Discarded => DropReason::Discarded,
        FrameDropReason::Superseded => DropReason::Superseded,
        FrameDropReason::MissedDeadline => DropReason::MissedDeadline,
        FrameDropReason::SurfaceUnavailable => DropReason::SurfaceUnavailable,
        FrameDropReason::OutputUnavailable => DropReason::OutputUnavailable,
        FrameDropReason::NotPresented => DropReason::NotPresented,
    }
}

/// How frameclock says the platform's predicted present time should be
/// read, as the `presentation_timing` arg on a predicted point.
const fn presentation_timing(timing: PresentationTiming) -> &'static str {
    match timing {
        PresentationTiming::Predictive => "predictive",
        PresentationTiming::Estimated => "estimated",
        PresentationTiming::PacingOnly => "pacing only",
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
    stream: FrameStream,
}

impl Recorder {
    /// Creates a recorder for host times in `timebase`.
    #[must_use]
    pub const fn new(timebase: Timebase) -> Self {
        Self {
            timebase,
            context: ContextHandle::ROOT,
            stream: FrameStream::DEFAULT,
        }
    }

    /// Returns this recorder, numbering frames in `stream`, such as one per
    /// window or surface when a program runs several frame drivers. Give a
    /// recreated driver, whose frame indices start again, a new generation
    /// of its stream.
    #[must_use]
    pub const fn with_stream(mut self, stream: FrameStream) -> Self {
        self.stream = stream;
        self
    }

    /// Returns the frame stream frames are numbered in.
    #[must_use]
    pub const fn stream(&self) -> FrameStream {
        self.stream
    }

    /// Returns this recorder, emitting frame markers and the submission-work
    /// histogram under `context`, such as the window or surface the driver
    /// paces.
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

    /// Returns the context frame markers and the histogram are emitted under.
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
        footfall_core::frame_begin_numbered(self.number(frame.plan().frame_index), self.context);
    }

    /// Marks the end of `frame`'s CPU work.
    ///
    /// Call it just before handing the frame to
    /// [`FrameDriver::submit_frame`](frameclock::FrameDriver::submit_frame) or
    /// [`FrameDriver::discard_frame`](frameclock::FrameDriver::discard_frame),
    /// so a frame whose work ends after its commit deadline shows as submitted
    /// late.
    pub fn end_frame(&self, frame: &ActiveFrame) {
        footfall_core::frame_end_numbered(self.number(frame.plan().frame_index), self.context);
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
    /// Each frame gets a frame point for every time frameclock knows, all in
    /// host ticks and numbered in the recorder's [`stream`](Self::stream):
    ///
    /// | Summary field | Frame point | Basis | Args |
    /// | --- | --- | --- | --- |
    /// | `predicted_present` | [`Predicted`](FramePointKind::Predicted) | | `presentation_timing` |
    /// | `commit_deadline` | [`Deadline`](FramePointKind::Deadline) | | the plan's scheduler state, below |
    /// | `target_present` | [`Target`](FramePointKind::Target) | | |
    /// | `sample_time` | [`Sampled`](FramePointKind::Sampled) | | |
    /// | `submitted_at` | [`Submitted`](FramePointKind::Submitted) | observed | `work_ns`, when reported |
    /// | `expected_present` | [`Expected`](FramePointKind::Expected) | | |
    /// | `actual_present` | [`Presented`](FramePointKind::Presented) | observed for `ActualPresent`, otherwise estimated | |
    /// | `drop_reason` | [`Dropped`](FramePointKind::Dropped), with the matching [`DropReason`] | | |
    ///
    /// The deadline point carries the scheduler state the frame was planned
    /// with, so it stays with that frame however late the summary arrives:
    /// `pipeline_depth`, `safety_margin_ns`, `frame_interval_ns`, and, when
    /// the scheduler sampled its state, `present_latency_ns`.
    ///
    /// Every drop is recorded with its reason, including frames the host
    /// discarded or superseded itself; Footfall's analysis counts only some
    /// reasons as misses. `NotPresented` stays
    /// [`DropReason::NotPresented`]: frameclock does not say whether a later
    /// frame replaced it.
    ///
    /// The submission work, when reported, also goes to the
    /// `frameclock_submission_work` histogram, in nanoseconds, under
    /// [`context`](Self::context).
    pub fn record(&self, summary: &FrameTimingSummary) {
        let number = self.number(summary.frame_index);
        if let Some(at) = summary.predicted_present {
            footfall_core::frame_point(&self.at(number, FramePointKind::Predicted, at).with_args(
                Args::none().with(
                    "presentation_timing",
                    presentation_timing(summary.presentation_timing),
                ),
            ));
        }
        let mut plan = Args::none()
            .with("pipeline_depth", summary.pipeline_depth)
            .with("safety_margin_ns", self.nanos(summary.safety_margin_ticks))
            .with(
                "frame_interval_ns",
                self.nanos(summary.frame_interval.ticks()),
            );
        if let Some(state) = summary.scheduler_state {
            plan = plan.with(
                "present_latency_ns",
                self.nanos(state.present_latency_ticks),
            );
        }
        footfall_core::frame_point(
            &self
                .at(number, FramePointKind::Deadline, summary.commit_deadline)
                .with_args(plan),
        );
        if let Some(at) = summary.target_present {
            footfall_core::frame_point(&self.at(number, FramePointKind::Target, at));
        }
        footfall_core::frame_point(&self.at(number, FramePointKind::Sampled, summary.sample_time));
        if let Some(at) = summary.submitted_at {
            let work = match summary.work {
                Some(work) => Args::none().with("work_ns", self.nanos(work.ticks())),
                None => Args::none(),
            };
            footfall_core::frame_point(
                &self
                    .at(number, FramePointKind::Submitted, at)
                    .with_basis(TimingBasis::Observed)
                    .with_args(work),
            );
        }
        if let Some(at) = summary.expected_present {
            footfall_core::frame_point(&self.at(number, FramePointKind::Expected, at));
        }
        if let Some(at) = summary.actual_present {
            let basis = match summary.timing_basis {
                FrameTimingBasis::ActualPresent => TimingBasis::Observed,
                _ => TimingBasis::Estimated,
            };
            footfall_core::frame_point(
                &self
                    .at(number, FramePointKind::Presented, at)
                    .with_basis(basis),
            );
        }
        if let Some(reason) = summary.drop_reason {
            footfall_core::frame_point(&FramePoint::dropped(number, drop_reason(reason)));
        }
        if let Some(work) = summary.work {
            footfall_core::histogram_record(
                &SUBMISSION_WORK,
                self.context,
                self.nanos(work.ticks()),
            );
        }
    }

    /// Returns frameclock's frame index as a frame of the recorder's stream.
    const fn number(&self, frame_index: u64) -> FrameNumber {
        self.stream.frame(frame_index)
    }

    fn at(&self, number: FrameNumber, kind: FramePointKind, at: HostTime) -> FramePoint {
        FramePoint::at(number, kind, &HOST_CLOCK, self.period(), at.ticks())
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
