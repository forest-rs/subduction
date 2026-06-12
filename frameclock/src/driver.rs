// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained frame lifecycle state.
//!
//! This module owns pending frame demand, queued frame plans, feedback
//! observation, and frameclock timing summaries. It explicitly does not own
//! platform event loops, app timers, redraw requests, renderer submission, or
//! surface lifecycle.

use crate::demand::FrameDemand;
use crate::diagnostics::{
    FrameDropEvent, FrameDropReason, FramePlanEvent, FrameTickEvent, FrameTimingSummary,
    FrameTimingSummaryBuilder, PresentFeedbackEvent, SchedulerStateEvent, SubmitEvent,
};
use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::time::HostTime;
use crate::timing::{FrameOpportunity, FramePlan, FrameTick, PresentFeedback, PresentHints};

/// A queued scheduler plan paired with the platform facts used to make it.
///
/// `FrameDriver` stores this internally while waiting for
/// [`FramePlan::frame_start`]. Hosts can inspect the current value through
/// [`FrameDriver::pending_frame`], but normal frame preparation receives an
/// [`ActiveFrame`] from [`FrameBeginResult::Ready`] instead.
///
/// The tick and hints must stay with the plan because diagnostics and
/// presentation feedback should be resolved against the same platform
/// opportunity, desired-present time, and deadline facts that were active when
/// the scheduler made the decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlannedFrame {
    /// Platform tick that produced [`Self::plan`].
    pub tick: FrameTick,
    /// Scheduler-selected frame timing.
    pub plan: FramePlan,
    /// Presentation constraints paired with [`Self::plan`].
    pub hints: PresentHints,
    /// Scheduler safety margin in host-time ticks when [`Self::plan`] was
    /// created.
    pub safety_margin_ticks: u64,
}

impl PlannedFrame {
    /// Creates a planned frame from a [`FrameTick`], [`FramePlan`], and
    /// matching [`PresentHints`].
    ///
    /// Most hosts do not call this directly; [`FrameDriver`] creates
    /// `PlannedFrame` values when it consumes pending demand.
    #[inline]
    #[must_use]
    pub const fn new(
        tick: FrameTick,
        plan: FramePlan,
        hints: PresentHints,
        safety_margin_ticks: u64,
    ) -> Self {
        Self {
            tick,
            plan,
            hints,
            safety_margin_ticks,
        }
    }
}

/// A planned frame returned to the host for preparation.
///
/// `FrameDriver::begin_frame` returns this inside
/// [`FrameBeginResult::Ready`]. It is the normal host handle for frame work:
/// use [`ActiveFrame::sample_time`] or [`ActiveFrame::plan`] to prepare app
/// state, then pass the same value to [`FrameDriver::submit_frame`] or
/// [`FrameDriver::discard_frame`].
///
/// The handle carries the scheduler plan plus the original planning facts
/// needed to resolve feedback and build a [`FrameTimingSummary`] when the host
/// submits the frame.
///
/// Treat an `ActiveFrame` as a single-use lifecycle token. A host should either
/// pass it to [`FrameDriver::submit_frame`] after renderer submission or to
/// [`FrameDriver::discard_frame`] / [`FrameDriver::discard_frame_with_reason`]
/// if no submission will happen. Dropping the Rust value without reporting it
/// loses diagnostics, but it does not update scheduler feedback.
#[derive(Debug)]
pub struct ActiveFrame {
    planned: PlannedFrame,
    build_start: HostTime,
}

impl ActiveFrame {
    const fn new(planned: PlannedFrame, build_start: HostTime) -> Self {
        Self {
            planned,
            build_start,
        }
    }

    /// Returns the platform tick that produced this frame's plan.
    #[must_use]
    pub const fn tick(&self) -> FrameTick {
        self.planned.tick
    }

    /// Returns the scheduler-selected plan.
    #[must_use]
    pub const fn plan(&self) -> FramePlan {
        self.planned.plan
    }

    /// Returns the presentation hints paired with this frame.
    #[must_use]
    pub const fn hints(&self) -> PresentHints {
        self.planned.hints
    }

    /// Returns the scheduler safety margin in host-time ticks used for this
    /// frame's plan.
    #[must_use]
    pub const fn safety_margin_ticks(&self) -> u64 {
        self.planned.safety_margin_ticks
    }

    /// Returns the host time at which the driver handed this frame to the host.
    #[must_use]
    pub const fn build_start(&self) -> HostTime {
        self.build_start
    }

    /// Returns the time application state should be sampled for.
    #[must_use]
    pub const fn sample_time(&self) -> HostTime {
        self.planned.plan.sample_time
    }
}

/// What presentation evidence is available for a submitted frame.
///
/// This tells [`FrameDriver::submit_frame`] whether it can return a complete
/// [`FrameTimingSummary`] immediately or should wait for a later
/// [`FrameTick::prev_actual_present`] value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PresentationObservation {
    /// The platform will not report an actual-present timestamp for this
    /// frame.
    ///
    /// The driver returns a complete summary immediately using commit timing
    /// as pacing evidence.
    Unavailable,
    /// The platform already reported when this frame was actually presented.
    ///
    /// The driver returns a complete summary immediately using this timestamp
    /// as presentation feedback.
    Actual(HostTime),
    /// The platform reports actual presentation on a later frame tick.
    ///
    /// The driver stores the submission and returns the summary in
    /// [`FrameBegin::resolved_feedback`] the next time
    /// [`FrameDriver::begin_frame`] is called.
    Deferred,
}

impl PresentationObservation {
    /// Returns the actual-present timestamp carried by this observation, if
    /// one is already known.
    #[must_use]
    pub const fn actual_present(self) -> Option<HostTime> {
        match self {
            Self::Actual(actual_present) => Some(actual_present),
            Self::Unavailable | Self::Deferred => None,
        }
    }
}

/// Submission facts passed to [`FrameDriver::submit_frame`].
///
/// Construct this after the host has submitted, committed, or otherwise handed
/// the [`ActiveFrame`] to its renderer/platform presentation path. It contains
/// the submission timestamp and the kind of presentation evidence the platform
/// can provide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameSubmission {
    /// Host time when the frame was submitted or committed.
    ///
    /// Pass the timestamp closest to the point where the frame leaves app-side
    /// control and enters the renderer, compositor, or presentation backend.
    pub submitted_at: HostTime,
    /// Presentation evidence available for this submission.
    pub presentation: PresentationObservation,
}

impl FrameSubmission {
    /// Creates submission facts for [`FrameDriver::submit_frame`].
    ///
    /// Passing `Some(actual_present)` records an immediate actual-present
    /// timestamp. Passing `None` means the platform will not provide actual
    /// present feedback for this frame. Use [`Self::deferred`] when a later
    /// frame tick will carry the actual-present timestamp.
    #[inline]
    #[must_use]
    pub const fn new(submitted_at: HostTime, actual_present: Option<HostTime>) -> Self {
        Self {
            submitted_at,
            presentation: match actual_present {
                Some(actual_present) => PresentationObservation::Actual(actual_present),
                None => PresentationObservation::Unavailable,
            },
        }
    }

    /// Creates submission facts for platforms that report actual present on a
    /// later frame tick.
    ///
    /// The summary for this frame is returned by the next
    /// [`FrameDriver::begin_frame`] call in [`FrameBegin::resolved_feedback`].
    #[inline]
    #[must_use]
    pub const fn deferred(submitted_at: HostTime) -> Self {
        Self {
            submitted_at,
            presentation: PresentationObservation::Deferred,
        }
    }

    /// Returns the immediate actual-present timestamp, if one is already
    /// available.
    #[inline]
    #[must_use]
    pub const fn actual_present(self) -> Option<HostTime> {
        self.presentation.actual_present()
    }
}

/// Result returned by [`FrameDriver::submit_frame`].
#[derive(Clone, Debug, PartialEq)]
pub struct FrameSubmitResult {
    /// Complete timing summary, when feedback was resolved immediately.
    ///
    /// This is `None` when [`FrameSubmission::deferred`] was used; in that case
    /// the summary is returned later through [`FrameBegin::resolved_feedback`].
    pub summary: Option<FrameTimingSummary>,
    /// Whether the driver is waiting for a later actual-present timestamp for
    /// this submitted frame.
    pub awaiting_actual_present: bool,
}

impl FrameSubmitResult {
    fn complete(summary: FrameTimingSummary) -> Self {
        Self {
            summary: Some(summary),
            awaiting_actual_present: false,
        }
    }

    const fn deferred() -> Self {
        Self {
            summary: None,
            awaiting_actual_present: true,
        }
    }
}

/// Result returned by [`FrameDriver::begin_frame`].
#[derive(Debug)]
pub enum FrameBeginResult {
    /// No demand or queued frame is waiting.
    Idle,
    /// A frame has been planned, but its selected start time has not arrived.
    ///
    /// Hosts should mirror this time into their event-loop or timer queue and
    /// call [`FrameDriver::begin_frame`] again from the wake/redraw path.
    WaitUntil(HostTime),
    /// A planned frame is ready for application update and rendering.
    ///
    /// Prepare the frame, then pass this [`ActiveFrame`] to
    /// [`FrameDriver::submit_frame`] or [`FrameDriver::discard_frame`].
    Ready(ActiveFrame),
    /// A queued frame missed its commit deadline before the host released it.
    ///
    /// The driver has dropped the planned frame without feeding scheduler
    /// feedback. Hosts should record the summary and request fresh demand if
    /// the work is still needed.
    Expired(FrameTimingSummary),
}

/// Result returned by [`FrameDriver::begin_frame`].
///
/// [`Self::result`] is the lifecycle action for the current frame opportunity.
/// [`Self::resolved_feedback`] carries the timing summary for a previous
/// deferred submission when the current tick supplied
/// [`FrameTick::prev_actual_present`].
#[derive(Debug)]
pub struct FrameBegin {
    /// Summary resolved from a previous [`FrameSubmission::deferred`] call.
    pub resolved_feedback: Option<FrameTimingSummary>,
    /// Current begin-frame lifecycle result.
    pub result: FrameBeginResult,
}

enum DriverBeginResult {
    Idle,
    WaitUntil(HostTime),
    Ready(PlannedFrame),
    Expired(PlannedFrame),
}

/// Driver for pending demand, queued frame plans, and frame feedback.
///
/// `FrameDriver` is a small retained layer above [`Scheduler`]. Hosts call
/// [`request`](Self::request) when app-visible frame demand arrives, then call
/// [`begin_frame`](Self::begin_frame) from a platform redraw/tick
/// opportunity. If the scheduler chooses a future [`FramePlan::frame_start`],
/// the driver keeps that plan queued and returns [`FrameBeginResult::WaitUntil`].
///
/// Submit completed frames with [`submit_frame`](Self::submit_frame) so
/// `frameclock` can observe feedback and produce the frame timing summary.
/// Discard frames that will not be submitted with
/// [`discard_frame`](Self::discard_frame).
///
/// # Lifecycle
///
/// 1. Call [`request`](Self::request) with non-empty [`FrameDemand`] when input,
///    animation, layout, or other app-visible work needs a frame.
/// 2. Call [`begin_frame`](Self::begin_frame) from a redraw/tick
///    opportunity. Record [`FrameBegin::resolved_feedback`] if it is
///    present; this is the summary for a previous deferred submission.
/// 3. Inspect [`FrameBegin::result`]. If it is
///    [`FrameBeginResult::WaitUntil`], mirror that time into the host timer
///    queue and sleep. If it is [`FrameBeginResult::Expired`], record the
///    returned drop summary and request fresh demand if the work is still
///    needed. If it is [`FrameBeginResult::Idle`], wait for more demand.
/// 4. If it is [`FrameBeginResult::Ready`], sample app state at
///    [`ActiveFrame::sample_time`], then either submit it with
///    [`submit_frame`](Self::submit_frame) or drop it with
///    [`discard_frame`](Self::discard_frame).
/// 5. After submit or discard, call
///    [`has_pending_demand`](Self::has_pending_demand). If it is true, request
///    another redraw so weaker retained demand can be planned on a fresh turn.
///
/// This keeps demand preemption, feedback observation, and summary construction
/// inside `frameclock` while leaving timer queues, event-loop wake mechanics,
/// renderer submission, and native surface lifecycle in the host.
///
/// # Queued plans
///
/// Queued plans are replayed as decided when they are released inside their
/// valid work window. If a plan is created at tick `T` with a future frame
/// start, a later on-time wake releases the same [`FramePlan`], [`FrameTick`],
/// and [`PresentHints`]. The returned [`ActiveFrame`] records the later wake
/// time as [`ActiveFrame::build_start`], so diagnostics and budget
/// calculations can still see wake latency.
///
/// If the host does not release a queued plan until after
/// [`FramePlan::commit_deadline`], the driver drops the plan and returns
/// [`FrameBeginResult::Expired`] with a [`FrameDropReason::MissedDeadline`]
/// summary instead of handing out a renderable frame.
#[derive(Debug)]
pub struct FrameDriver {
    scheduler: Scheduler,
    pending_demand: FrameDemand,
    pending_frame: Option<PlannedFrame>,
    pending_feedback: Option<DeferredFrameFeedback>,
}

#[derive(Debug)]
struct DeferredFrameFeedback {
    planned: PlannedFrame,
    build_start: HostTime,
    submitted_at: HostTime,
}

impl DeferredFrameFeedback {
    const fn new(frame: ActiveFrame, submitted_at: HostTime) -> Self {
        Self {
            planned: frame.planned,
            build_start: frame.build_start,
            submitted_at,
        }
    }
}

impl FrameDriver {
    /// Creates a driver with a new [`Scheduler`] using `config`.
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self::from_scheduler(Scheduler::new(config))
    }

    /// Creates a driver around an existing [`Scheduler`].
    #[must_use]
    pub const fn from_scheduler(scheduler: Scheduler) -> Self {
        Self {
            scheduler,
            pending_demand: FrameDemand::NONE,
            pending_frame: None,
            pending_feedback: None,
        }
    }

    /// Returns the underlying scheduler.
    #[must_use]
    pub const fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Returns demand that has not yet been planned into a frame.
    #[must_use]
    pub const fn pending_demand(&self) -> FrameDemand {
        self.pending_demand
    }

    /// Returns whether demand is waiting for another planning turn.
    ///
    /// Hosts usually request another redraw after consuming a ready
    /// [`PlannedFrame`] when this returns `true`.
    #[must_use]
    pub const fn has_pending_demand(&self) -> bool {
        !self.pending_demand.is_empty()
    }

    /// Returns the currently queued frame, if any.
    #[must_use]
    pub const fn pending_frame(&self) -> Option<PlannedFrame> {
        self.pending_frame
    }

    /// Returns the frame-start time for the queued plan, if any.
    #[must_use]
    pub const fn next_frame_start(&self) -> Option<HostTime> {
        match self.pending_frame {
            Some(frame) => Some(frame.plan.frame_start),
            None => None,
        }
    }

    /// Adds app-visible frame demand for a future [`begin_frame`](Self::begin_frame) call.
    ///
    /// Call this when input, animation, layout, resize, timers, or background
    /// visual work make a frame necessary. The driver keeps the demand until it
    /// can plan a frame for a later [`FrameOpportunity`].
    ///
    /// If a frame is already queued, stronger demand invalidates that queued
    /// plan so the next [`begin_frame`](Self::begin_frame) call replans with the
    /// combined old and new demand. Weaker or equal demand never mutates the
    /// queued plan because the plan's timing was selected for its original
    /// demand. Uncovered weaker demand is retained for a later planning turn.
    pub fn request(&mut self, demand: FrameDemand) {
        if demand.is_empty() {
            return;
        }

        if let Some(frame) = self.pending_frame {
            if demand.preempts(frame.plan.demand) {
                self.pending_demand.insert(frame.plan.demand | demand);
                self.pending_frame = None;
            } else if !frame.plan.demand.contains(demand) {
                let uncovered =
                    FrameDemand::from_bits_truncate(demand.bits() & !frame.plan.demand.bits());
                self.pending_demand.insert(uncovered);
            }
            return;
        }

        self.pending_demand.insert(demand);
    }

    /// Begins a frame, waits for a queued frame start, or reports idleness.
    ///
    /// This is the normal host entry point for converting a platform frame
    /// opportunity into frame work. It is the preferred public method that
    /// consumes pending demand into a ready frame. If the selected frame start
    /// is still in the future, this stores the planned frame internally and
    /// returns [`FrameBeginResult::WaitUntil`] with the wake time the host
    /// should mirror into its timer queue.
    ///
    /// When a queued frame becomes ready before its commit deadline, the
    /// returned [`ActiveFrame`] carries the original planning tick and plan. Its
    /// [`ActiveFrame::build_start`] is the current opportunity time that
    /// released the queued frame. If the queued frame is released after its
    /// commit deadline, the driver returns [`FrameBeginResult::Expired`]
    /// instead and records a dropped-frame summary.
    #[must_use]
    pub fn begin_frame(&mut self, opportunity: FrameOpportunity) -> FrameBegin {
        let resolved_feedback =
            self.resolve_deferred_feedback(opportunity.tick.prev_actual_present);
        let result = match self.take_next(opportunity) {
            DriverBeginResult::Idle => FrameBeginResult::Idle,
            DriverBeginResult::WaitUntil(frame_start) => FrameBeginResult::WaitUntil(frame_start),
            DriverBeginResult::Ready(frame) => {
                FrameBeginResult::Ready(ActiveFrame::new(frame, opportunity.tick.now))
            }
            DriverBeginResult::Expired(frame) => {
                let frame = ActiveFrame::new(frame, opportunity.tick.now);
                FrameBeginResult::Expired(
                    self.discard_frame_with_reason(frame, FrameDropReason::MissedDeadline),
                )
            }
        };
        FrameBegin {
            resolved_feedback,
            result,
        }
    }

    /// Reports that an active frame was submitted and returns its timing
    /// result.
    ///
    /// For [`PresentationObservation::Unavailable`] and
    /// [`PresentationObservation::Actual`], this constructs presentation
    /// feedback, feeds it to the scheduler, and returns a complete summary in
    /// [`FrameSubmitResult::summary`].
    ///
    /// For [`PresentationObservation::Deferred`], this stores the submission
    /// until the next [`begin_frame`](Self::begin_frame) call, then resolves it
    /// with that tick's [`FrameTick::prev_actual_present`]. This keeps Apple-
    /// style previous-frame present feedback inside `frameclock` while keeping
    /// one public submit method.
    #[must_use]
    pub fn submit_frame(
        &mut self,
        frame: ActiveFrame,
        submission: FrameSubmission,
    ) -> FrameSubmitResult {
        match submission.presentation {
            PresentationObservation::Deferred => {
                debug_assert!(
                    self.pending_feedback.is_none(),
                    "deferred feedback should resolve before another deferred submission"
                );
                self.pending_feedback =
                    Some(DeferredFrameFeedback::new(frame, submission.submitted_at));
                FrameSubmitResult::deferred()
            }
            PresentationObservation::Unavailable | PresentationObservation::Actual(_) => {
                let feedback = PresentFeedback::new(
                    &frame.plan(),
                    frame.build_start(),
                    submission.submitted_at,
                    submission.actual_present(),
                );
                FrameSubmitResult::complete(self.finish_submitted_frame(
                    frame.planned,
                    frame.build_start,
                    submission.submitted_at,
                    &feedback,
                ))
            }
        }
    }

    fn resolve_deferred_feedback(
        &mut self,
        actual_present: Option<HostTime>,
    ) -> Option<FrameTimingSummary> {
        let pending = self.pending_feedback.take()?;
        let feedback = PresentFeedback::new(
            &pending.planned.plan,
            pending.build_start,
            pending.submitted_at,
            actual_present,
        );
        Some(self.finish_submitted_frame(
            pending.planned,
            pending.build_start,
            pending.submitted_at,
            &feedback,
        ))
    }

    fn finish_submitted_frame(
        &mut self,
        planned: PlannedFrame,
        build_start: HostTime,
        submitted_at: HostTime,
        feedback: &PresentFeedback,
    ) -> FrameTimingSummary {
        let tick_event = FrameTickEvent::from(&planned.tick);
        let plan = planned.plan;
        let plan_event = FramePlanEvent::new(&plan, planned.safety_margin_ticks);
        let submit_event = SubmitEvent {
            frame_index: plan.frame_index,
            submitted_at,
            expected_present: feedback.expected_present,
        };
        debug_assert_eq!(
            feedback.build_start, build_start,
            "submitted frame summary must use feedback for the same build start"
        );
        let feedback_event = PresentFeedbackEvent::new(plan.frame_index, feedback);

        self.scheduler.observe(feedback);

        let state_event = SchedulerStateEvent {
            state: self.scheduler.state(),
        };
        let mut summary = FrameTimingSummaryBuilder::from_tick_and_plan(&tick_event, &plan_event);
        summary
            .record_submit(&submit_event)
            .record_present_feedback(&feedback_event)
            .record_scheduler_state(&state_event);
        summary
            .finish()
            .expect("driver-created tick and plan describe the same frame")
    }

    /// Drops an active frame without feeding scheduler feedback.
    ///
    /// This returns a [`FrameTimingSummary`] with
    /// [`FrameTimingSummary::drop_reason`] set to
    /// [`FrameDropReason::Discarded`]. Scheduler adaptation state is sampled
    /// for diagnostics, but [`Scheduler::observe`] is not called.
    #[must_use]
    pub fn discard_frame(&mut self, frame: ActiveFrame) -> FrameTimingSummary {
        self.discard_frame_with_reason(frame, FrameDropReason::Discarded)
    }

    /// Drops an active frame with an explicit reason and without feeding
    /// scheduler feedback.
    ///
    /// Dropping a frame is a host lifecycle fact, not presentation feedback.
    /// The returned summary is useful for traces and devtools, but this method
    /// deliberately does not call [`Scheduler::observe`].
    #[must_use]
    pub fn discard_frame_with_reason(
        &mut self,
        frame: ActiveFrame,
        reason: FrameDropReason,
    ) -> FrameTimingSummary {
        let tick_event = FrameTickEvent::from(&frame.tick());
        let plan = frame.plan();
        let plan_event = FramePlanEvent::new(&plan, frame.safety_margin_ticks());
        let drop_event = FrameDropEvent::new(&plan, reason);
        let state_event = SchedulerStateEvent {
            state: self.scheduler.state(),
        };

        let mut summary = FrameTimingSummaryBuilder::from_tick_and_plan(&tick_event, &plan_event);
        summary
            .record_frame_drop(&drop_event)
            .record_scheduler_state(&state_event);
        summary
            .finish()
            .expect("driver-created tick and plan describe the same frame")
    }

    /// Returns the next driver lifecycle result for a frame opportunity.
    ///
    /// This plans pending demand when needed, retains future-start plans, and
    /// reports an expired result instead of returning a frame whose commit
    /// deadline has already passed.
    fn take_next(&mut self, opportunity: FrameOpportunity) -> DriverBeginResult {
        let tick = opportunity.tick;
        if let Some(frame) = self.pending_frame {
            if tick.now < frame.plan.frame_start {
                return DriverBeginResult::WaitUntil(frame.plan.frame_start);
            }

            self.pending_frame = None;
            if tick.now > frame.plan.commit_deadline {
                return DriverBeginResult::Expired(frame);
            }
            return DriverBeginResult::Ready(frame);
        }

        if self.pending_demand.is_empty() {
            return DriverBeginResult::Idle;
        }

        let demand = self.pending_demand;
        self.pending_demand = FrameDemand::NONE;
        let plan = self.scheduler.plan(opportunity, demand);
        let frame = PlannedFrame::new(
            tick,
            plan,
            opportunity.hints,
            self.scheduler.safety_margin_ticks(),
        );
        if tick.now < frame.plan.frame_start {
            let frame_start = frame.plan.frame_start;
            self.pending_frame = Some(frame);
            DriverBeginResult::WaitUntil(frame_start)
        } else if tick.now > frame.plan.commit_deadline {
            DriverBeginResult::Expired(frame)
        } else {
            DriverBeginResult::Ready(frame)
        }
    }

    /// Drops the queued frame and returns whether one existed.
    ///
    /// This does not clear [`pending_demand`](Self::pending_demand). Demand
    /// retained behind the queued frame remains available for the next
    /// planning turn.
    pub fn clear_pending_frame(&mut self) -> bool {
        self.pending_frame.take().is_some()
    }

    /// Feeds presentation feedback to the underlying [`Scheduler`].
    pub fn observe(&mut self, feedback: &PresentFeedback) {
        self.scheduler.observe(feedback);
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::{
        FramePlanEvent, FrameTickEvent, FrameTimingBasis, FrameTimingSummaryBuilder,
        PresentFeedbackEvent, SchedulerStateEvent, SubmitEvent,
    };
    use crate::output::OutputId;
    use crate::time::Duration;
    use crate::timing::{DisplayTiming, PresentFeedback, PresentationTiming};

    use super::*;

    const REFRESH_INTERVAL: Duration = Duration(100);
    const START_MARGIN: Duration = Duration(10);

    fn driver() -> FrameDriver {
        let mut config = SchedulerConfig::predictive();
        config.initial_depth = 1;
        config.minimum_frame_start_margin = START_MARGIN;
        FrameDriver::new(config)
    }

    fn tick(now: u64, frame_index: u64) -> FrameTick {
        FrameTick {
            now: HostTime(now),
            predicted_present: None,
            refresh_interval: Some(REFRESH_INTERVAL.ticks()),
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    fn hints(deadline: u64) -> PresentHints {
        PresentHints::pacing_only(HostTime(deadline))
    }

    fn opportunity(now: u64, frame_index: u64) -> FrameOpportunity {
        FrameOpportunity::new(
            tick(now, frame_index),
            hints(now + REFRESH_INTERVAL.ticks()),
            DisplayTiming::fixed(REFRESH_INTERVAL),
        )
    }

    fn predictive_opportunity(
        now: u64,
        frame_index: u64,
        desired_present: u64,
        latest_commit: u64,
    ) -> FrameOpportunity {
        let tick = FrameTick {
            now: HostTime(now),
            predicted_present: Some(HostTime(desired_present)),
            refresh_interval: Some(REFRESH_INTERVAL.ticks()),
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = PresentHints::predictive(HostTime(desired_present), HostTime(latest_commit));
        FrameOpportunity::new(tick, hints, DisplayTiming::fixed(REFRESH_INTERVAL))
    }

    fn predictive_opportunity_with_prev_actual(
        now: u64,
        frame_index: u64,
        desired_present: u64,
        latest_commit: u64,
        prev_actual_present: Option<HostTime>,
    ) -> FrameOpportunity {
        let mut opportunity =
            predictive_opportunity(now, frame_index, desired_present, latest_commit);
        opportunity.tick.prev_actual_present = prev_actual_present;
        opportunity
    }

    fn begin_at(driver: &mut FrameDriver, now: u64) -> FrameBeginResult {
        driver.begin_frame(opportunity(now, 0)).result
    }

    fn ready_at(driver: &mut FrameDriver, now: u64) -> ActiveFrame {
        let FrameBeginResult::Ready(frame) = begin_at(driver, now) else {
            panic!("expected ready frame");
        };
        frame
    }

    #[test]
    fn pacing_only_opportunity_fills_common_host_defaults() {
        let opportunity =
            FrameOpportunity::pacing_only(HostTime(12), REFRESH_INTERVAL, 42, OutputId(9));

        assert_eq!(opportunity.tick.now, HostTime(12));
        assert_eq!(opportunity.tick.predicted_present, None);
        assert_eq!(
            opportunity.tick.refresh_interval,
            Some(REFRESH_INTERVAL.ticks())
        );
        assert_eq!(opportunity.tick.frame_index, 42);
        assert_eq!(opportunity.tick.output, OutputId(9));
        assert_eq!(
            opportunity.hints.presentation_timing(),
            PresentationTiming::PacingOnly
        );
        assert_eq!(opportunity.hints.desired_present(), None);
        assert_eq!(
            opportunity.hints.latest_commit(),
            HostTime(12 + REFRESH_INTERVAL.ticks())
        );
        assert_eq!(
            opportunity.display_timing,
            DisplayTiming::fixed(REFRESH_INTERVAL)
        );
    }

    #[test]
    fn no_demand_does_not_plan_frame() {
        let mut driver = driver();

        assert!(matches!(begin_at(&mut driver, 0), FrameBeginResult::Idle));
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn demand_queues_future_frame_start_until_ready() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);

        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));
        assert!(matches!(
            begin_at(&mut driver, 89),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        let frame = ready_at(&mut driver, 90);
        assert_eq!(frame.tick().now, HostTime(0));
        assert_eq!(frame.build_start(), HostTime(90));
        assert_eq!(frame.plan().demand, FrameDemand::ANIMATION);
        assert_eq!(frame.plan().frame_start, HostTime(90));
        assert_eq!(frame.plan().sample_time, HostTime(100));
        assert_eq!(frame.plan().commit_deadline, HostTime(100));
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn begin_frame_returns_ready_frame() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        let frame = ready_at(&mut driver, 90);
        assert_eq!(frame.tick().now, HostTime(0));
        assert_eq!(frame.build_start(), HostTime(90));
        assert_eq!(frame.plan().sample_time, HostTime(100));
    }

    #[test]
    fn expired_queued_frame_returns_drop_summary() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));
        let state_before = driver.scheduler().state();

        let result = begin_at(&mut driver, 101);
        let FrameBeginResult::Expired(summary) = result else {
            panic!("expected expired queued frame");
        };

        assert_eq!(summary.frame_index, 0);
        assert_eq!(summary.output, OutputId(0));
        assert_eq!(summary.demand, FrameDemand::ANIMATION);
        assert_eq!(summary.frame_start, HostTime(90));
        assert_eq!(summary.commit_deadline, HostTime(100));
        assert_eq!(summary.submitted_at, None);
        assert_eq!(summary.drop_reason, Some(FrameDropReason::MissedDeadline));
        assert_eq!(summary.scheduler_state, Some(state_before));
        assert_eq!(driver.scheduler().state(), state_before);
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn begin_frame_does_not_return_expired_queued_frame_as_ready() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        assert!(matches!(
            begin_at(&mut driver, 101),
            FrameBeginResult::Expired(_)
        ));
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn stronger_demand_replans_with_combined_demand() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));

        driver.request(FrameDemand::INPUT);
        assert_eq!(driver.next_frame_start(), None);

        let frame = ready_at(&mut driver, 1);
        assert_eq!(frame.tick().now, HostTime(1));
        assert!(frame.plan().demand.contains(FrameDemand::INPUT));
        assert!(frame.plan().demand.contains(FrameDemand::ANIMATION));
        assert_eq!(frame.plan().frame_start, HostTime(1));
    }

    #[test]
    fn weaker_demand_waits_behind_queued_plan() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        driver.request(FrameDemand::BACKGROUND);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));
        assert!(driver.has_pending_demand());
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        let frame = ready_at(&mut driver, 90);
        assert_eq!(frame.plan().demand, FrameDemand::ANIMATION);
        assert!(driver.has_pending_demand());
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        assert!(matches!(
            begin_at(&mut driver, 90),
            FrameBeginResult::WaitUntil(HostTime(280))
        ));
        assert_eq!(driver.next_frame_start(), Some(HostTime(280)));
    }

    #[test]
    fn covered_demand_does_not_create_extra_pending_work() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        driver.request(FrameDemand::ANIMATION);

        assert_eq!(driver.pending_demand(), FrameDemand::NONE);
        let frame = ready_at(&mut driver, 90);
        assert_eq!(frame.plan().demand, FrameDemand::ANIMATION);
    }

    #[test]
    fn clear_pending_frame_drops_plan_without_clearing_demand() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));
        driver.request(FrameDemand::BACKGROUND);

        assert!(driver.clear_pending_frame());
        assert_eq!(driver.next_frame_start(), None);
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);
        assert!(!driver.clear_pending_frame());
    }

    #[test]
    fn weaker_retained_demand_survives_stronger_frame_submission() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        driver.request(FrameDemand::BACKGROUND);
        let frame = ready_at(&mut driver, 90);
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        let summary = driver
            .submit_frame(frame, FrameSubmission::new(HostTime(91), None))
            .summary
            .expect("unavailable present feedback should resolve immediately");

        assert_eq!(summary.demand, FrameDemand::ANIMATION);
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);
    }

    #[test]
    fn submit_frame_returns_summary_from_driver_lifecycle_facts() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = ready_at(&mut driver, 10);
        let submission = FrameSubmission::new(HostTime(20), None);
        let expected_feedback = PresentFeedback::new(
            &frame.plan(),
            frame.build_start(),
            submission.submitted_at,
            submission.actual_present(),
        );
        let tick_event = FrameTickEvent::from(&frame.tick());
        let plan = frame.plan();
        let plan_event = FramePlanEvent::new(&plan, frame.safety_margin_ticks());
        let submit_event = SubmitEvent {
            frame_index: plan.frame_index,
            submitted_at: submission.submitted_at,
            expected_present: expected_feedback.expected_present,
        };
        let feedback_event = PresentFeedbackEvent::new(plan.frame_index, &expected_feedback);

        let summary = driver
            .submit_frame(frame, submission)
            .summary
            .expect("unavailable present feedback should resolve immediately");

        let state_event = SchedulerStateEvent {
            state: driver.scheduler().state(),
        };
        let mut expected = FrameTimingSummaryBuilder::from_tick_and_plan(&tick_event, &plan_event);
        expected
            .record_submit(&submit_event)
            .record_present_feedback(&feedback_event)
            .record_scheduler_state(&state_event);

        assert_eq!(summary, expected.finish().expect("summary should finish"));
    }

    #[test]
    fn discard_frame_returns_drop_summary_without_observe() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = ready_at(&mut driver, 10);
        let plan = frame.plan();
        let state_before = driver.scheduler().state();

        let summary = driver.discard_frame_with_reason(frame, FrameDropReason::SurfaceUnavailable);

        assert_eq!(driver.scheduler().state(), state_before);
        assert_eq!(summary.frame_index, plan.frame_index);
        assert_eq!(summary.output, plan.output);
        assert_eq!(summary.submitted_at, None);
        assert_eq!(summary.actual_present, None);
        assert_eq!(
            summary.drop_reason,
            Some(FrameDropReason::SurfaceUnavailable)
        );
        assert_eq!(summary.scheduler_state, Some(state_before));
    }

    #[test]
    fn retained_demand_survives_discarded_active_frame() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            begin_at(&mut driver, 0),
            FrameBeginResult::WaitUntil(HostTime(90))
        ));

        driver.request(FrameDemand::BACKGROUND);
        let frame = ready_at(&mut driver, 90);
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        let summary = driver.discard_frame(frame);

        assert_eq!(summary.drop_reason, Some(FrameDropReason::Discarded));
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);
    }

    #[test]
    fn pacing_only_submission_summary_has_no_actual_present() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = ready_at(&mut driver, 10);

        let summary = driver
            .submit_frame(frame, FrameSubmission::new(HostTime(20), None))
            .summary
            .expect("unavailable present feedback should resolve immediately");

        assert_eq!(summary.timing_basis, FrameTimingBasis::PacingOnly);
        assert_eq!(summary.actual_present, None);
        assert_eq!(summary.expected_present, None);
        assert_eq!(summary.pacing_overrun, Some(false));
    }

    #[test]
    fn actual_present_submission_summary_records_deadline_facts() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let FrameBeginResult::Ready(frame) = driver
            .begin_frame(predictive_opportunity(10, 3, 100, 90))
            .result
        else {
            panic!("input should start immediately");
        };

        let summary = driver
            .submit_frame(
                frame,
                FrameSubmission::new(HostTime(20), Some(HostTime(101))),
            )
            .summary
            .expect("actual-present feedback should resolve immediately");

        assert_eq!(summary.timing_basis, FrameTimingBasis::ActualPresent);
        assert_eq!(summary.expected_present, Some(HostTime(100)));
        assert_eq!(summary.actual_present, Some(HostTime(101)));
        assert_eq!(summary.missed_deadline, Some(true));
        assert_eq!(summary.pacing_overrun, None);
    }

    #[test]
    fn deferred_submission_resolves_feedback_on_next_begin_frame() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let FrameBeginResult::Ready(frame) = driver
            .begin_frame(predictive_opportunity(10, 3, 100, 90))
            .result
        else {
            panic!("input should start immediately");
        };
        let state_before_submit = driver.scheduler().state();

        let submit = driver.submit_frame(frame, FrameSubmission::deferred(HostTime(20)));

        assert_eq!(submit.summary, None);
        assert!(submit.awaiting_actual_present);
        assert_eq!(driver.scheduler().state(), state_before_submit);

        let begin = driver.begin_frame(predictive_opportunity_with_prev_actual(
            110,
            4,
            200,
            190,
            Some(HostTime(99)),
        ));
        let summary = begin
            .resolved_feedback
            .expect("next tick should resolve deferred feedback");

        assert!(matches!(begin.result, FrameBeginResult::Idle));
        assert_eq!(summary.timing_basis, FrameTimingBasis::ActualPresent);
        assert_eq!(summary.submitted_at, Some(HostTime(20)));
        assert_eq!(summary.expected_present, Some(HostTime(100)));
        assert_eq!(summary.actual_present, Some(HostTime(99)));
        assert_eq!(summary.missed_deadline, Some(false));
        assert!(driver.scheduler().safety_margin_ticks() > 0);
    }

    #[test]
    fn submit_frame_judges_feedback_against_shifted_plan() {
        let mut config = SchedulerConfig::predictive();
        config.initial_depth = 2;
        config.minimum_frame_start_margin = Duration::ZERO;
        let mut driver = FrameDriver::new(config);

        driver.request(FrameDemand::ANIMATION);
        assert!(matches!(
            driver
                .begin_frame(predictive_opportunity(0, 4, 100, 90))
                .result,
            FrameBeginResult::WaitUntil(HostTime(190))
        ));

        let FrameBeginResult::Ready(frame) = driver
            .begin_frame(predictive_opportunity(190, 4, 290, 280))
            .result
        else {
            panic!("shifted frame should be ready at its planned start");
        };
        let plan = frame.plan();
        assert_eq!(plan.target_present, Some(HostTime(200)));
        assert_eq!(plan.commit_deadline, HostTime(190));

        let summary = driver
            .submit_frame(
                frame,
                FrameSubmission::new(HostTime(190), Some(HostTime(200))),
            )
            .summary
            .expect("actual-present feedback should resolve immediately");

        assert_eq!(summary.expected_present, Some(HostTime(200)));
        assert_eq!(summary.missed_deadline, Some(false));
        assert_eq!(summary.pacing_overrun, None);
    }

    #[test]
    fn observe_updates_underlying_scheduler() {
        let mut driver = driver();
        driver.observe(&PresentFeedback {
            submitted_at: HostTime(20),
            build_start: HostTime(10),
            expected_present: None,
            actual_present: None,
            missed_deadline: None,
            pacing_overrun: Some(false),
        });

        assert!(driver.scheduler().safety_margin_ticks() > 0);
    }
}
