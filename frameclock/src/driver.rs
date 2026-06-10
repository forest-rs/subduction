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
    FramePlanEvent, FrameTickEvent, FrameTimingSummary, FrameTimingSummaryBuilder,
    PresentFeedbackEvent, SchedulerStateEvent, SubmitEvent,
};
use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::time::HostTime;
use crate::timing::{
    DisplayTiming, FramePlan, FrameRequest, FrameTick, PresentFeedback, PresentHints,
};

/// A platform frame opportunity for retained driver lifecycle APIs.
///
/// Hosts construct this from the current display/frame callback. `FrameDriver`
/// combines it with pending demand to decide whether a frame should begin now
/// or whether a planned frame should remain queued until its frame-start time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameOpportunity {
    /// Platform frame opportunity.
    pub tick: FrameTick,
    /// Backend submission constraints for this opportunity.
    pub hints: PresentHints,
    /// Display timing constraints for the target output.
    pub display_timing: DisplayTiming,
}

impl FrameOpportunity {
    /// Creates a frame opportunity from platform timing facts.
    #[inline]
    #[must_use]
    pub const fn new(tick: FrameTick, hints: PresentHints, display_timing: DisplayTiming) -> Self {
        Self {
            tick,
            hints,
            display_timing,
        }
    }
}

/// A scheduler plan paired with the platform facts used to make it.
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

/// A planned frame that has been handed to the host for preparation.
///
/// `ActiveFrame` is the normal host handle for frame work. It carries the
/// scheduler plan plus the original planning facts needed to resolve feedback
/// and build a [`FrameTimingSummary`] when the host submits the frame.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// Host submission facts for an [`ActiveFrame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameSubmission {
    /// Host time when the frame was submitted/committed.
    pub submitted_at: HostTime,
    /// Actual presentation time, if the platform reported it.
    pub actual_present: Option<HostTime>,
}

impl FrameSubmission {
    /// Creates frame submission facts.
    #[inline]
    #[must_use]
    pub const fn new(submitted_at: HostTime, actual_present: Option<HostTime>) -> Self {
        Self {
            submitted_at,
            actual_present,
        }
    }
}

/// Driver for pending demand, queued frame plans, and frame feedback.
///
/// `FrameDriver` is a small retained layer above [`Scheduler`]. Hosts call
/// [`request`](Self::request) when app-visible frame demand arrives, then call
/// [`begin_frame`](Self::begin_frame) from a platform redraw/tick opportunity. If
/// the scheduler chooses a future [`FramePlan::frame_start`], the driver keeps
/// that plan queued and exposes the wake time through
/// [`next_frame_start`](Self::next_frame_start).
///
/// Submit completed frames with [`submit_frame`](Self::submit_frame) so
/// `frameclock` can observe feedback and produce the frame timing summary.
/// Discard frames that will not be submitted with
/// [`discard_frame`](Self::discard_frame).
///
/// This keeps demand preemption, feedback observation, and summary construction
/// inside `frameclock` while leaving timer queues, event-loop wake mechanics,
/// renderer submission, and native surface lifecycle in the host.
#[derive(Debug)]
pub struct FrameDriver {
    scheduler: Scheduler,
    pending_demand: FrameDemand,
    pending_frame: Option<PlannedFrame>,
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

    /// Adds app-visible frame demand.
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

    /// Begins a frame when pending demand is ready to prepare.
    ///
    /// This is the normal host entry point for converting a platform frame
    /// opportunity into frame work. It is the only public method that consumes
    /// pending demand into a ready frame. If the selected frame start is still
    /// in the future, this stores the planned frame internally and returns
    /// `None`; hosts should arm their own timer for
    /// [`next_frame_start`](Self::next_frame_start).
    #[must_use]
    pub fn begin_frame(&mut self, opportunity: FrameOpportunity) -> Option<ActiveFrame> {
        self.take_ready(opportunity)
            .map(|frame| ActiveFrame::new(frame, opportunity.tick.now))
    }

    /// Reports that an active frame was submitted and returns its timing
    /// summary.
    ///
    /// This constructs presentation feedback, feeds it to the scheduler, and
    /// returns the corresponding [`FrameTimingSummary`]. Hosts using this API
    /// do not need to manually assemble frameclock diagnostics events or a
    /// [`FrameTimingSummaryBuilder`].
    #[must_use]
    pub fn submit_frame(
        &mut self,
        frame: ActiveFrame,
        submission: FrameSubmission,
    ) -> FrameTimingSummary {
        let feedback = PresentFeedback::new(
            &frame.hints(),
            frame.build_start(),
            submission.submitted_at,
            submission.actual_present,
        );
        let tick_event = FrameTickEvent::from(&frame.tick());
        let plan = frame.plan();
        let plan_event = FramePlanEvent::new(&plan, frame.safety_margin_ticks());
        let submit_event = SubmitEvent {
            frame_index: plan.frame_index,
            submitted_at: submission.submitted_at,
            expected_present: feedback.expected_present,
        };
        let feedback_event = PresentFeedbackEvent::new(plan.frame_index, &feedback);

        self.scheduler.observe(&feedback);

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
    pub fn discard_frame(&mut self, frame: ActiveFrame) {
        _ = frame;
    }

    /// Returns a queued frame when it is ready, or plans pending demand.
    ///
    /// `opportunity` describes the current platform
    /// redraw/tick opportunity. If demand exists but the scheduler-selected
    /// frame start is still in the future, this stores the plan internally and
    /// returns `None`; hosts should then arm their own timer for
    /// [`next_frame_start`](Self::next_frame_start).
    fn take_ready(&mut self, opportunity: FrameOpportunity) -> Option<PlannedFrame> {
        let tick = opportunity.tick;
        if let Some(frame) = self.pending_frame {
            if tick.now >= frame.plan.frame_start {
                self.pending_frame = None;
                return Some(frame);
            }
            return None;
        }

        if self.pending_demand.is_empty() {
            return None;
        }

        let demand = self.pending_demand;
        self.pending_demand = FrameDemand::NONE;
        let plan = self.scheduler.plan(FrameRequest::new(
            tick,
            opportunity.hints,
            demand,
            opportunity.display_timing,
        ));
        let frame = PlannedFrame::new(
            tick,
            plan,
            opportunity.hints,
            self.scheduler.safety_margin_ticks(),
        );
        if tick.now >= frame.plan.frame_start {
            Some(frame)
        } else {
            self.pending_frame = Some(frame);
            None
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
    use crate::timing::{PresentFeedback, TimingConfidence};

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
            confidence: TimingConfidence::PacingOnly,
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    fn hints(deadline: u64) -> PresentHints {
        PresentHints {
            desired_present: None,
            latest_commit: HostTime(deadline),
        }
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
            confidence: TimingConfidence::Predictive,
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let hints = PresentHints {
            desired_present: Some(HostTime(desired_present)),
            latest_commit: HostTime(latest_commit),
        };
        FrameOpportunity::new(tick, hints, DisplayTiming::fixed(REFRESH_INTERVAL))
    }

    fn begin_at(driver: &mut FrameDriver, now: u64) -> Option<ActiveFrame> {
        driver.begin_frame(opportunity(now, 0))
    }

    #[test]
    fn no_demand_does_not_plan_frame() {
        let mut driver = driver();

        assert_eq!(begin_at(&mut driver, 0), None);
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn demand_queues_future_frame_start_until_ready() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);

        assert_eq!(begin_at(&mut driver, 0), None);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));
        assert_eq!(begin_at(&mut driver, 89), None);

        let frame = begin_at(&mut driver, 90).expect("queued frame should be ready");
        assert_eq!(frame.tick().now, HostTime(0));
        assert_eq!(frame.build_start(), HostTime(90));
        assert_eq!(frame.plan().demand, FrameDemand::ANIMATION);
        assert_eq!(frame.plan().frame_start, HostTime(90));
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn stronger_demand_replans_with_combined_demand() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(begin_at(&mut driver, 0), None);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));

        driver.request(FrameDemand::INPUT);
        assert_eq!(driver.next_frame_start(), None);

        let frame = begin_at(&mut driver, 1).expect("input should be ready immediately");
        assert_eq!(frame.tick().now, HostTime(1));
        assert!(frame.plan().demand.contains(FrameDemand::INPUT));
        assert!(frame.plan().demand.contains(FrameDemand::ANIMATION));
        assert_eq!(frame.plan().frame_start, HostTime(1));
    }

    #[test]
    fn weaker_demand_waits_behind_queued_plan() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(begin_at(&mut driver, 0), None);

        driver.request(FrameDemand::BACKGROUND);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));
        assert!(driver.has_pending_demand());
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        let frame = begin_at(&mut driver, 90).expect("queued frame should be ready");
        assert_eq!(frame.plan().demand, FrameDemand::ANIMATION);
        assert!(driver.has_pending_demand());
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        assert_eq!(begin_at(&mut driver, 90), None);
        assert_eq!(driver.next_frame_start(), Some(HostTime(280)));
    }

    #[test]
    fn covered_demand_does_not_create_extra_pending_work() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(begin_at(&mut driver, 0), None);

        driver.request(FrameDemand::ANIMATION);

        assert_eq!(driver.pending_demand(), FrameDemand::NONE);
        let frame = begin_at(&mut driver, 90).expect("queued frame should be ready");
        assert_eq!(frame.plan().demand, FrameDemand::ANIMATION);
    }

    #[test]
    fn clear_pending_frame_drops_plan_without_clearing_demand() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(begin_at(&mut driver, 0), None);
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
        assert_eq!(begin_at(&mut driver, 0), None);

        driver.request(FrameDemand::BACKGROUND);
        let frame = begin_at(&mut driver, 90).expect("animation frame should be ready");
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        let summary = driver.submit_frame(
            frame,
            FrameSubmission {
                submitted_at: HostTime(91),
                actual_present: None,
            },
        );

        assert_eq!(summary.demand, FrameDemand::ANIMATION);
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);
    }

    #[test]
    fn submit_frame_returns_summary_from_driver_lifecycle_facts() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = begin_at(&mut driver, 10).expect("input should start immediately");
        let submission = FrameSubmission {
            submitted_at: HostTime(20),
            actual_present: None,
        };
        let expected_feedback = PresentFeedback::new(
            &frame.hints(),
            frame.build_start(),
            submission.submitted_at,
            submission.actual_present,
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

        let summary = driver.submit_frame(frame, submission);

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
    fn discard_frame_does_not_observe_feedback() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = begin_at(&mut driver, 10).expect("input should start immediately");
        let state_before = driver.scheduler().state();

        driver.discard_frame(frame);

        assert_eq!(driver.scheduler().state(), state_before);
    }

    #[test]
    fn pacing_only_submission_summary_has_no_actual_present() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = begin_at(&mut driver, 10).expect("input should start immediately");

        let summary = driver.submit_frame(
            frame,
            FrameSubmission {
                submitted_at: HostTime(20),
                actual_present: None,
            },
        );

        assert_eq!(summary.timing_basis, FrameTimingBasis::PacingOnly);
        assert_eq!(summary.actual_present, None);
        assert_eq!(summary.expected_present, None);
        assert_eq!(summary.pacing_overrun, Some(false));
    }

    #[test]
    fn actual_present_submission_summary_records_deadline_facts() {
        let mut driver = driver();
        driver.request(FrameDemand::INPUT);
        let frame = driver
            .begin_frame(predictive_opportunity(10, 3, 100, 90))
            .expect("input should start immediately");

        let summary = driver.submit_frame(
            frame,
            FrameSubmission {
                submitted_at: HostTime(20),
                actual_present: Some(HostTime(101)),
            },
        );

        assert_eq!(summary.timing_basis, FrameTimingBasis::ActualPresent);
        assert_eq!(summary.expected_present, Some(HostTime(100)));
        assert_eq!(summary.actual_present, Some(HostTime(101)));
        assert_eq!(summary.missed_deadline, Some(true));
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
