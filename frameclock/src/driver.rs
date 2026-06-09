// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained frame planning state.
//!
//! This module owns pending frame demand and queued frame plans. It explicitly
//! does not own platform event loops, app timers, redraw requests, renderer
//! submission, or surface lifecycle.

use crate::demand::FrameDemand;
use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::time::HostTime;
use crate::timing::{
    DisplayTiming, FramePlan, FrameRequest, FrameTick, PresentFeedback, PresentHints,
};

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
}

impl PlannedFrame {
    /// Creates a planned frame from a [`FrameTick`], [`FramePlan`], and
    /// matching [`PresentHints`].
    #[inline]
    #[must_use]
    pub const fn new(tick: FrameTick, plan: FramePlan, hints: PresentHints) -> Self {
        Self { tick, plan, hints }
    }
}

/// Driver for pending demand and queued frame plans.
///
/// `FrameDriver` is a small retained layer above [`Scheduler`]. Hosts call
/// [`request`](Self::request) when app-visible frame demand arrives, then call
/// [`take_ready`](Self::take_ready) from a platform redraw/tick opportunity. If
/// the scheduler chooses a future [`FramePlan::frame_start`], the driver keeps
/// that plan queued and exposes the wake time through
/// [`next_frame_start`](Self::next_frame_start).
///
/// This keeps the demand preemption policy inside `frameclock` while leaving
/// timer queues and event-loop wake mechanics in the host.
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
    /// plan so the next [`take_ready`](Self::take_ready) call replans with the
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

    /// Returns a queued frame when it is ready, or plans pending demand.
    ///
    /// `tick`, `hints`, and `display_timing` describe the current platform
    /// redraw/tick opportunity. If demand exists but the scheduler-selected
    /// frame start is still in the future, this stores the plan internally and
    /// returns `None`; hosts should then arm their own timer for
    /// [`next_frame_start`](Self::next_frame_start).
    #[must_use]
    pub fn take_ready(
        &mut self,
        tick: FrameTick,
        hints: PresentHints,
        display_timing: DisplayTiming,
    ) -> Option<PlannedFrame> {
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
        let plan = self
            .scheduler
            .plan(FrameRequest::new(tick, hints, demand, display_timing));
        let frame = PlannedFrame::new(tick, plan, hints);
        if tick.now >= frame.plan.frame_start {
            Some(frame)
        } else {
            self.pending_frame = Some(frame);
            None
        }
    }

    /// Drops the queued frame and returns it.
    ///
    /// This does not clear [`pending_demand`](Self::pending_demand). Demand
    /// retained behind the queued frame remains available for the next
    /// planning turn.
    pub fn clear_pending_frame(&mut self) -> Option<PlannedFrame> {
        self.pending_frame.take()
    }

    /// Feeds presentation feedback to the underlying [`Scheduler`].
    pub fn observe(&mut self, feedback: &PresentFeedback) {
        self.scheduler.observe(feedback);
    }
}

#[cfg(test)]
mod tests {
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

    fn take_ready_at(driver: &mut FrameDriver, now: u64) -> Option<PlannedFrame> {
        driver.take_ready(
            tick(now, 0),
            hints(now + REFRESH_INTERVAL.ticks()),
            DisplayTiming::fixed(REFRESH_INTERVAL),
        )
    }

    #[test]
    fn no_demand_does_not_plan_frame() {
        let mut driver = driver();

        assert_eq!(take_ready_at(&mut driver, 0), None);
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn demand_queues_future_frame_start_until_ready() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);

        assert_eq!(take_ready_at(&mut driver, 0), None);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));
        assert_eq!(take_ready_at(&mut driver, 89), None);

        let frame = take_ready_at(&mut driver, 90).expect("queued frame should be ready");
        assert_eq!(frame.tick.now, HostTime(0));
        assert_eq!(frame.plan.demand, FrameDemand::ANIMATION);
        assert_eq!(frame.plan.frame_start, HostTime(90));
        assert_eq!(driver.next_frame_start(), None);
    }

    #[test]
    fn stronger_demand_replans_with_combined_demand() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(take_ready_at(&mut driver, 0), None);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));

        driver.request(FrameDemand::INPUT);
        assert_eq!(driver.next_frame_start(), None);

        let frame = take_ready_at(&mut driver, 1).expect("input should be ready immediately");
        assert_eq!(frame.tick.now, HostTime(1));
        assert!(frame.plan.demand.contains(FrameDemand::INPUT));
        assert!(frame.plan.demand.contains(FrameDemand::ANIMATION));
        assert_eq!(frame.plan.frame_start, HostTime(1));
    }

    #[test]
    fn weaker_demand_waits_behind_queued_plan() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(take_ready_at(&mut driver, 0), None);

        driver.request(FrameDemand::BACKGROUND);
        assert_eq!(driver.next_frame_start(), Some(HostTime(90)));
        assert!(driver.has_pending_demand());
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        let frame = take_ready_at(&mut driver, 90).expect("queued frame should be ready");
        assert_eq!(frame.plan.demand, FrameDemand::ANIMATION);
        assert!(driver.has_pending_demand());
        assert_eq!(driver.pending_demand(), FrameDemand::BACKGROUND);

        assert_eq!(take_ready_at(&mut driver, 90), None);
        assert_eq!(driver.next_frame_start(), Some(HostTime(280)));
    }

    #[test]
    fn covered_demand_does_not_create_extra_pending_work() {
        let mut driver = driver();
        driver.request(FrameDemand::ANIMATION);
        assert_eq!(take_ready_at(&mut driver, 0), None);

        driver.request(FrameDemand::ANIMATION);

        assert_eq!(driver.pending_demand(), FrameDemand::NONE);
        let frame = take_ready_at(&mut driver, 90).expect("queued frame should be ready");
        assert_eq!(frame.plan.demand, FrameDemand::ANIMATION);
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
