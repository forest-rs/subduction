// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Matching `wp_presentation` feedback to the frames it reports on.

use std::collections::VecDeque;

use frameclock::HostTime;
use frameclock::timing::{FramePlan, PresentFeedback};
use frameclock_wayland::{PresentEvent, SubmissionId};

/// How many committed frames wait for their feedback before the oldest is
/// judged without it.
///
/// This matches how many feedback requests the backend keeps pending: past
/// that, a commit gets none, so its frame would otherwise wait forever.
pub const MAX_WAITING: usize = 64;

/// Frames committed to the compositor, waiting for their own
/// `wp_presentation` feedback.
///
/// [`FrameTick::prev_actual_present`](frameclock::FrameTick::prev_actual_present)
/// is when the *previous* commit was shown, so judging the frame just
/// committed by it attributes every frame's presentation to the frame after
/// it, and a frame shown late is almost never seen as late. Feedback carries
/// the [`SubmissionId`] of the commit it is for, so each frame is judged by
/// its own presentation instead.
///
/// Plan each frame with [`next_frame_index`](Self::next_frame_index), record
/// it with [`committed`](Self::committed) once it is committed, and pass
/// every [`PresentEvent`] to [`resolve`](Self::resolve). Feed what they
/// return to the scheduler.
#[derive(Debug, Default)]
pub struct FrameFeedback {
    next_frame_index: u64,
    waiting: VecDeque<(SubmissionId, Committed)>,
}

/// What a committed frame's feedback is built from.
#[derive(Clone, Copy, Debug)]
struct Committed {
    plan: FramePlan,
    build_start: HostTime,
    submitted_at: HostTime,
}

impl Committed {
    fn feedback(&self, actual_present: Option<HostTime>) -> PresentFeedback {
        PresentFeedback::new(
            &self.plan,
            self.build_start,
            self.submitted_at,
            actual_present,
        )
    }
}

impl FrameFeedback {
    /// Creates an empty tracker whose first frame index is `0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the index to plan the next frame with, counting up from `0`,
    /// so each frame's feedback and diagnostics name that frame.
    pub fn next_frame_index(&mut self) -> u64 {
        let index = self.next_frame_index;
        self.next_frame_index += 1;
        index
    }

    /// Records that the frame planned as `plan`, built from `build_start`,
    /// was committed as `id` at `submitted_at`.
    ///
    /// Returns commit-only feedback for the oldest waiting frame when more
    /// than [`MAX_WAITING`] frames wait: its feedback was never requested or
    /// never came.
    pub fn committed(
        &mut self,
        id: SubmissionId,
        plan: &FramePlan,
        build_start: HostTime,
        submitted_at: HostTime,
    ) -> Option<PresentFeedback> {
        self.waiting.push_back((
            id,
            Committed {
                plan: *plan,
                build_start,
                submitted_at,
            },
        ));
        if self.waiting.len() > MAX_WAITING {
            let (_, oldest) = self.waiting.pop_front()?;
            return Some(oldest.feedback(None));
        }
        None
    }

    /// Returns the feedback for the frame `event` reports on: judged by its
    /// own presentation time, or by its commit alone if the compositor
    /// discarded it.
    ///
    /// Returns `None` when no waiting frame was committed as the event's
    /// submission.
    pub fn resolve(&mut self, event: &PresentEvent) -> Option<PresentFeedback> {
        let (id, actual_present) = match *event {
            PresentEvent::Presented {
                id, actual_present, ..
            } => (id, Some(actual_present)),
            PresentEvent::Discarded { id } => (id, None),
        };
        let position = self
            .waiting
            .iter()
            .position(|(waiting, _)| *waiting == id)?;
        let (_, committed) = self.waiting.remove(position)?;
        Some(committed.feedback(actual_present))
    }

    /// Returns how many committed frames are waiting for feedback.
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameFeedback, MAX_WAITING};
    use frameclock::scheduler::Scheduler;
    use frameclock::timing::FramePlan;
    use frameclock::{Duration, FrameDemand, FrameTick, HostTime, OutputId, SchedulerConfig};
    use frameclock_wayland::{PresentEvent, SubmissionId};

    const REFRESH: u64 = 16_666_667;

    /// Plans a frame at a tick at `now`, after a refresh shown a millisecond
    /// earlier, so the plan has a target to be judged against.
    fn plan(scheduler: &mut Scheduler, frames: &mut FrameFeedback, now: u64) -> FramePlan {
        let tick = FrameTick {
            now: HostTime(now),
            predicted_present: None,
            refresh_interval: Some(REFRESH),
            output: OutputId(0),
            prev_actual_present: Some(HostTime(now - 1_000_000)),
        };
        let opportunity = frameclock_wayland::frame_opportunity(tick, Duration(REFRESH));
        scheduler.plan(
            opportunity,
            FrameDemand::ANIMATION,
            frames.next_frame_index(),
        )
    }

    fn presented(id: u64, at: HostTime) -> PresentEvent {
        PresentEvent::Presented {
            id: SubmissionId(id),
            actual_present: at,
            refresh_interval: Some(REFRESH),
            output: None,
            flags: 0,
        }
    }

    #[test]
    fn each_frame_is_judged_by_its_own_presentation() {
        let mut scheduler = Scheduler::new(SchedulerConfig::estimated());
        let mut frames = FrameFeedback::new();
        let first = plan(&mut scheduler, &mut frames, 100_000_000);
        let second = plan(&mut scheduler, &mut frames, 100_000_000 + REFRESH);
        assert_eq!((first.frame_index, second.frame_index), (0, 1));
        let first_target = first.target_present.expect("the plan has a target");
        let second_target = second.target_present.expect("the plan has a target");

        let start = HostTime(99_500_000);
        assert!(
            frames
                .committed(SubmissionId(7), &first, start, HostTime(101_000_000))
                .is_none()
        );
        assert!(
            frames
                .committed(SubmissionId(8), &second, start, HostTime(118_000_000))
                .is_none()
        );

        // The second frame's feedback arrives first, two refreshes late.
        let late = HostTime(second_target.ticks() + 2 * REFRESH);
        let second_feedback = frames
            .resolve(&presented(8, late))
            .expect("frame 1 was waiting");
        assert_eq!(second_feedback.actual_present, Some(late));
        assert_eq!(second_feedback.missed_deadline, Some(true));

        let first_feedback = frames
            .resolve(&presented(7, first_target))
            .expect("frame 0 was waiting");
        assert_eq!(first_feedback.actual_present, Some(first_target));
        assert_eq!(first_feedback.missed_deadline, Some(false));
        assert_eq!(frames.waiting(), 0);
    }

    #[test]
    fn a_discarded_frame_is_judged_by_its_commit_alone() {
        let mut scheduler = Scheduler::new(SchedulerConfig::estimated());
        let mut frames = FrameFeedback::new();
        let frame = plan(&mut scheduler, &mut frames, 100_000_000);
        let _ = frames.committed(
            SubmissionId(3),
            &frame,
            HostTime(99_000_000),
            HostTime(101_000_000),
        );
        let feedback = frames
            .resolve(&PresentEvent::Discarded {
                id: SubmissionId(3),
            })
            .expect("the frame was waiting");
        assert_eq!(feedback.actual_present, None);
        assert_eq!(feedback.missed_deadline, None);
    }

    #[test]
    fn feedback_for_no_waiting_frame_is_ignored() {
        let mut frames = FrameFeedback::new();
        assert!(frames.resolve(&presented(1, HostTime(5))).is_none());
    }

    #[test]
    fn a_frame_whose_feedback_never_comes_is_judged_by_its_commit() {
        let mut scheduler = Scheduler::new(SchedulerConfig::estimated());
        let mut frames = FrameFeedback::new();
        let first = plan(&mut scheduler, &mut frames, 100_000_000);
        let first_submit = HostTime(101_000_000);
        let _ = frames.committed(SubmissionId(0), &first, HostTime(99_000_000), first_submit);
        for id in 1..MAX_WAITING as u64 {
            assert!(
                frames
                    .committed(SubmissionId(id), &first, HostTime(99_000_000), first_submit)
                    .is_none()
            );
        }
        let oldest = frames
            .committed(
                SubmissionId(MAX_WAITING as u64),
                &first,
                HostTime(99_000_000),
                first_submit,
            )
            .expect("the oldest frame is resolved to make room");
        assert_eq!(oldest.actual_present, None);
        assert_eq!(oldest.submitted_at, first_submit);
        assert_eq!(frames.waiting(), MAX_WAITING);
        assert!(frames.resolve(&presented(0, HostTime(5))).is_none());
    }
}
