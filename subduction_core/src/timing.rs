// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capability-graded timing model.
//!
//! This module defines the types that flow between backends and the scheduler:
//!
//! - [`TimingConfidence`] — how much the platform can tell us about presentation
//! - [`FrameTick`] — a frame opportunity delivered by the backend
//! - [`FramePlan`] — what the engine uses to evaluate the scene for a frame
//! - [`PresentHints`] — submission constraints from the backend
//! - [`PresentFeedback`] — post-submit observations fed back to the scheduler
//!
//! # Data flow
//!
//! Each frame follows a pipeline through these types:
//!
//! 1. The backend produces a [`FrameTick`] from a platform callback (e.g.
//!    `CADisplayLink`, `requestAnimationFrame`).
//! 2. The backend computes [`PresentHints`] from the tick and platform
//!    knowledge (deadlines, desired present time).
//! 3. [`Scheduler::plan()`](crate::scheduler::Scheduler::plan) consumes the
//!    tick and hints to produce a [`FramePlan`] with the semantic time,
//!    present time, and commit deadline.
//! 4. The application uses the plan to evaluate the scene and build/submit
//!    the frame.
//! 5. After submission, the backend constructs [`PresentFeedback`] from
//!    timing observations and feeds it back to
//!    [`Scheduler::observe()`](crate::scheduler::Scheduler::observe) to
//!    adapt pipeline depth and safety margins.

use crate::output::OutputId;
use crate::time::HostTime;

/// How reliable the predicted present time is.
///
/// Platforms differ in how well they can predict when pixels will appear on
/// screen. This enum captures that spectrum so the scheduler can adapt its
/// strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimingConfidence {
    /// Strong predicted present time available (e.g. macOS `CVDisplayLink`).
    Predictive,
    /// Vsync-ish timing but less strict (e.g. Android Choreographer).
    Estimated,
    /// No reliable present time; frame pacing only (e.g. Web `rAF`, X11 fallback).
    PacingOnly,
}

/// A frame opportunity delivered by the backend.
///
/// Backends produce a `FrameTick` each time a new frame can be submitted. Not
/// all fields are populated on every platform — [`Option`] fields reflect the
/// capability-graded timing model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameTick {
    /// Current host time when the tick was generated.
    pub now: HostTime,
    /// Predicted time when pixels will be presented, if known.
    pub predicted_present: Option<HostTime>,
    /// Display refresh interval in host-time ticks, if known.
    pub refresh_interval: Option<u64>,
    /// Confidence level for timing information in this tick.
    pub confidence: TimingConfidence,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
    /// Which output this tick is for.
    pub output: OutputId,
    /// Actual present time of the *previous* frame, if the backend can report
    /// it (e.g. from `CADisplayLink.timestamp`).
    pub prev_actual_present: Option<HostTime>,
}

/// The plan for evaluating a single frame.
///
/// Produced by the [`Scheduler`](crate::scheduler::Scheduler) from a
/// [`FrameTick`] and [`PresentHints`]. All engine evaluation and content
/// selection should be driven by the times in this plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FramePlan {
    /// The time the scene represents (animations, simulation, overlays).
    pub semantic_time: HostTime,
    /// Intended display time, if known.
    pub present_time: Option<HostTime>,
    /// Latest time by which the frame must be committed/submitted.
    pub commit_deadline: HostTime,
    /// Current pipeline depth (1–3).
    pub pipeline_depth: u8,
    /// Which output this frame targets.
    pub output: OutputId,
    /// Frame counter, carried from the originating [`FrameTick`].
    pub frame_index: u64,
}

/// Submission constraints provided by the backend.
///
/// Backends compute these from the current [`FrameTick`] and their own
/// knowledge of the presentation pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentHints {
    /// Target present time, if known.
    pub desired_present: Option<HostTime>,
    /// Latest time by which a commit must occur to hit the desired present.
    pub latest_commit: HostTime,
}

/// Timing feedback constructed by the caller at the end of each tick handler.
///
/// Fed back to the [`Scheduler`](crate::scheduler::Scheduler) so it can adapt
/// pipeline depth and safety margins.
///
/// This type intentionally separates two different claims:
///
/// - [`PresentFeedback::missed_deadline`] answers "do we know this frame
///   missed a real presentation deadline?"
/// - [`PresentFeedback::pacing_overrun`] answers "did frame building run past
///   the pacing boundary exposed by this backend?"
///
/// Pacing-only backends often know the second thing but not the first. Keeping
/// those signals separate avoids laundering weak timing evidence into false
/// deadline truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentFeedback {
    /// Host time when the frame was submitted/committed.
    pub submitted_at: HostTime,
    /// When the frame began building (for build cost estimation).
    pub build_start: HostTime,
    /// Expected present time at submission, if known.
    pub expected_present: Option<HostTime>,
    /// Actual present time, if the platform reports it.
    pub actual_present: Option<HostTime>,
    /// Whether the commit deadline was missed, if determinable.
    pub missed_deadline: Option<bool>,
    /// Whether frame building overran the pacing tick budget.
    ///
    /// This is a weaker signal than [`missed_deadline`]: it says only that the
    /// frame was submitted after the backend's pacing boundary, not that the
    /// compositor actually presented it late.
    ///
    /// [`missed_deadline`]: Self::missed_deadline
    pub pacing_overrun: Option<bool>,
}

impl PresentFeedback {
    /// Constructs feedback from timing observations and [`PresentHints`].
    ///
    /// This derives both the strict deadline signal and the weaker pacing
    /// signal.
    ///
    /// `missed_deadline` should only answer "the frame was late" when the
    /// backend has enough information to support that claim.
    /// `pacing_overrun` answers the weaker question "we ran long relative to
    /// the pacing tick budget" for backends that only expose pacing.
    ///
    /// The derivation rules are:
    ///
    /// - If both `actual_present` and `hints.desired_present` are known, a
    ///   frame is missed when `actual_present > desired_present`.
    /// - Otherwise, if `hints.desired_present` is known, fall back to commit
    ///   timing: missed when `submitted_at > hints.latest_commit`.
    /// - Otherwise, the result is `None`: pacing-only backends without real
    ///   presentation feedback do not have enough information to classify the
    ///   frame as hit or miss honestly.
    ///
    /// When `hints.desired_present` is `None`, [`PresentFeedback::pacing_overrun`]
    /// is populated from commit timing as the weaker pacing-only signal.
    #[must_use]
    pub fn new(
        hints: &PresentHints,
        build_start: HostTime,
        submitted_at: HostTime,
        actual_present: Option<HostTime>,
    ) -> Self {
        let expected_present = hints.desired_present;
        // Pacing-only backends still expose a useful "we ran long relative to
        // the pacing boundary" signal. Keep that distinct from real deadline
        // truth.
        let pacing_overrun = if expected_present.is_none() {
            Some(submitted_at > hints.latest_commit)
        } else {
            None
        };
        // Only report deadline truth when the backend can honestly support it.
        let missed_deadline = match (actual_present, expected_present) {
            (Some(actual), Some(expected)) => Some(actual > expected),
            (None, Some(_expected)) => Some(submitted_at > hints.latest_commit),
            (_, None) => None,
        };

        Self {
            submitted_at,
            build_start,
            expected_present,
            actual_present,
            missed_deadline,
            pacing_overrun,
        }
    }
}

/// Holds the data needed to construct [`PresentFeedback`] once the actual
/// present time becomes available on the *next* frame.
///
/// Actual present time for frame N is typically only known at frame N+1
/// (e.g. from `CADisplayLink.timestamp`). This type captures what we know at
/// submission time so the feedback can be resolved one frame later.
///
/// # Usage
///
/// ```text
/// // Frame N: submit, then store pending.
/// let pending = PendingFeedback { hints, build_start, submitted_at };
///
/// // Frame N+1: resolve with actual_present from the new tick.
/// let feedback = pending.resolve(tick.prev_actual_present);
/// scheduler.observe(&feedback);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PendingFeedback {
    /// The [`PresentHints`] that were active when the frame was planned.
    pub hints: PresentHints,
    /// Host time when frame building began.
    pub build_start: HostTime,
    /// Host time when the frame was submitted/committed.
    pub submitted_at: HostTime,
}

impl PendingFeedback {
    /// Resolves this pending feedback into a [`PresentFeedback`], using the
    /// actual present time reported by the backend (if available).
    #[must_use]
    pub fn resolve(self, actual_present: Option<HostTime>) -> PresentFeedback {
        PresentFeedback::new(
            &self.hints,
            self.build_start,
            self.submitted_at,
            actual_present,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_with_actual_present_compares_to_expected() {
        let hints = PresentHints {
            desired_present: Some(HostTime(2_000_000)),
            latest_commit: HostTime(1_800_000),
        };
        let fb = PresentFeedback::new(
            &hints,
            HostTime(1_700_000),
            HostTime(1_750_000),
            Some(HostTime(2_100_000)),
        );
        assert_eq!(fb.missed_deadline, Some(true));
        assert_eq!(fb.expected_present, Some(HostTime(2_000_000)));
        assert_eq!(fb.actual_present, Some(HostTime(2_100_000)));
        assert_eq!(fb.pacing_overrun, None);

        // On time.
        let fb = PresentFeedback::new(
            &hints,
            HostTime(1_700_000),
            HostTime(1_750_000),
            Some(HostTime(1_999_000)),
        );
        assert_eq!(fb.missed_deadline, Some(false));
        assert_eq!(fb.pacing_overrun, None);
    }

    #[test]
    fn new_without_actual_present_uses_commit_deadline() {
        let hints = PresentHints {
            desired_present: Some(HostTime(2_000_000)),
            latest_commit: HostTime(1_800_000),
        };
        // submitted_at > latest_commit → missed
        let fb = PresentFeedback::new(&hints, HostTime(1_700_000), HostTime(1_900_000), None);
        assert_eq!(fb.missed_deadline, Some(true));
        assert_eq!(fb.pacing_overrun, None);

        // submitted_at <= latest_commit → not missed
        let fb = PresentFeedback::new(&hints, HostTime(1_700_000), HostTime(1_750_000), None);
        assert_eq!(fb.missed_deadline, Some(false));
        assert_eq!(fb.pacing_overrun, None);
    }

    #[test]
    fn new_without_desired_present_is_unknown() {
        let hints = PresentHints {
            desired_present: None,
            latest_commit: HostTime(1_000_000),
        };
        let fb = PresentFeedback::new(&hints, HostTime(900_000), HostTime(1_100_000), None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.expected_present, None);
        assert_eq!(fb.pacing_overrun, Some(true));
    }

    #[test]
    fn new_with_actual_present_but_no_expected_present_is_unknown() {
        let hints = PresentHints {
            desired_present: None,
            latest_commit: HostTime(1_000_000),
        };
        let fb = PresentFeedback::new(
            &hints,
            HostTime(900_000),
            HostTime(1_100_000),
            Some(HostTime(1_200_000)),
        );
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.actual_present, Some(HostTime(1_200_000)));
        assert_eq!(fb.pacing_overrun, Some(true));
    }

    #[test]
    fn pending_feedback_resolve_with_actual_present() {
        let pending = PendingFeedback {
            hints: PresentHints {
                desired_present: Some(HostTime(2_000_000)),
                latest_commit: HostTime(1_800_000),
            },
            build_start: HostTime(1_700_000),
            submitted_at: HostTime(1_750_000),
        };

        // Actual present arrived late → missed.
        let fb = pending.resolve(Some(HostTime(2_100_000)));
        assert_eq!(fb.missed_deadline, Some(true));
        assert_eq!(fb.actual_present, Some(HostTime(2_100_000)));
        assert_eq!(fb.pacing_overrun, None);

        // Actual present on time → not missed.
        let fb = pending.resolve(Some(HostTime(1_999_000)));
        assert_eq!(fb.missed_deadline, Some(false));
        assert_eq!(fb.actual_present, Some(HostTime(1_999_000)));
        assert_eq!(fb.pacing_overrun, None);
    }

    #[test]
    fn pending_feedback_resolve_without_actual_present() {
        let pending = PendingFeedback {
            hints: PresentHints {
                desired_present: Some(HostTime(2_000_000)),
                latest_commit: HostTime(1_800_000),
            },
            build_start: HostTime(1_700_000),
            submitted_at: HostTime(1_750_000),
        };

        // No actual present → falls back to commit deadline (on time).
        let fb = pending.resolve(None);
        assert_eq!(fb.missed_deadline, Some(false));
        assert_eq!(fb.actual_present, None);
        assert_eq!(fb.pacing_overrun, None);
    }

    #[test]
    fn pending_feedback_without_expected_present_stays_unknown() {
        let pending = PendingFeedback {
            hints: PresentHints {
                desired_present: None,
                latest_commit: HostTime(1_800_000),
            },
            build_start: HostTime(1_700_000),
            submitted_at: HostTime(1_950_000),
        };

        let fb = pending.resolve(None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.actual_present, None);
        assert_eq!(fb.pacing_overrun, Some(true));
    }

    #[test]
    fn pacing_only_on_time_submission_reports_no_overrun() {
        let hints = PresentHints {
            desired_present: None,
            latest_commit: HostTime(1_000_000),
        };
        let fb = PresentFeedback::new(&hints, HostTime(900_000), HostTime(950_000), None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.pacing_overrun, Some(false));
    }
}
