// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capability-graded timing model.
//!
//! This module defines the types that flow between backends and the scheduler:
//!
//! - [`PresentationTiming`] — whether presentation timestamps are available
//! - [`DisplayTiming`] — fixed/variable display timing constraints
//! - [`FrameTick`] — a frame opportunity delivered by the backend
//! - [`FrameOpportunity`] — tick, presentation hints, and display timing
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
//! 3. The host combines those facts with [`DisplayTiming`] into a
//!    [`FrameOpportunity`].
//! 4. [`Scheduler::plan()`](crate::scheduler::Scheduler::plan) consumes the
//!    opportunity plus [`FrameDemand`] to produce a [`FramePlan`] with a frame
//!    start time, sampling time, target presentation time, and commit deadline.
//! 5. The application schedules frame work at
//!    [`frame_start`](FramePlan::frame_start), uses the plan's
//!    [`sample_time`](FramePlan::sample_time) to evaluate animation/simulation
//!    state, and builds/submits the frame.
//! 6. After submission, the backend constructs [`PresentFeedback`] from
//!    timing observations and feeds it back to
//!    [`Scheduler::observe()`](crate::scheduler::Scheduler::observe) to
//!    adapt pipeline depth and safety margins.

pub use crate::demand::{FrameDemand, FrameDemandClass};
use crate::output::OutputId;
use crate::time::{Duration, HostTime};

/// Fixed or variable display timing constraints for one output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DisplayTiming {
    min_interval: Duration,
    max_interval: Duration,
    granularity: Option<Duration>,
}

impl DisplayTiming {
    /// Creates fixed-rate display timing.
    #[inline]
    #[must_use]
    pub const fn fixed(interval: Duration) -> Self {
        Self {
            min_interval: interval,
            max_interval: interval,
            granularity: Some(interval),
        }
    }

    /// Creates variable-refresh display timing.
    ///
    /// `min_interval` is the fastest direct display interval, `max_interval`
    /// is the slowest direct display interval, and `granularity` describes a
    /// fixed direct-interval step when the platform exposes one.
    ///
    /// `None` means the backend knows a variable range exists, but does not
    /// know the direct interval granularity or cannot request arbitrary direct
    /// presentation durations. In that case the scheduler uses stable
    /// fixed-rate-like multiples of `min_interval` instead of inventing
    /// unsupported direct intervals inside the range.
    #[inline]
    #[must_use]
    pub const fn variable(
        min_interval: Duration,
        max_interval: Duration,
        granularity: Option<Duration>,
    ) -> Self {
        let max_interval = if max_interval.0 < min_interval.0 {
            min_interval
        } else {
            max_interval
        };
        Self {
            min_interval,
            max_interval,
            granularity,
        }
    }

    /// Creates fixed-rate display timing from a tick's timing facts.
    ///
    /// Prefer [`FrameTick::refresh_interval`] when it is present because it is
    /// the backend's explicit cadence fact. If no refresh interval is reported,
    /// this falls back to the delta from `tick.now` to
    /// [`FrameTick::predicted_present`] and finally to `fallback_interval`.
    #[inline]
    #[must_use]
    pub fn from_tick(tick: &FrameTick, fallback_interval: Duration) -> Self {
        let interval = tick
            .refresh_interval
            .filter(|ticks| *ticks > 0)
            .map(Duration)
            .or_else(|| {
                tick.predicted_present
                    .map(|present| present.saturating_duration_since(tick.now))
                    .filter(|interval| !interval.is_zero())
            })
            .unwrap_or(fallback_interval);
        Self::fixed(interval)
    }

    /// Fastest direct display interval.
    #[inline]
    #[must_use]
    pub const fn min_interval(self) -> Duration {
        self.min_interval
    }

    /// Slowest direct display interval.
    #[inline]
    #[must_use]
    pub const fn max_interval(self) -> Duration {
        self.max_interval
    }

    /// Optional direct interval granularity for displays with known steps.
    ///
    /// `None` is conservative: the scheduler treats direct interval
    /// granularity as unknown and chooses stable multiples of
    /// [`Self::min_interval`].
    #[inline]
    #[must_use]
    pub const fn granularity(self) -> Option<Duration> {
        self.granularity
    }

    /// Returns true when the display exposes a range of direct intervals.
    #[inline]
    #[must_use]
    pub const fn is_variable(self) -> bool {
        self.min_interval.0 != self.max_interval.0
    }

    /// Chooses a stable delivery interval that can contain `needed`.
    ///
    /// Fixed-rate displays and variable displays without known direct
    /// granularity choose multiples of [`Self::min_interval`]. Variable
    /// displays with an explicit granularity choose the first supported direct
    /// interval step that can contain the needed work.
    #[must_use]
    pub fn choose_interval(self, needed: Duration) -> Duration {
        if self.min_interval.is_zero() {
            return needed;
        }

        if !self.is_variable() {
            return round_up_to_multiple(needed, self.min_interval);
        }

        if needed > self.max_interval {
            return round_up_to_multiple(needed, self.min_interval);
        }

        let clamped = clamp_duration(needed, self.min_interval, self.max_interval);
        match self.granularity {
            Some(step) if !step.is_zero() && step != self.min_interval => clamp_duration(
                round_up_to_multiple(clamped, step),
                self.min_interval,
                self.max_interval,
            ),
            Some(_) => clamped,
            None => round_up_to_multiple(clamped, self.min_interval),
        }
    }
}

fn clamp_duration(value: Duration, min_value: Duration, max_value: Duration) -> Duration {
    value.max(min_value).min(max_value)
}

fn round_up_to_multiple(needed: Duration, interval: Duration) -> Duration {
    if interval.is_zero() {
        return needed;
    }
    let count = needed
        .ticks()
        .saturating_add(interval.ticks().saturating_sub(1))
        / interval.ticks();
    interval.saturating_mul(count.max(1))
}

/// How presentation timing should be interpreted for one planning request.
///
/// This is attached to [`PresentHints`], not [`FrameTick`], because it is part
/// of the backend's scheduling contract: it tells the scheduler whether
/// [`PresentHints::desired_present`] may be reported as presentation truth.
///
/// Adaptive scheduling policy is still selected by
/// [`SchedulerConfig`](crate::scheduler::SchedulerConfig). For example, hosts
/// should normally pair [`PresentationTiming::Estimated`] hints with
/// [`SchedulerConfig::estimated`](crate::scheduler::SchedulerConfig::estimated).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PresentationTiming {
    /// Strong predicted present time available (e.g. macOS `CVDisplayLink`).
    Predictive,
    /// Vsync-ish timing but less strict (e.g. Android Choreographer).
    Estimated,
    /// No reliable present time; frame pacing only (e.g. Web `rAF`, X11 fallback).
    PacingOnly,
}

impl PresentationTiming {
    /// Returns whether this timing mode can report a target present time.
    #[inline]
    #[must_use]
    pub const fn has_target_present(self) -> bool {
        matches!(self, Self::Predictive | Self::Estimated)
    }
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
    /// Host-owned monotonically increasing frame counter for this output.
    ///
    /// Keep this stable for the full lifecycle of one planned content frame:
    /// tick, plan, submit/feedback, and drop diagnostics all use this value to
    /// join events. With [`FrameDriver`](crate::FrameDriver), increment it
    /// after an [`ActiveFrame`](crate::ActiveFrame) is submitted or discarded,
    /// not every time a frame-start wake fires while a plan is queued.
    pub frame_index: u64,
    /// Which output this tick is for.
    pub output: OutputId,
    /// Actual present time of the *previous* frame, if the backend can report
    /// it (e.g. from `CADisplayLink.timestamp`).
    pub prev_actual_present: Option<HostTime>,
}

/// A platform frame opportunity for scheduler and retained driver APIs.
///
/// Hosts construct this from the current display/frame callback. It packages
/// the platform tick, backend submission constraints, and output timing model
/// that the scheduler needs to plan a frame.
///
/// `frame_index` lives on [`FrameTick`] and is owned by the host/backend. When
/// using [`FrameDriver`](crate::FrameDriver), advance it after an
/// [`ActiveFrame`](crate::ActiveFrame) is submitted or discarded. Do not
/// advance it merely because a frame-start wake fired while an older planned
/// frame was still queued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameOpportunity {
    /// Platform frame opportunity.
    pub tick: FrameTick,
    /// Backend submission constraints for the opportunity.
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

    /// Creates a pacing-only frame opportunity.
    ///
    /// Use this from hosts that have a monotonic clock and a nominal refresh
    /// interval but no reliable predicted present timestamp. The returned
    /// opportunity uses:
    ///
    /// - [`PresentationTiming::PacingOnly`],
    /// - no predicted or desired present time,
    /// - `latest_commit = now + refresh_interval`, saturating at `u64::MAX`,
    /// - [`DisplayTiming::fixed`] with `refresh_interval`.
    #[inline]
    #[must_use]
    pub fn pacing_only(
        now: HostTime,
        refresh_interval: Duration,
        frame_index: u64,
        output: OutputId,
    ) -> Self {
        let tick = FrameTick {
            now,
            predicted_present: None,
            refresh_interval: Some(refresh_interval.ticks()),
            frame_index,
            output,
            prev_actual_present: None,
        };
        let hints = PresentHints::pacing_only(
            now.checked_add(refresh_interval)
                .unwrap_or(HostTime(u64::MAX)),
        );
        Self::new(tick, hints, DisplayTiming::fixed(refresh_interval))
    }
}

/// The plan for evaluating a single frame.
///
/// Produced by the [`Scheduler`](crate::scheduler::Scheduler) from a
/// [`FrameOpportunity`] and [`FrameDemand`]. All engine evaluation and content
/// selection should be driven by the times in this plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FramePlan {
    /// Demand that selected this frame.
    pub demand: FrameDemand,
    /// Scheduler-selected delivery interval for this frame.
    pub frame_interval: Duration,
    /// Time applications should wake or start app-side frame work.
    ///
    /// This is derived from the commit deadline and scheduler safety margin.
    /// It is clamped to the originating tick time, so callers can request
    /// redraw immediately when this time is already due.
    pub frame_start: HostTime,
    /// Time applications should sample animation and simulation state for.
    pub sample_time: HostTime,
    /// Intended display time, if known.
    pub target_present: Option<HostTime>,
    /// How [`target_present`](Self::target_present) should be interpreted.
    pub presentation_timing: PresentationTiming,
    /// Latest time by which the frame must be committed/submitted.
    pub commit_deadline: HostTime,
    /// Current scheduler pipeline depth.
    pub pipeline_depth: u8,
    /// Which output this frame targets.
    pub output: OutputId,
    /// Frame counter, carried from the originating [`FrameTick`].
    ///
    /// This identifies the planned content frame, not necessarily the host
    /// wake that eventually made the queued frame ready.
    pub frame_index: u64,
}

/// Submission constraints provided by the backend.
///
/// Backends compute these from the current [`FrameTick`] and their own
/// knowledge of the presentation pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentHints {
    /// Whether [`desired_present`](Self::desired_present) can be used as
    /// presentation timing.
    presentation_timing: PresentationTiming,
    /// Target present time, if known.
    desired_present: Option<HostTime>,
    /// Latest time by which a commit must occur to hit the desired present.
    latest_commit: HostTime,
}

impl PresentHints {
    /// Creates hints with explicit presentation timing.
    ///
    /// If `presentation_timing` is [`PresentationTiming::PacingOnly`],
    /// `desired_present` is discarded. Pacing-only backends expose a commit
    /// boundary but not presentation truth.
    #[inline]
    #[must_use]
    pub const fn new(
        presentation_timing: PresentationTiming,
        desired_present: Option<HostTime>,
        latest_commit: HostTime,
    ) -> Self {
        let desired_present = if presentation_timing.has_target_present() {
            desired_present
        } else {
            None
        };
        Self {
            presentation_timing,
            desired_present,
            latest_commit,
        }
    }

    /// Returns how [`Self::desired_present`] should be interpreted.
    #[inline]
    #[must_use]
    pub const fn presentation_timing(self) -> PresentationTiming {
        self.presentation_timing
    }

    /// Returns the target present time when this backend can provide one.
    #[inline]
    #[must_use]
    pub const fn desired_present(self) -> Option<HostTime> {
        self.desired_present
    }

    /// Returns the latest time by which the frame should be committed.
    #[inline]
    #[must_use]
    pub const fn latest_commit(self) -> HostTime {
        self.latest_commit
    }

    /// Creates predictive present hints.
    #[inline]
    #[must_use]
    pub const fn predictive(desired_present: HostTime, latest_commit: HostTime) -> Self {
        Self::new(
            PresentationTiming::Predictive,
            Some(desired_present),
            latest_commit,
        )
    }

    /// Creates estimated present hints.
    #[inline]
    #[must_use]
    pub const fn estimated(desired_present: HostTime, latest_commit: HostTime) -> Self {
        Self::new(
            PresentationTiming::Estimated,
            Some(desired_present),
            latest_commit,
        )
    }

    /// Creates pacing-only hints with no presentation timestamp.
    #[inline]
    #[must_use]
    pub const fn pacing_only(latest_commit: HostTime) -> Self {
        Self::new(PresentationTiming::PacingOnly, None, latest_commit)
    }
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
    /// - If both `actual_present` and [`PresentHints::desired_present`] are known, a
    ///   frame is missed when `actual_present > desired_present`.
    /// - Otherwise, the result is `None`: without actual presentation
    ///   feedback, commit timing is useful pacing evidence but not real
    ///   presentation truth.
    ///
    /// When the frame cannot be classified as a real hit or miss,
    /// [`PresentFeedback::pacing_overrun`] is populated from commit timing as
    /// the weaker pacing signal.
    #[must_use]
    pub fn new(
        hints: &PresentHints,
        build_start: HostTime,
        submitted_at: HostTime,
        actual_present: Option<HostTime>,
    ) -> Self {
        let expected_present = if hints.presentation_timing().has_target_present() {
            hints.desired_present()
        } else {
            None
        };
        // Only report deadline truth when the backend can honestly support it.
        let missed_deadline = match (actual_present, expected_present) {
            (Some(actual), Some(expected)) => Some(actual > expected),
            (_, None) => None,
            (None, Some(_)) => None,
        };
        let pacing_overrun = if missed_deadline.is_none() {
            Some(submitted_at > hints.latest_commit())
        } else {
            None
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

    fn tick_with_timing(
        now: u64,
        predicted: Option<u64>,
        refresh_interval: Option<u64>,
    ) -> FrameTick {
        FrameTick {
            now: HostTime(now),
            predicted_present: predicted.map(HostTime),
            refresh_interval,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    #[test]
    fn display_timing_from_tick_prefers_reported_refresh_interval() {
        let tick = tick_with_timing(10_000_000, Some(11_000_000), Some(16_666_667));

        let display = DisplayTiming::from_tick(&tick, Duration(8_333_333));

        assert_eq!(display, DisplayTiming::fixed(Duration(16_666_667)));
    }

    #[test]
    fn display_timing_from_tick_falls_back_to_predicted_delta() {
        let tick = tick_with_timing(10_000_000, Some(26_666_667), None);

        let display = DisplayTiming::from_tick(&tick, Duration(8_333_333));

        assert_eq!(display, DisplayTiming::fixed(Duration(16_666_667)));
    }

    #[test]
    fn display_timing_from_tick_ignores_stale_prediction() {
        let tick = tick_with_timing(20_000_000, Some(10_000_000), None);

        let display = DisplayTiming::from_tick(&tick, Duration(16_666_667));

        assert_eq!(display, DisplayTiming::fixed(Duration(16_666_667)));
    }

    #[test]
    fn pacing_only_hints_discard_desired_present() {
        let hints = PresentHints::new(
            PresentationTiming::PacingOnly,
            Some(HostTime(2_000_000)),
            HostTime(1_000_000),
        );

        assert_eq!(hints.presentation_timing(), PresentationTiming::PacingOnly);
        assert_eq!(hints.desired_present(), None);
        assert_eq!(hints.latest_commit(), HostTime(1_000_000));
    }

    #[test]
    fn new_with_actual_present_compares_to_expected() {
        let hints = PresentHints::predictive(HostTime(2_000_000), HostTime(1_800_000));
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
    fn new_without_actual_present_uses_commit_deadline_as_pacing_evidence() {
        let hints = PresentHints::predictive(HostTime(2_000_000), HostTime(1_800_000));
        // submitted_at > latest_commit is weak pacing evidence until the
        // backend reports actual present.
        let fb = PresentFeedback::new(&hints, HostTime(1_700_000), HostTime(1_900_000), None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.pacing_overrun, Some(true));

        // submitted_at <= latest_commit is still not presentation truth.
        let fb = PresentFeedback::new(&hints, HostTime(1_700_000), HostTime(1_750_000), None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.pacing_overrun, Some(false));
    }

    #[test]
    fn new_without_desired_present_is_unknown() {
        let hints = PresentHints::pacing_only(HostTime(1_000_000));
        let fb = PresentFeedback::new(&hints, HostTime(900_000), HostTime(1_100_000), None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.expected_present, None);
        assert_eq!(fb.pacing_overrun, Some(true));
    }

    #[test]
    fn new_with_actual_present_but_no_expected_present_is_unknown() {
        let hints = PresentHints::pacing_only(HostTime(1_000_000));
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
            hints: PresentHints::predictive(HostTime(2_000_000), HostTime(1_800_000)),
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
            hints: PresentHints::predictive(HostTime(2_000_000), HostTime(1_800_000)),
            build_start: HostTime(1_700_000),
            submitted_at: HostTime(1_750_000),
        };

        // No actual present → commit timing is pacing evidence, not deadline truth.
        let fb = pending.resolve(None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.actual_present, None);
        assert_eq!(fb.pacing_overrun, Some(false));
    }

    #[test]
    fn pending_feedback_without_expected_present_stays_unknown() {
        let pending = PendingFeedback {
            hints: PresentHints::pacing_only(HostTime(1_800_000)),
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
        let hints = PresentHints::pacing_only(HostTime(1_000_000));
        let fb = PresentFeedback::new(&hints, HostTime(900_000), HostTime(950_000), None);
        assert_eq!(fb.missed_deadline, None);
        assert_eq!(fb.pacing_overrun, Some(false));
    }
}
