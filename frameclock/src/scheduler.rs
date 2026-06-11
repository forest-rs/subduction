// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame scheduling with adaptive pipeline depth and safety margins.
//!
//! The [`Scheduler`] converts a [`FrameOpportunity`] and [`FrameDemand`] into
//! a [`FramePlan`], adapting its pipeline depth and safety margin based on
//! observed [`PresentFeedback`]. See the [`Scheduler`] struct docs for
//! details on pipeline depth and adaptive behavior.

use crate::demand::{FrameDemand, FrameDemandClass};
use crate::time::{Duration, HostTime};
use crate::timing::{
    DisplayTiming, FrameOpportunity, FramePlan, PresentFeedback, PresentationTiming,
};

/// Controls how the scheduler adapts pipeline depth in response to deadline
/// misses and hits.
///
/// Passed to the [`Scheduler`] via [`SchedulerConfig::degradation_policy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DegradationPolicy {
    /// Adapt pipeline depth automatically.
    ///
    /// Increases depth by 1 after `miss_threshold` consecutive misses.
    /// Decreases depth by 1 after `recovery_threshold` consecutive hits.
    /// Bounded by [`SchedulerConfig::min_depth`] and
    /// [`SchedulerConfig::max_depth`].
    Adaptive {
        /// Number of consecutive misses before increasing depth.
        miss_threshold: u32,
        /// Number of consecutive hits before decreasing depth.
        recovery_threshold: u32,
    },
    /// Keep pipeline depth fixed at [`SchedulerConfig::initial_depth`].
    ///
    /// Build cost EMA and safety margin are still tracked, but depth
    /// never changes.
    Fixed,
}

/// Configuration for the [`Scheduler`].
#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// Initial pipeline depth.
    ///
    /// A value of 1 means no extra whole-frame lookahead. Higher values shift
    /// non-input plans further into the future by whole selected frame
    /// intervals.
    pub initial_depth: u8,
    /// Minimum pipeline depth.
    pub min_depth: u8,
    /// Maximum pipeline depth.
    pub max_depth: u8,
    /// EMA smoothing factor for build cost estimation (0.0–1.0).
    /// Smaller values = more smoothing.
    pub ema_alpha: f64,
    /// Safety margin multiplier applied to the EMA build cost.
    pub safety_multiplier: f64,
    /// Minimum margin subtracted from the commit deadline to compute
    /// [`FramePlan::frame_start`].
    ///
    /// The scheduler uses the larger of this value and the learned safety
    /// margin. This gives first-frame scheduling a conservative work window
    /// before feedback has trained the build-cost estimate.
    pub minimum_frame_start_margin: Duration,
    /// Nominal latency (in ticks) used for pacing-only mode when no predicted
    /// present time is available.
    pub nominal_latency: Duration,
    /// Policy for adapting pipeline depth.
    pub degradation_policy: DegradationPolicy,
}

impl SchedulerConfig {
    /// Default configuration for predictive timing.
    ///
    /// Use this when a platform exposes a strong predicted presentation time.
    #[must_use]
    pub const fn predictive() -> Self {
        Self {
            initial_depth: 1,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.2,
            safety_multiplier: 1.5,
            minimum_frame_start_margin: Duration(1_000_000),
            nominal_latency: Duration(0),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }

    /// Default configuration for estimated timing.
    ///
    /// Use this when a platform exposes useful but less reliable predicted
    /// presentation timing.
    #[must_use]
    pub const fn estimated() -> Self {
        Self {
            initial_depth: 1,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.2,
            safety_multiplier: 2.0,
            minimum_frame_start_margin: Duration(1_000_000),
            nominal_latency: Duration(0),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }

    /// Default configuration for pacing-only timing.
    ///
    /// Use this when a platform exposes frame pacing but no reliable target
    /// presentation time.
    #[must_use]
    pub const fn pacing_only() -> Self {
        Self {
            initial_depth: 1,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.15,
            safety_multiplier: 2.0,
            minimum_frame_start_margin: Duration(1_000_000),
            // ~16ms at 1ns tick resolution.
            nominal_latency: Duration(16_000_000),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }
}

/// Snapshot of scheduler adaptation state for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedulerState {
    /// Current pipeline depth.
    pub pipeline_depth: u8,
    /// Current estimated safety margin in host-time ticks.
    pub safety_margin_ticks: u64,
    /// Consecutive strong misses or pacing overruns currently accumulated.
    pub consecutive_misses: u32,
    /// Consecutive strong hits currently accumulated.
    pub consecutive_hits: u32,
}

/// Exponential moving average tracker.
#[derive(Clone, Copy, Debug)]
struct Ema {
    value: f64,
    alpha: f64,
    initialized: bool,
}

impl Ema {
    const fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha,
            initialized: false,
        }
    }

    fn update(&mut self, sample: f64) {
        if self.initialized {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        } else {
            self.value = sample;
            self.initialized = true;
        }
    }

    const fn get(&self) -> f64 {
        self.value
    }

    const fn initialized(&self) -> bool {
        self.initialized
    }
}

fn sanitize_ema_alpha(alpha: f64) -> f64 {
    if !alpha.is_finite() {
        return 1.0;
    }
    alpha.clamp(0.0, 1.0)
}

fn sanitize_safety_multiplier(multiplier: f64) -> f64 {
    if !multiplier.is_finite() || multiplier < 0.0 {
        return 1.0;
    }
    multiplier
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite non-negative tick values are clamped before conversion"
)]
fn f64_ticks_to_u64(ticks: f64) -> u64 {
    if !ticks.is_finite() || ticks <= 0.0 {
        return 0;
    }
    if ticks >= u64::MAX as f64 {
        return u64::MAX;
    }
    ticks as u64
}

/// Frame scheduler that converts [`FrameOpportunity`] values into
/// [`FramePlan`]s and adapts over time.
///
/// # Pipeline depth
///
/// Pipeline depth controls how many selected frame intervals ahead the
/// scheduler plans non-input work, bounded by [`SchedulerConfig::min_depth`]
/// and [`SchedulerConfig::max_depth`]. A depth of 1 means no extra whole-frame
/// lookahead (lowest latency, highest risk of missing deadlines). A depth of 3
/// means animation/background/continuous-input plans target two selected
/// intervals beyond the next eligible present slot. One-shot input remains
/// latency-first and is not shifted by pipeline depth.
///
/// # Adaptive behavior
///
/// With [`DegradationPolicy::Adaptive`] (set via
/// [`SchedulerConfig::degradation_policy`]), the scheduler increases depth
/// after consecutive deadline misses (trading latency for safety) and
/// decreases depth after sustained hits (reclaiming latency). The
/// EMA-smoothed build cost feeds into a safety margin (build cost ×
/// multiplier) that backends can query via
/// [`Scheduler::safety_margin_ticks()`]. The [`FramePlan::frame_start`] field
/// applies that margin directly so hosts can schedule a redraw wake without
/// duplicating scheduler policy.
///
/// # Usage
///
/// ```rust,ignore
/// let plan = scheduler.plan(opportunity, demand);
/// // ... build and submit frame ...
/// scheduler.observe(&feedback);
/// ```
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    pipeline_depth: u8,
    build_cost_ema: Ema,
    safety_margin_ticks: u64,
    consecutive_misses: u32,
    consecutive_hits: u32,
}

impl Scheduler {
    /// Creates a new scheduler with the given configuration.
    #[must_use]
    pub fn new(mut config: SchedulerConfig) -> Self {
        config.min_depth = config.min_depth.max(1);
        config.max_depth = config.max_depth.max(config.min_depth);
        config.initial_depth = config
            .initial_depth
            .clamp(config.min_depth, config.max_depth);
        config.ema_alpha = sanitize_ema_alpha(config.ema_alpha);
        config.safety_multiplier = sanitize_safety_multiplier(config.safety_multiplier);

        Self {
            pipeline_depth: config.initial_depth,
            build_cost_ema: Ema::new(config.ema_alpha),
            safety_margin_ticks: 0,
            consecutive_misses: 0,
            consecutive_hits: 0,
            config,
        }
    }

    /// Produces a [`FramePlan`] for the given frame opportunity and demand.
    ///
    /// Hosts should usually call this only with non-empty [`FrameDemand`].
    /// `FrameDemand::NONE` is accepted for passive pacing diagnostics or
    /// backend bookkeeping, but it should not be treated as ordinary render
    /// demand.
    #[must_use]
    pub fn plan(&mut self, opportunity: FrameOpportunity, demand: FrameDemand) -> FramePlan {
        let tick = opportunity.tick;
        let hints = opportunity.hints;
        let source_interval = self.source_interval(opportunity);
        let build_cost = self.build_cost_estimate();
        let frame_interval = self.frame_interval(
            demand,
            opportunity.display_timing,
            source_interval,
            build_cost,
        );
        let schedule_delta = self.schedule_delta(demand, frame_interval, source_interval);
        let presentation_timing = hints.presentation_timing();
        let platform_present = if presentation_timing.has_target_present() {
            hints
                .desired_present()
                .filter(|present| *present >= tick.now)
        } else {
            None
        };
        let base_present = platform_present
            .unwrap_or_else(|| tick.now.checked_add(source_interval).unwrap_or(tick.now));
        let scheduled_present = base_present
            .checked_add(schedule_delta)
            .unwrap_or(base_present);
        let base_commit_deadline = hints.latest_commit().max(tick.now);
        let commit_deadline = base_commit_deadline
            .checked_add(schedule_delta)
            .unwrap_or(base_commit_deadline);

        let (target_present, sample_time) = match presentation_timing {
            PresentationTiming::Predictive | PresentationTiming::Estimated => {
                let target_present = platform_present.map(|_| scheduled_present);
                let sample_time = target_present.unwrap_or(scheduled_present);
                (target_present, sample_time)
            }
            PresentationTiming::PacingOnly => {
                // No reliable present time; sample at the scheduler-selected
                // pacing target but do not report it as presentation truth.
                (None, scheduled_present)
            }
        };

        FramePlan {
            demand,
            frame_interval,
            frame_start: self.frame_start(tick.now, commit_deadline, demand),
            sample_time,
            target_present,
            presentation_timing,
            commit_deadline,
            pipeline_depth: self.pipeline_depth,
            output: tick.output,
            frame_index: tick.frame_index,
        }
    }

    fn schedule_delta(
        &self,
        demand: FrameDemand,
        frame_interval: Duration,
        source_interval: Duration,
    ) -> Duration {
        let cadence_delta = frame_interval.saturating_sub(source_interval);
        let depth_delta = self.depth_lookahead_delta(demand, frame_interval);
        cadence_delta.saturating_add(depth_delta)
    }

    fn depth_lookahead_delta(&self, demand: FrameDemand, frame_interval: Duration) -> Duration {
        if demand.dominant_class() == FrameDemandClass::Input {
            return Duration::ZERO;
        }

        frame_interval.saturating_mul(u64::from(self.pipeline_depth.saturating_sub(1)))
    }

    fn source_interval(&self, opportunity: FrameOpportunity) -> Duration {
        opportunity
            .tick
            .refresh_interval
            .filter(|ticks| *ticks > 0)
            .map(Duration)
            .unwrap_or_else(|| {
                let display_min = opportunity.display_timing.min_interval();
                if display_min.is_zero() {
                    self.config.nominal_latency
                } else {
                    display_min
                }
            })
    }

    fn frame_interval(
        &self,
        demand: FrameDemand,
        display: DisplayTiming,
        source_interval: Duration,
        build_cost: Duration,
    ) -> Duration {
        let source_interval = if source_interval.is_zero() {
            display.min_interval()
        } else {
            source_interval
        };
        let policy = demand.dominant_class();
        let needed = match policy {
            FrameDemandClass::None | FrameDemandClass::Input => return source_interval,
            FrameDemandClass::ContinuousInput => build_cost,
            FrameDemandClass::Animation => build_cost.saturating_add(source_interval.div_u64(4)),
            FrameDemandClass::Background => build_cost
                .saturating_add(source_interval.div_u64(4))
                .max(source_interval.saturating_mul(2)),
        };

        display.choose_interval(needed).max(source_interval)
    }

    fn frame_start(
        &self,
        now: HostTime,
        commit_deadline: HostTime,
        demand: FrameDemand,
    ) -> HostTime {
        if demand.dominant_class() == FrameDemandClass::Input {
            return now;
        }

        commit_deadline
            .checked_sub(self.frame_start_margin())
            .unwrap_or(now)
            .max(now)
    }

    fn frame_start_margin(&self) -> Duration {
        let learned = Duration(self.safety_margin_ticks);
        if learned > self.config.minimum_frame_start_margin {
            learned
        } else {
            self.config.minimum_frame_start_margin
        }
    }

    fn build_cost_estimate(&self) -> Duration {
        if !self.build_cost_ema.initialized() {
            return Duration::ZERO;
        }

        Duration(f64_ticks_to_u64(self.build_cost_ema.get()))
    }

    /// Feeds presentation feedback to adapt scheduling parameters.
    pub fn observe(&mut self, feedback: &PresentFeedback) {
        // Update build cost EMA.
        let build_ticks = feedback
            .submitted_at
            .saturating_duration_since(feedback.build_start)
            .ticks();
        self.build_cost_ema.update(build_ticks as f64);

        // Update safety margin.
        self.safety_margin_ticks =
            f64_ticks_to_u64(self.build_cost_ema.get() * self.config.safety_multiplier);

        // Adapt pipeline depth according to degradation policy.
        //
        // `missed_deadline` is the strong signal: the backend believes it can
        // classify the frame as a real hit or miss.
        // `pacing_overrun` is weaker: it only says we ran past a pacing
        // boundary. We still use it, but more conservatively, so pacing-only
        // backends can apply pressure without pretending they know actual
        // presentation truth.
        match self.config.degradation_policy {
            DegradationPolicy::Adaptive {
                miss_threshold,
                recovery_threshold,
            } => match feedback.missed_deadline {
                Some(true) => {
                    // Real miss: react using the normal threshold.
                    self.consecutive_misses += 1;
                    self.consecutive_hits = 0;
                    if self.consecutive_misses >= miss_threshold
                        && self.pipeline_depth < self.config.max_depth
                    {
                        self.pipeline_depth += 1;
                        self.consecutive_misses = 0;
                    }
                }
                Some(false) => {
                    // Real hit: count toward recovery.
                    self.consecutive_hits += 1;
                    self.consecutive_misses = 0;
                    if self.consecutive_hits >= recovery_threshold
                        && self.pipeline_depth > self.config.min_depth
                    {
                        self.pipeline_depth -= 1;
                        self.consecutive_hits = 0;
                    }
                }
                None => match feedback.pacing_overrun {
                    Some(true) => {
                        // Pacing-only overrun is weaker than a real miss, so
                        // require more repeated evidence before raising depth.
                        self.consecutive_misses += 1;
                        self.consecutive_hits = 0;
                        let pacing_threshold = miss_threshold.saturating_mul(2).max(1);
                        if self.consecutive_misses >= pacing_threshold
                            && self.pipeline_depth < self.config.max_depth
                        {
                            self.pipeline_depth += 1;
                            self.consecutive_misses = 0;
                        }
                    }
                    Some(false) | None => {
                        // Unknown or clear pacing feedback should not pretend
                        // to be a hit. Reset adaptation counters and continue
                        // using the build-cost EMA for safety-margin training.
                        self.consecutive_misses = 0;
                        self.consecutive_hits = 0;
                    }
                },
            },
            DegradationPolicy::Fixed => {}
        }
    }

    /// Returns the current pipeline depth.
    #[must_use]
    pub fn pipeline_depth(&self) -> u8 {
        self.pipeline_depth
    }

    /// Returns the current estimated safety margin in ticks.
    #[must_use]
    pub fn safety_margin_ticks(&self) -> u64 {
        self.safety_margin_ticks
    }

    /// Returns a snapshot of the current scheduler adaptation state.
    #[must_use]
    pub const fn state(&self) -> SchedulerState {
        SchedulerState {
            pipeline_depth: self.pipeline_depth,
            safety_margin_ticks: self.safety_margin_ticks,
            consecutive_misses: self.consecutive_misses,
            consecutive_hits: self.consecutive_hits,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::output::OutputId;
    use crate::time::HostTime;
    use crate::timing::{FrameTick, PresentHints};

    use super::*;

    const REFRESH_INTERVAL: Duration = Duration(16_666_667);

    fn make_tick(now: u64, predicted: Option<u64>) -> FrameTick {
        FrameTick {
            now: HostTime(now),
            predicted_present: predicted.map(HostTime),
            refresh_interval: Some(REFRESH_INTERVAL.ticks()),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    fn make_hints(
        presentation_timing: PresentationTiming,
        predicted: Option<u64>,
        deadline: u64,
    ) -> PresentHints {
        if let Some(predicted) = predicted
            && presentation_timing.has_target_present()
        {
            return PresentHints::new(
                presentation_timing,
                Some(HostTime(predicted)),
                HostTime(deadline),
            );
        }

        PresentHints::new(presentation_timing, None, HostTime(deadline))
    }

    fn make_opportunity(
        presentation_timing: PresentationTiming,
        now: u64,
        predicted: Option<u64>,
        deadline: u64,
    ) -> FrameOpportunity {
        let tick = make_tick(now, predicted);
        let hints = make_hints(presentation_timing, predicted, deadline);
        FrameOpportunity {
            tick,
            hints,
            display_timing: DisplayTiming::fixed(REFRESH_INTERVAL),
        }
    }

    #[test]
    fn predictive_plan_uses_predicted_present() {
        let config = SchedulerConfig::predictive();
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1000, Some(2000), 1800),
            FrameDemand::ANIMATION,
        );

        assert_eq!(plan.demand, FrameDemand::ANIMATION);
        assert_eq!(plan.frame_interval, REFRESH_INTERVAL);
        assert_eq!(plan.target_present, Some(HostTime(2000)));
        assert_eq!(plan.sample_time, HostTime(2000));
        assert_eq!(plan.commit_deadline, HostTime(1800));
        assert_eq!(plan.frame_start, HostTime(1000));
    }

    #[test]
    fn pacing_only_plan_has_no_target_present() {
        let config = SchedulerConfig::pacing_only();
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::PacingOnly, 1_000_000, None, 17_000_000),
            FrameDemand::ANIMATION,
        );

        assert_eq!(plan.target_present, None);
        // sample_time = now + the selected pacing interval.
        assert_eq!(plan.sample_time, HostTime(17_666_667));
        assert_eq!(plan.frame_start, HostTime(16_000_000));
    }

    #[test]
    fn frame_start_uses_configured_start_margin() {
        let mut config = SchedulerConfig::predictive();
        config.minimum_frame_start_margin = Duration(250);
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1000, Some(2000), 1800),
            FrameDemand::ANIMATION,
        );

        assert_eq!(plan.frame_start, HostTime(1550));
    }

    #[test]
    fn frame_start_uses_learned_safety_margin_when_larger() {
        let mut config = SchedulerConfig::predictive();
        config.minimum_frame_start_margin = Duration(100);
        config.safety_multiplier = 2.0;
        let mut sched = Scheduler::new(config);
        let feedback = PresentFeedback {
            submitted_at: HostTime(1_200),
            build_start: HostTime(1_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };
        sched.observe(&feedback);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1_000, Some(2_000), 2_000),
            FrameDemand::ANIMATION,
        );

        assert_eq!(sched.safety_margin_ticks(), 400);
        assert_eq!(plan.frame_start, HostTime(1_600));
    }

    #[test]
    fn scheduler_sanitizes_non_finite_float_config() {
        let mut config = SchedulerConfig::predictive();
        config.ema_alpha = f64::NAN;
        config.safety_multiplier = f64::INFINITY;
        let mut sched = Scheduler::new(config);
        let feedback = PresentFeedback {
            submitted_at: HostTime(1_200),
            build_start: HostTime(1_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };

        sched.observe(&feedback);

        assert_eq!(sched.safety_margin_ticks(), 200);
    }

    #[test]
    fn frame_start_clamps_to_tick_now_when_start_is_due() {
        let mut config = SchedulerConfig::predictive();
        config.minimum_frame_start_margin = Duration(1_000);
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1_500, Some(2_000), 1_800),
            FrameDemand::ANIMATION,
        );

        assert_eq!(plan.frame_start, HostTime(1_500));
    }

    #[test]
    fn stale_predictive_present_is_ignored() {
        let mut config = SchedulerConfig::predictive();
        config.minimum_frame_start_margin = Duration(250);
        let mut sched = Scheduler::new(config);
        let tick = make_tick(2_000, Some(1_500));
        let opportunity = FrameOpportunity {
            tick,
            hints: PresentHints::predictive(HostTime(1_500), HostTime(1_250)),
            display_timing: DisplayTiming::fixed(REFRESH_INTERVAL),
        };

        let plan = sched.plan(opportunity, FrameDemand::ANIMATION);

        assert_eq!(plan.target_present, None);
        assert_eq!(plan.sample_time, HostTime(2_000) + REFRESH_INTERVAL);
        assert_eq!(plan.commit_deadline, HostTime(2_000));
        assert_eq!(plan.frame_start, HostTime(2_000));
    }

    #[test]
    fn input_demand_starts_immediately() {
        let mut config = SchedulerConfig::predictive();
        config.minimum_frame_start_margin = Duration(250);
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1_000, Some(2_000), 1_800),
            FrameDemand::INPUT,
        );

        assert_eq!(plan.demand, FrameDemand::INPUT);
        assert_eq!(plan.frame_interval, REFRESH_INTERVAL);
        assert_eq!(plan.frame_start, HostTime(1_000));
    }

    #[test]
    fn animation_uses_stable_fixed_rate_divisor_when_work_is_slow() {
        let mut config = SchedulerConfig::predictive();
        config.ema_alpha = 1.0;
        config.safety_multiplier = 1.0;
        let mut sched = Scheduler::new(config);
        let feedback = PresentFeedback {
            submitted_at: HostTime(21_000_000),
            build_start: HostTime(1_000_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };
        sched.observe(&feedback);

        let plan = sched.plan(
            make_opportunity(
                PresentationTiming::Predictive,
                1_000_000,
                Some(17_666_667),
                16_666_667,
            ),
            FrameDemand::ANIMATION,
        );

        assert_eq!(plan.frame_interval, REFRESH_INTERVAL.saturating_mul(2));
        assert_eq!(plan.target_present, Some(HostTime(34_333_334)));
        assert_eq!(plan.commit_deadline, HostTime(33_333_334));
        assert_eq!(plan.frame_start, HostTime(13_333_334));
    }

    #[test]
    fn safety_multiplier_does_not_force_cadence_drop() {
        let mut config = SchedulerConfig::predictive();
        config.ema_alpha = 1.0;
        config.safety_multiplier = 4.0;
        config.minimum_frame_start_margin = Duration::ZERO;
        let mut sched = Scheduler::new(config);
        let feedback = PresentFeedback {
            submitted_at: HostTime(11_000_000),
            build_start: HostTime(1_000_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };
        sched.observe(&feedback);

        let plan = sched.plan(
            make_opportunity(
                PresentationTiming::Predictive,
                1_000_000,
                Some(50_000_000),
                50_000_000,
            ),
            FrameDemand::ANIMATION,
        );

        assert_eq!(sched.safety_margin_ticks(), 40_000_000);
        assert_eq!(plan.frame_interval, REFRESH_INTERVAL);
        assert_eq!(plan.commit_deadline, HostTime(50_000_000));
        assert_eq!(plan.frame_start, HostTime(10_000_000));
    }

    #[test]
    fn variable_refresh_without_granularity_uses_stable_divisor() {
        let mut config = SchedulerConfig::predictive();
        config.ema_alpha = 1.0;
        config.safety_multiplier = 1.0;
        let mut sched = Scheduler::new(config);
        let feedback = PresentFeedback {
            submitted_at: HostTime(13_000_000),
            build_start: HostTime(1_000_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };
        sched.observe(&feedback);
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(9_333_333)),
            refresh_interval: Some(8_333_333),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let opportunity = FrameOpportunity {
            tick,
            hints: PresentHints::predictive(HostTime(9_333_333), HostTime(8_333_333)),
            display_timing: DisplayTiming::variable(
                Duration(8_333_333),
                Duration(33_333_333),
                None,
            ),
        };

        let plan = sched.plan(opportunity, FrameDemand::ANIMATION);

        assert_eq!(plan.frame_interval, Duration(16_666_666));
        assert_eq!(plan.target_present, Some(HostTime(17_666_666)));
        assert_eq!(plan.commit_deadline, HostTime(16_666_666));
        assert_eq!(plan.frame_start, HostTime(4_666_666));
    }

    #[test]
    fn variable_refresh_with_granularity_uses_direct_step() {
        let mut config = SchedulerConfig::predictive();
        config.ema_alpha = 1.0;
        config.safety_multiplier = 1.0;
        let mut sched = Scheduler::new(config);
        let feedback = PresentFeedback {
            submitted_at: HostTime(13_000_000),
            build_start: HostTime(1_000_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };
        sched.observe(&feedback);
        let tick = FrameTick {
            now: HostTime(1_000_000),
            predicted_present: Some(HostTime(9_333_333)),
            refresh_interval: Some(8_333_333),
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        };
        let opportunity = FrameOpportunity {
            tick,
            hints: PresentHints::predictive(HostTime(9_333_333), HostTime(8_333_333)),
            display_timing: DisplayTiming::variable(
                Duration(8_333_333),
                Duration(33_333_333),
                Some(Duration(1_000_000)),
            ),
        };

        let plan = sched.plan(opportunity, FrameDemand::ANIMATION);

        assert_eq!(plan.frame_interval, Duration(15_000_000));
        assert_eq!(plan.target_present, Some(HostTime(16_000_000)));
        assert_eq!(plan.commit_deadline, HostTime(15_000_000));
        assert_eq!(plan.frame_start, HostTime(3_000_000));
    }

    #[test]
    fn pipeline_depth_increases_after_misses() {
        let config = SchedulerConfig::predictive();
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.pipeline_depth(), 1);

        let feedback = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };

        sched.observe(&feedback);
        assert_eq!(sched.pipeline_depth(), 1); // 1 miss
        sched.observe(&feedback);
        assert_eq!(sched.pipeline_depth(), 1); // 2 misses
        sched.observe(&feedback);
        assert_eq!(sched.pipeline_depth(), 2); // 3 misses → increase
    }

    #[test]
    fn pipeline_depth_shifts_non_input_plan_by_whole_intervals() {
        let mut config = SchedulerConfig::predictive();
        config.initial_depth = 3;
        config.minimum_frame_start_margin = Duration(250);
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1_000, Some(2_000), 1_800),
            FrameDemand::ANIMATION,
        );

        let lookahead = REFRESH_INTERVAL.saturating_mul(2);
        assert_eq!(plan.pipeline_depth, 3);
        assert_eq!(plan.frame_interval, REFRESH_INTERVAL);
        assert_eq!(plan.target_present, Some(HostTime(2_000) + lookahead));
        assert_eq!(plan.sample_time, HostTime(2_000) + lookahead);
        assert_eq!(plan.commit_deadline, HostTime(1_800) + lookahead);
        assert_eq!(
            plan.frame_start,
            HostTime(1_800) + lookahead - Duration(250)
        );
    }

    #[test]
    fn pipeline_depth_does_not_shift_one_shot_input() {
        let mut config = SchedulerConfig::predictive();
        config.initial_depth = 3;
        config.minimum_frame_start_margin = Duration(250);
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Predictive, 1_000, Some(2_000), 1_800),
            FrameDemand::INPUT,
        );

        assert_eq!(plan.pipeline_depth, 3);
        assert_eq!(plan.target_present, Some(HostTime(2_000)));
        assert_eq!(plan.sample_time, HostTime(2_000));
        assert_eq!(plan.commit_deadline, HostTime(1_800));
        assert_eq!(plan.frame_start, HostTime(1_000));
    }

    #[test]
    fn consecutive_miss_counter_resets_on_success() {
        let config = SchedulerConfig::predictive();
        let mut sched = Scheduler::new(config);

        let miss = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };
        let hit = PresentFeedback {
            missed_deadline: Some(false),
            ..miss
        };

        sched.observe(&miss);
        sched.observe(&miss);
        sched.observe(&hit); // resets counter
        sched.observe(&miss);
        sched.observe(&miss);
        // Only 2 consecutive misses, should not increase.
        assert_eq!(sched.pipeline_depth(), 1);
    }

    #[test]
    fn pipeline_depth_decreases_after_sustained_hits() {
        let mut config = SchedulerConfig::predictive();
        config.initial_depth = 3;
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.pipeline_depth(), 3);

        let hit = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };

        for _ in 0..9 {
            sched.observe(&hit);
        }
        assert_eq!(sched.pipeline_depth(), 3); // 9 hits, not enough
        sched.observe(&hit); // 10th hit → decrease
        assert_eq!(sched.pipeline_depth(), 2);
    }

    #[test]
    fn fixed_policy_never_changes_depth() {
        let mut config = SchedulerConfig::predictive();
        config.degradation_policy = DegradationPolicy::Fixed;
        config.initial_depth = 2;
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.pipeline_depth(), 2);

        let miss = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };

        for _ in 0..10 {
            sched.observe(&miss);
        }
        assert_eq!(sched.pipeline_depth(), 2); // unchanged

        // Safety margin should still be tracked.
        assert!(sched.safety_margin_ticks() > 0);
    }

    #[test]
    fn custom_miss_threshold() {
        let mut config = SchedulerConfig::predictive();
        config.degradation_policy = DegradationPolicy::Adaptive {
            miss_threshold: 5,
            recovery_threshold: 10,
        };
        let mut sched = Scheduler::new(config);

        let miss = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };

        for _ in 0..4 {
            sched.observe(&miss);
        }
        assert_eq!(sched.pipeline_depth(), 1); // 4 misses, threshold is 5
        sched.observe(&miss); // 5th miss → increase
        assert_eq!(sched.pipeline_depth(), 2);
    }

    #[test]
    fn recovery_respects_min_depth() {
        let mut config = SchedulerConfig::predictive();
        config.initial_depth = 2;
        config.degradation_policy = DegradationPolicy::Adaptive {
            miss_threshold: 3,
            recovery_threshold: 2,
        };
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.pipeline_depth(), 2);

        let hit = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };

        // 2 hits → decrease to 1 (min_depth).
        sched.observe(&hit);
        sched.observe(&hit);
        assert_eq!(sched.pipeline_depth(), 1);

        // 2 more hits → should stay at 1, can't go below min_depth.
        sched.observe(&hit);
        sched.observe(&hit);
        assert_eq!(sched.pipeline_depth(), 1);
    }

    #[test]
    fn estimated_presentation_timing_uses_desired_present() {
        let config = SchedulerConfig::predictive();
        let mut sched = Scheduler::new(config);

        let plan = sched.plan(
            make_opportunity(PresentationTiming::Estimated, 1000, Some(2000), 1800),
            FrameDemand::ANIMATION,
        );

        // Estimated behaves like Predictive for target selection; hosts choose
        // a more conservative SchedulerConfig separately.
        assert_eq!(plan.target_present, Some(HostTime(2000)));
        assert_eq!(plan.sample_time, HostTime(2000));
    }

    #[test]
    fn observe_with_unknown_deadline_resets_counters() {
        let config = SchedulerConfig::predictive();
        let mut sched = Scheduler::new(config);

        let miss = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };
        let unknown = PresentFeedback {
            missed_deadline: None,
            pacing_overrun: None,
            ..miss
        };

        // Accumulate 2 misses, then an unknown.
        sched.observe(&miss);
        sched.observe(&miss);
        sched.observe(&unknown); // resets both counters

        // Two more misses — total consecutive is only 2, not 4.
        sched.observe(&miss);
        sched.observe(&miss);
        assert_eq!(sched.pipeline_depth(), 1, "counter should have been reset");

        // One more miss pushes to 3 consecutive → depth increases.
        sched.observe(&miss);
        assert_eq!(sched.pipeline_depth(), 2);
    }

    #[test]
    fn pacing_only_unknown_feedback_does_not_raise_depth() {
        let config = SchedulerConfig::pacing_only();
        let mut sched = Scheduler::new(config);

        let unknown = PresentFeedback {
            submitted_at: HostTime(2_000),
            build_start: HostTime(1_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: None,
            pacing_overrun: None,
        };

        for _ in 0..8 {
            sched.observe(&unknown);
        }

        assert_eq!(sched.pipeline_depth(), 1);
        assert!(
            sched.safety_margin_ticks() > 0,
            "unknown deadline feedback should still train build-cost EMA"
        );
    }

    #[test]
    fn depth_clamped_at_max() {
        let mut config = SchedulerConfig::predictive();
        config.max_depth = 3;
        config.degradation_policy = DegradationPolicy::Adaptive {
            miss_threshold: 1,
            recovery_threshold: 10,
        };
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.pipeline_depth(), 1);

        let miss = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };

        // Many consecutive misses should not exceed max_depth.
        for _ in 0..20 {
            sched.observe(&miss);
        }
        assert_eq!(sched.pipeline_depth(), 3, "depth should be clamped at max");
    }

    #[test]
    fn build_cost_ema_updates() {
        let config = SchedulerConfig::predictive();
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.safety_margin_ticks(), 0);

        let feedback = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(false),
            pacing_overrun: None,
        };

        sched.observe(&feedback);
        assert!(
            sched.safety_margin_ticks() > 0,
            "safety margin should increase after observing build cost"
        );
    }

    #[test]
    fn pacing_overrun_raises_depth_more_conservatively() {
        let config = SchedulerConfig::pacing_only();
        let mut sched = Scheduler::new(config);

        let overrun = PresentFeedback {
            submitted_at: HostTime(2_000),
            build_start: HostTime(1_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: None,
            pacing_overrun: Some(true),
        };

        for _ in 0..5 {
            sched.observe(&overrun);
        }
        assert_eq!(sched.pipeline_depth(), 1);

        sched.observe(&overrun);
        assert_eq!(sched.pipeline_depth(), 2);
    }

    #[test]
    fn pacing_overrun_counters_reset_on_clear_tick() {
        let config = SchedulerConfig::pacing_only();
        let mut sched = Scheduler::new(config);

        let overrun = PresentFeedback {
            submitted_at: HostTime(2_000),
            build_start: HostTime(1_000),
            expected_present: None,
            actual_present: None,
            missed_deadline: None,
            pacing_overrun: Some(true),
        };
        let clear = PresentFeedback {
            pacing_overrun: Some(false),
            ..overrun
        };

        for _ in 0..3 {
            sched.observe(&overrun);
        }
        sched.observe(&clear);
        for _ in 0..5 {
            sched.observe(&overrun);
        }

        assert_eq!(sched.pipeline_depth(), 1);
        sched.observe(&overrun);
        assert_eq!(sched.pipeline_depth(), 2);
    }
}
