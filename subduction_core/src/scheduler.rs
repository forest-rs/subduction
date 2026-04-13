// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame scheduling with adaptive pipeline depth and safety margins.
//!
//! The [`Scheduler`] converts a [`FrameTick`] and [`PresentHints`] into a
//! [`FramePlan`], adapting its pipeline depth and safety margin based on
//! observed [`PresentFeedback`]. See the [`Scheduler`] struct docs for
//! details on pipeline depth and adaptive behavior.

use crate::time::Duration;
use crate::timing::{FramePlan, FrameTick, PresentFeedback, PresentHints, TimingConfidence};

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
    /// Initial pipeline depth (1–3).
    pub initial_depth: u8,
    /// Minimum pipeline depth.
    pub min_depth: u8,
    /// Maximum pipeline depth.
    pub max_depth: u8,
    /// EMA smoothing factor for build cost estimation (0.0–1.0).
    /// Smaller values = more smoothing.
    pub ema_alpha: f32,
    /// Safety margin multiplier applied to the EMA build cost.
    pub safety_multiplier: f32,
    /// Nominal latency (in ticks) used for pacing-only mode when no predicted
    /// present time is available.
    pub nominal_latency: Duration,
    /// Policy for adapting pipeline depth.
    pub degradation_policy: DegradationPolicy,
}

impl SchedulerConfig {
    /// Default configuration for macOS (predictive timing).
    #[must_use]
    pub const fn macos() -> Self {
        Self {
            initial_depth: 2,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.2,
            safety_multiplier: 1.5,
            nominal_latency: Duration(0),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }

    /// Default configuration for Windows (estimated timing via DWM/DXGI).
    ///
    /// Similar to macOS but with a wider safety multiplier to account for
    /// less predictable present timing from `DwmFlush`/DXGI frame stats.
    #[must_use]
    pub const fn windows() -> Self {
        Self {
            initial_depth: 2,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.2,
            safety_multiplier: 2.0,
            nominal_latency: Duration(0),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }

    /// Default configuration for Web (pacing-only timing).
    #[must_use]
    pub const fn web() -> Self {
        Self {
            initial_depth: 2,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.15,
            safety_multiplier: 2.0,
            // ~16ms at 1ns tick resolution.
            nominal_latency: Duration(16_000_000),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }

    /// Default configuration for Wayland (pacing-only timing via frame
    /// callbacks, with optional `wp_presentation` feedback).
    #[must_use]
    pub const fn wayland() -> Self {
        Self {
            initial_depth: 2,
            min_depth: 1,
            max_depth: 3,
            ema_alpha: 0.15,
            safety_multiplier: 2.0,
            // ~16ms at 1ns tick resolution.
            nominal_latency: Duration(16_000_000),
            degradation_policy: DegradationPolicy::Adaptive {
                miss_threshold: 3,
                recovery_threshold: 10,
            },
        }
    }
}

/// Exponential moving average tracker.
#[derive(Clone, Copy, Debug)]
struct Ema {
    value: f32,
    alpha: f32,
    initialized: bool,
}

impl Ema {
    const fn new(alpha: f32) -> Self {
        Self {
            value: 0.0,
            alpha,
            initialized: false,
        }
    }

    fn update(&mut self, sample: f32) {
        if self.initialized {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        } else {
            self.value = sample;
            self.initialized = true;
        }
    }

    const fn get(&self) -> f32 {
        self.value
    }
}

/// Frame scheduler that converts [`FrameTick`]s into [`FramePlan`]s and
/// adapts over time.
///
/// # Pipeline depth
///
/// Pipeline depth controls how many frames ahead the engine works, bounded
/// by [`SchedulerConfig::min_depth`] and [`SchedulerConfig::max_depth`].
/// A depth of 1 means the engine builds and submits each frame just before
/// its deadline (lowest latency, highest risk of missing deadlines). A depth
/// of 3 means the engine may be building frames two intervals ahead
/// (higher latency, but more time budget to absorb spikes).
///
/// # Adaptive behavior
///
/// With [`DegradationPolicy::Adaptive`] (set via
/// [`SchedulerConfig::degradation_policy`]), the scheduler increases depth
/// after consecutive deadline misses (trading latency for safety) and
/// decreases depth after sustained hits (reclaiming latency). The
/// EMA-smoothed build cost feeds into a safety margin (build cost ×
/// multiplier) that backends can query via
/// [`Scheduler::safety_margin_ticks()`] to decide how early to begin frame
/// work.
///
/// # Usage
///
/// ```rust,ignore
/// let plan = scheduler.plan(&tick, &hints);
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
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            pipeline_depth: config.initial_depth,
            build_cost_ema: Ema::new(config.ema_alpha),
            safety_margin_ticks: 0,
            consecutive_misses: 0,
            consecutive_hits: 0,
            config,
        }
    }

    /// Produces a [`FramePlan`] for the given tick and hints.
    #[must_use]
    pub fn plan(&mut self, tick: &FrameTick, hints: &PresentHints) -> FramePlan {
        let target_present = hints.desired_present.or(tick.predicted_present);

        let (present_time, semantic_time) = match tick.confidence {
            TimingConfidence::Predictive | TimingConfidence::Estimated => {
                // semantic_time = present_time for tight sync.
                let ts = target_present.unwrap_or(tick.now);
                (target_present, ts)
            }
            TimingConfidence::PacingOnly => {
                // No reliable present time; use nominal latency for semantic time.
                let ts = tick
                    .now
                    .checked_add(self.config.nominal_latency)
                    .unwrap_or(tick.now);
                (None, ts)
            }
        };

        FramePlan {
            semantic_time,
            present_time,
            commit_deadline: hints.latest_commit,
            pipeline_depth: self.pipeline_depth,
            output: tick.output,
            frame_index: tick.frame_index,
        }
    }

    /// Feeds presentation feedback to adapt scheduling parameters.
    pub fn observe(&mut self, feedback: &PresentFeedback) {
        // Update build cost EMA.
        let build_ticks = feedback
            .submitted_at
            .saturating_duration_since(feedback.build_start)
            .ticks();
        self.build_cost_ema.update(build_ticks as f32);

        // Update safety margin.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "EMA-smoothed build cost in ticks fits in u64"
        )]
        {
            self.safety_margin_ticks =
                (self.build_cost_ema.get() * self.config.safety_multiplier) as u64;
        }

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
}

#[cfg(test)]
mod tests {
    use crate::output::OutputId;
    use crate::time::HostTime;

    use super::*;

    fn make_tick(confidence: TimingConfidence, now: u64, predicted: Option<u64>) -> FrameTick {
        FrameTick {
            now: HostTime(now),
            predicted_present: predicted.map(HostTime),
            refresh_interval: Some(16_666_667),
            confidence,
            frame_index: 0,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    fn make_hints(deadline: u64) -> PresentHints {
        PresentHints {
            desired_present: None,
            latest_commit: HostTime(deadline),
        }
    }

    #[test]
    fn predictive_plan_uses_predicted_present() {
        let config = SchedulerConfig::macos();
        let mut sched = Scheduler::new(config);

        let tick = make_tick(TimingConfidence::Predictive, 1000, Some(2000));
        let hints = make_hints(1800);

        let plan = sched.plan(&tick, &hints);

        assert_eq!(plan.present_time, Some(HostTime(2000)));
        assert_eq!(plan.semantic_time, HostTime(2000));
        assert_eq!(plan.commit_deadline, HostTime(1800));
    }

    #[test]
    fn pacing_only_plan_has_no_present_time() {
        let config = SchedulerConfig::web();
        let mut sched = Scheduler::new(config);

        let tick = make_tick(TimingConfidence::PacingOnly, 1_000_000, None);
        let hints = make_hints(17_000_000);

        let plan = sched.plan(&tick, &hints);

        assert_eq!(plan.present_time, None);
        // semantic_time = now + nominal_latency (16ms).
        assert_eq!(plan.semantic_time, HostTime(17_000_000));
    }

    #[test]
    fn pipeline_depth_increases_after_misses() {
        let config = SchedulerConfig::macos();
        let mut sched = Scheduler::new(config);
        assert_eq!(sched.pipeline_depth(), 2);

        let feedback = PresentFeedback {
            submitted_at: HostTime(2000),
            build_start: HostTime(1000),
            expected_present: None,
            actual_present: None,
            missed_deadline: Some(true),
            pacing_overrun: None,
        };

        sched.observe(&feedback);
        assert_eq!(sched.pipeline_depth(), 2); // 1 miss
        sched.observe(&feedback);
        assert_eq!(sched.pipeline_depth(), 2); // 2 misses
        sched.observe(&feedback);
        assert_eq!(sched.pipeline_depth(), 3); // 3 misses → increase
    }

    #[test]
    fn consecutive_miss_counter_resets_on_success() {
        let config = SchedulerConfig::macos();
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
        assert_eq!(sched.pipeline_depth(), 2);
    }

    #[test]
    fn pipeline_depth_decreases_after_sustained_hits() {
        let mut config = SchedulerConfig::macos();
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
        let mut config = SchedulerConfig::macos();
        config.degradation_policy = DegradationPolicy::Fixed;
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
        let mut config = SchedulerConfig::macos();
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
        assert_eq!(sched.pipeline_depth(), 2); // 4 misses, threshold is 5
        sched.observe(&miss); // 5th miss → increase
        assert_eq!(sched.pipeline_depth(), 3);
    }

    #[test]
    fn recovery_respects_min_depth() {
        let mut config = SchedulerConfig::macos();
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
    fn estimated_confidence_uses_predicted_present() {
        let config = SchedulerConfig::macos();
        let mut sched = Scheduler::new(config);

        let tick = make_tick(TimingConfidence::Estimated, 1000, Some(2000));
        let hints = make_hints(1800);

        let plan = sched.plan(&tick, &hints);

        // Estimated behaves like Predictive: uses predicted_present.
        assert_eq!(plan.present_time, Some(HostTime(2000)));
        assert_eq!(plan.semantic_time, HostTime(2000));
    }

    #[test]
    fn observe_with_unknown_deadline_resets_counters() {
        let config = SchedulerConfig::macos();
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
        assert_eq!(sched.pipeline_depth(), 2, "counter should have been reset");

        // One more miss pushes to 3 consecutive → depth increases.
        sched.observe(&miss);
        assert_eq!(sched.pipeline_depth(), 3);
    }

    #[test]
    fn pacing_only_unknown_feedback_does_not_raise_depth() {
        let config = SchedulerConfig::web();
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

        assert_eq!(sched.pipeline_depth(), 2);
        assert!(
            sched.safety_margin_ticks() > 0,
            "unknown deadline feedback should still train build-cost EMA"
        );
    }

    #[test]
    fn depth_clamped_at_max() {
        let mut config = SchedulerConfig::macos();
        config.max_depth = 3;
        config.degradation_policy = DegradationPolicy::Adaptive {
            miss_threshold: 1,
            recovery_threshold: 10,
        };
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

        // Many consecutive misses should not exceed max_depth.
        for _ in 0..20 {
            sched.observe(&miss);
        }
        assert_eq!(sched.pipeline_depth(), 3, "depth should be clamped at max");
    }

    #[test]
    fn build_cost_ema_updates() {
        let config = SchedulerConfig::macos();
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
        let config = SchedulerConfig::web();
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
        assert_eq!(sched.pipeline_depth(), 2);

        sched.observe(&overrun);
        assert_eq!(sched.pipeline_depth(), 3);
    }

    #[test]
    fn pacing_overrun_counters_reset_on_clear_tick() {
        let config = SchedulerConfig::web();
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

        assert_eq!(sched.pipeline_depth(), 2);
        sched.observe(&overrun);
        assert_eq!(sched.pipeline_depth(), 3);
    }
}
