// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stateful media timeline wrapper.

use crate::{AffineClock, AffineClockUpdate};
use frameclock::HostTime;

/// Host-facing media timeline built on [`AffineClock`].
///
/// `MediaTimeline` owns playback-rate and discontinuity policy for one external
/// media timeline. Feed observations from the media backend with
/// [`observe`](Self::observe), call [`reanchor`](Self::reanchor) for known
/// jumps such as seeks or loops, call [`set_paused`](Self::set_paused) when
/// playback pauses or resumes, and query [`media_time_at`](Self::media_time_at)
/// with a frameclock-planned host time.
///
/// This type performs only pure timeline math. It does not decide whether to
/// drop, duplicate, decode, seek, or present media frames; hosts apply those
/// policies using the returned media time.
#[derive(Clone, Debug)]
pub struct MediaTimeline {
    clock: AffineClock,
    seconds_per_host_tick: f64,
    playback_rate: f64,
    discontinuity_threshold: f64,
    paused: bool,
    paused_media_time: f64,
}

impl MediaTimeline {
    /// Default smoothing factor used for rate correction.
    pub const DEFAULT_RATE_ALPHA: f64 = 0.1;
    /// Default smoothing factor used for offset correction.
    pub const DEFAULT_OFFSET_ALPHA: f64 = 0.1;
    /// Default media-time error, in seconds, that is treated as a discontinuity.
    pub const DEFAULT_DISCONTINUITY_THRESHOLD: f64 = 0.25;

    /// Creates a media timeline for the given host-time scale.
    ///
    /// `seconds_per_host_tick` is the conversion from one frameclock host tick
    /// to seconds at normal `1.0x` playback. For nanosecond host ticks, pass
    /// `1e-9`.
    #[must_use]
    pub fn new(seconds_per_host_tick: f64) -> Self {
        Self::with_smoothing(
            seconds_per_host_tick,
            Self::DEFAULT_RATE_ALPHA,
            Self::DEFAULT_OFFSET_ALPHA,
            Self::DEFAULT_DISCONTINUITY_THRESHOLD,
        )
    }

    /// Creates a media timeline with explicit smoothing and discontinuity
    /// settings.
    #[must_use]
    pub fn with_smoothing(
        seconds_per_host_tick: f64,
        rate_alpha: f64,
        offset_alpha: f64,
        discontinuity_threshold: f64,
    ) -> Self {
        Self {
            clock: AffineClock::new(seconds_per_host_tick, rate_alpha, offset_alpha),
            seconds_per_host_tick,
            playback_rate: 1.0,
            discontinuity_threshold,
            paused: false,
            paused_media_time: 0.0,
        }
    }

    /// Returns the current commanded playback rate.
    ///
    /// This is the host-requested playback rate. The effective affine mapping
    /// rate can drift from this value as observations correct clock skew; use
    /// [`effective_rate`](Self::effective_rate) to inspect the learned mapping.
    #[must_use]
    pub const fn playback_rate(&self) -> f64 {
        self.playback_rate
    }

    /// Returns the current effective media-seconds-per-host-tick rate.
    ///
    /// This includes the commanded playback rate and any drift learned from
    /// observations.
    #[must_use]
    pub const fn effective_rate(&self) -> f64 {
        self.clock.rate()
    }

    /// Sets the commanded playback rate.
    ///
    /// This changes the clock rate immediately and is intended for known
    /// playback-rate commands, not drift learning. Non-finite rates are ignored.
    pub fn set_playback_rate(&mut self, playback_rate: f64) {
        if !playback_rate.is_finite() {
            return;
        }
        self.playback_rate = playback_rate;
        self.clock
            .set_rate(self.seconds_per_host_tick * playback_rate);
    }

    /// Returns whether the timeline is currently paused.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// Sets whether the media timeline is paused.
    ///
    /// This reanchors at the supplied observation so pause/resume boundaries do
    /// not train the drift estimator. While paused, [`observe`](Self::observe)
    /// also reanchors instead of smoothing, and
    /// [`media_time_at`](Self::media_time_at) returns the latest paused media
    /// time.
    pub fn set_paused(&mut self, paused: bool, host: HostTime, media_time: f64) {
        if !media_time.is_finite() {
            return;
        }

        self.paused = paused;
        self.paused_media_time = media_time;
        self.clock.reanchor(host, media_time);
    }

    /// Returns the current discontinuity threshold in media seconds.
    #[must_use]
    pub const fn discontinuity_threshold(&self) -> f64 {
        self.discontinuity_threshold
    }

    /// Sets the media-time error threshold that causes future observations to
    /// reanchor instead of smooth.
    ///
    /// Invalid thresholds are accepted by storage but disable snapping in the
    /// underlying affine clock, matching [`AffineClock::update_or_reanchor`].
    pub fn set_discontinuity_threshold(&mut self, threshold: f64) {
        self.discontinuity_threshold = threshold;
    }

    /// Feeds one media-time observation.
    ///
    /// The observation is smoothed unless its error exceeds the discontinuity
    /// threshold, in which case the timeline reanchors at the observation. When
    /// the timeline is paused, observations reanchor instead of smoothing so a
    /// frozen media clock cannot train the learned rate toward zero.
    pub fn observe(&mut self, host: HostTime, media_time: f64) -> AffineClockUpdate {
        if self.paused {
            if !media_time.is_finite() {
                return AffineClockUpdate::Ignored;
            }
            self.paused_media_time = media_time;
            self.clock.reanchor(host, media_time);
            return AffineClockUpdate::Reanchored;
        }

        self.clock
            .update_or_reanchor(host, media_time, self.discontinuity_threshold)
    }

    /// Reanchors the timeline exactly at `(host, media_time)`.
    ///
    /// Use this when the host knows the media timeline jumped, such as after a
    /// seek, loop, pause/resume boundary, or decoder reset.
    pub fn reanchor(&mut self, host: HostTime, media_time: f64) {
        if self.paused && media_time.is_finite() {
            self.paused_media_time = media_time;
        }
        self.clock.reanchor(host, media_time);
    }

    /// Returns the estimated media time at `host`.
    #[must_use]
    pub fn media_time_at(&self, host: HostTime) -> Option<f64> {
        if self.paused {
            return Some(self.paused_media_time);
        }
        self.clock.media_time_at(host)
    }

    /// Returns the underlying affine clock.
    #[must_use]
    pub const fn clock(&self) -> &AffineClock {
        &self.clock
    }

    /// Resets accumulated observations while preserving the current commanded
    /// playback rate.
    pub fn reset(&mut self) {
        self.clock.reset();
        self.paused = false;
        self.paused_media_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameclock::HostTime;

    const fn host(ticks: u64) -> HostTime {
        HostTime(ticks)
    }

    #[test]
    fn timeline_observes_and_queries_media_time() {
        let mut timeline = MediaTimeline::new(1e-9);

        assert_eq!(
            timeline.observe(host(1_000_000_000), 1.0),
            AffineClockUpdate::Initialized
        );

        let media = timeline.media_time_at(host(2_000_000_000)).unwrap();
        assert!((media - 2.0).abs() < 1e-6, "expected ~2.0, got {media}");
    }

    #[test]
    fn timeline_reanchors_large_discontinuity() {
        let mut timeline = MediaTimeline::with_smoothing(1e-9, 0.1, 0.1, 0.25);

        assert_eq!(
            timeline.observe(host(0), 0.0),
            AffineClockUpdate::Initialized
        );
        assert_eq!(
            timeline.observe(host(1_000_000_000), 1.0),
            AffineClockUpdate::Smoothed
        );
        assert_eq!(
            timeline.observe(host(2_000_000_000), 10.0),
            AffineClockUpdate::Reanchored
        );
        assert!((timeline.media_time_at(host(2_000_000_000)).unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn playback_rate_changes_future_mapping_without_jump() {
        let mut timeline = MediaTimeline::new(1e-9);
        timeline.observe(host(0), 0.0);
        timeline.observe(host(1_000_000_000), 1.0);

        timeline.set_playback_rate(2.0);

        assert_eq!(timeline.playback_rate(), 2.0);
        assert_eq!(timeline.effective_rate(), 2e-9);
        assert!((timeline.media_time_at(host(1_000_000_000)).unwrap() - 1.0).abs() < 1e-12);
        assert!((timeline.media_time_at(host(2_000_000_000)).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn reset_preserves_commanded_playback_rate() {
        let mut timeline = MediaTimeline::new(1e-9);
        timeline.set_playback_rate(0.5);
        timeline.observe(host(0), 4.0);
        assert!(timeline.media_time_at(host(1_000_000_000)).is_some());

        timeline.reset();
        assert_eq!(timeline.playback_rate(), 0.5);
        assert!(timeline.media_time_at(host(1_000_000_000)).is_none());

        timeline.observe(host(10_000_000_000), 10.0);
        assert!((timeline.media_time_at(host(11_000_000_000)).unwrap() - 10.5).abs() < 1e-12);
    }

    #[test]
    fn paused_observations_do_not_train_rate_toward_zero() {
        let mut timeline = MediaTimeline::new(1e-9);
        timeline.observe(host(0), 0.0);
        timeline.observe(host(1_000_000_000), 1.0);

        timeline.set_paused(true, host(2_000_000_000), 2.0);
        assert!(timeline.is_paused());
        assert_eq!(timeline.media_time_at(host(10_000_000_000)), Some(2.0));

        for i in 3..=8 {
            assert_eq!(
                timeline.observe(host(i * 1_000_000_000), 2.0),
                AffineClockUpdate::Reanchored
            );
        }

        assert_eq!(timeline.effective_rate(), 1e-9);
        timeline.set_paused(false, host(9_000_000_000), 2.0);
        assert!(!timeline.is_paused());
        assert!((timeline.media_time_at(host(10_000_000_000)).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn paused_non_finite_observation_is_ignored() {
        let mut timeline = MediaTimeline::new(1e-9);
        timeline.observe(host(0), 0.0);
        timeline.set_paused(true, host(1_000_000_000), 1.0);

        assert_eq!(
            timeline.observe(host(2_000_000_000), f64::NAN),
            AffineClockUpdate::Ignored
        );
        assert_eq!(timeline.media_time_at(host(3_000_000_000)), Some(1.0));
    }
}
