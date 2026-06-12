// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Affine clock for media synchronization.
//!
//! [`AffineClock`] maintains a smoothed affine mapping from host time to media
//! time:
//!
//! ```text
//! media_time = epoch_media + rate * (host_time - epoch_host)
//! ```
//!
//! Rate and epoch media are updated smoothly (via EMA) each time a new
//! observation is fed in, avoiding jitter while tracking drift.
//!
//! This module is a helper for hosts that need to map frameclock host times into
//! an external timeline, such as video PTS or an audio-master media clock. A
//! media layer feeds observations from the media backend, then queries with a
//! planned frame's sample or target-present time to choose content for the frame
//! being prepared.

use frameclock::HostTime;

/// Result returned by [`AffineClock::update_or_reanchor`].
///
/// Use this for diagnostics or media-sync policy that wants to distinguish
/// normal drift correction from seeks, loops, and other discontinuities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AffineClockUpdate {
    /// The observation was ignored because it contained non-finite media time or
    /// was not newer than the last accepted observation.
    Ignored,
    /// The observation initialized an unanchored clock.
    Initialized,
    /// The observation was applied through the normal smoothing path.
    Smoothed,
    /// The observation exceeded a discontinuity threshold and reanchored the
    /// clock instead of being treated as drift.
    Reanchored,
}

/// A smoothed affine mapping from host time to media time seconds.
///
/// Media or playback code creates this when it needs to choose external
/// timeline content for a frameclock-planned host time. Feed observations via
/// [`update`](Self::update) or [`update_or_reanchor`](Self::update_or_reanchor),
/// and query via [`media_time_at`](Self::media_time_at) using host times from
/// frameclock plans or platform media timing callbacks.
///
/// Ordinary UI/render hosts that only need frame pacing can ignore this type
/// and use `frameclock` directly.
#[derive(Clone, Debug)]
pub struct AffineClock {
    /// Current estimated rate (media seconds per host tick).
    rate: f64,
    /// Baseline rate restored by [`reset`](Self::reset).
    initial_rate: f64,
    /// Host tick used as the affine mapping epoch.
    epoch_host: u64,
    /// Media time at [`Self::epoch_host`].
    epoch_media: f64,
    /// EMA smoothing factor for rate correction (0.0–1.0).
    rate_alpha: f64,
    /// EMA smoothing factor for offset correction (0.0–1.0).
    offset_alpha: f64,
    /// Whether at least one observation has been fed.
    initialized: bool,
    /// Last host time observation (for rate estimation).
    last_host: u64,
    /// Last media time observation.
    last_media: f64,
}

impl AffineClock {
    /// Creates a new media-clock mapper.
    ///
    /// `initial_rate` is in media-seconds per host-tick (e.g. for nanosecond
    /// ticks, this would be `1e-9`). It is also the initial rate that
    /// [`reset`](Self::reset) restores, so the value chosen here sets the
    /// post-reset baseline unless [`set_rate`](Self::set_rate) replaces it.
    #[must_use]
    pub fn new(initial_rate: f64, rate_alpha: f64, offset_alpha: f64) -> Self {
        Self {
            rate: initial_rate,
            initial_rate,
            epoch_host: 0,
            epoch_media: 0.0,
            rate_alpha,
            offset_alpha,
            initialized: false,
            last_host: 0,
            last_media: 0.0,
        }
    }

    /// Queries the estimated media time at a host time.
    ///
    /// Returns `None` if no observations have been fed yet.
    #[must_use]
    pub fn media_time_at(&self, host: HostTime) -> Option<f64> {
        if !self.initialized {
            return None;
        }
        Some(self.media_time_at_initialized(host.ticks()))
    }

    /// Returns the current effective media-seconds-per-host-tick rate.
    ///
    /// This includes both commanded rate changes made through
    /// [`set_rate`](Self::set_rate) and rate drift learned from observations.
    #[must_use]
    pub const fn rate(&self) -> f64 {
        self.rate
    }

    /// Feeds a `(host_time, media_time_seconds)` observation to update the
    /// mapping.
    ///
    /// On the first call, this sets the mapping exactly. Subsequent calls
    /// smooth the rate and offset via EMA.
    ///
    /// A non-finite `media_time` (NaN or infinity) is ignored, leaving the
    /// mapping unchanged, so a single bad observation cannot poison the clock.
    pub fn update(&mut self, host: HostTime, media_time: f64) {
        _ = self.update_inner(host.ticks(), media_time);
    }

    /// Feeds an observation, snapping to it when the current mapping error
    /// exceeds `discontinuity_threshold`.
    ///
    /// Use this for media timelines where seeks, loops, and discontinuous PTS
    /// jumps are expected. A finite, non-negative threshold is interpreted in
    /// media seconds. Invalid thresholds disable snapping and use the normal
    /// smoothing path.
    #[must_use]
    pub fn update_or_reanchor(
        &mut self,
        host: HostTime,
        media_time: f64,
        discontinuity_threshold: f64,
    ) -> AffineClockUpdate {
        if !media_time.is_finite() {
            return AffineClockUpdate::Ignored;
        }

        if self.initialized && host.ticks() <= self.last_host {
            return AffineClockUpdate::Ignored;
        }

        if self.initialized && discontinuity_threshold.is_finite() && discontinuity_threshold >= 0.0
        {
            let predicted = self.media_time_at_initialized(host.ticks());
            if (media_time - predicted).abs() > discontinuity_threshold {
                self.reanchor(host, media_time);
                return AffineClockUpdate::Reanchored;
            }
        }

        self.update_inner(host.ticks(), media_time)
    }

    fn update_inner(&mut self, host_ticks: u64, media_time: f64) -> AffineClockUpdate {
        if !media_time.is_finite() {
            return AffineClockUpdate::Ignored;
        }

        if !self.initialized {
            self.reanchor(HostTime(host_ticks), media_time);
            return AffineClockUpdate::Initialized;
        }

        let dt_host = host_ticks.saturating_sub(self.last_host);
        if dt_host == 0 {
            return AffineClockUpdate::Ignored;
        }

        // Estimate instantaneous rate from this pair of observations.
        let dt_media = media_time - self.last_media;
        let observed_rate = dt_media / dt_host as f64;

        // Smooth rate.
        self.rate = self.rate_alpha * observed_rate + (1.0 - self.rate_alpha) * self.rate;

        // Correct the epoch media value from the current rate.
        let predicted_media = self.media_time_at_initialized(host_ticks);
        let offset_error = media_time - predicted_media;

        // Smooth offset correction.
        self.epoch_media += self.offset_alpha * offset_error;

        self.last_host = host_ticks;
        self.last_media = media_time;
        AffineClockUpdate::Smoothed
    }

    /// Reanchors the mapping exactly at `(host_time, media_time)`.
    ///
    /// This keeps the current rate but resets accumulated offset state. Use it
    /// for known timeline discontinuities such as seek, loop, and pause/resume
    /// points.
    pub fn reanchor(&mut self, host: HostTime, media_time: f64) {
        if !media_time.is_finite() {
            return;
        }

        self.epoch_host = host.ticks();
        self.epoch_media = media_time;
        self.last_host = host.ticks();
        self.last_media = media_time;
        self.initialized = true;
    }

    /// Sets the commanded host-to-media rate immediately.
    ///
    /// This is for known playback-rate changes, not clock drift. If the clock
    /// is initialized, the current mapping is preserved at the last observed
    /// host time before the rate changes so media time remains continuous. The
    /// new rate also becomes the baseline restored by [`reset`](Self::reset).
    pub fn set_rate(&mut self, rate: f64) {
        if !rate.is_finite() {
            return;
        }

        if self.initialized {
            let anchor_host = self.last_host;
            let anchor_media = self.media_time_at_initialized(anchor_host);
            self.rate = rate;
            self.initial_rate = rate;
            self.epoch_host = anchor_host;
            self.epoch_media = anchor_media;
            self.last_media = anchor_media;
        } else {
            self.rate = rate;
            self.initial_rate = rate;
        }
    }

    fn media_time_at_initialized(&self, host_ticks: u64) -> f64 {
        let host_delta = if host_ticks >= self.epoch_host {
            host_ticks.saturating_sub(self.epoch_host) as f64
        } else {
            -(self.epoch_host.saturating_sub(host_ticks) as f64)
        };
        self.epoch_media + self.rate * host_delta
    }

    /// Resets all accumulated state — including the rate, which is restored to
    /// the current baseline rate — requiring new observations before queries
    /// return values.
    pub fn reset(&mut self) {
        self.rate = self.initial_rate;
        self.epoch_host = 0;
        self.epoch_media = 0.0;
        self.initialized = false;
        self.last_host = 0;
        self.last_media = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn host(ticks: u64) -> HostTime {
        HostTime(ticks)
    }

    #[test]
    fn uninitialized_returns_none() {
        let clock = AffineClock::new(1e-9, 0.1, 0.1);
        assert!(clock.media_time_at(host(1_000_000_000)).is_none());
    }

    #[test]
    fn first_observation_sets_mapping_exactly() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        // 1 second of host ticks (at 1ns resolution) = 1.0s media time.
        clock.update(host(1_000_000_000), 1.0);

        let mt = clock.media_time_at(host(2_000_000_000)).unwrap();
        // rate * 2e9 + offset = 1e-9 * 2e9 + (1.0 - 1e-9 * 1e9) = 2.0 + 0.0 = 2.0
        assert!((mt - 2.0).abs() < 1e-6, "expected ~2.0, got {mt}");
    }

    #[test]
    fn large_absolute_host_times_keep_delta_precision() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        let host_epoch = 10_000_000_000_000_000_000_u64;
        clock.update(host(host_epoch), 10.0);

        let mt = clock.media_time_at(host(host_epoch + 1_000)).unwrap();

        assert!(
            (mt - 10.000_001).abs() < 1e-12,
            "expected microsecond delta at large epoch, got {mt}"
        );
    }

    #[test]
    fn rate_converges() {
        // Rate is initially 1e-9 (1ns ticks = seconds).
        // Feed observations at exactly that rate; rate should stay stable.
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(0), 0.0);
        for i in 1..=10 {
            let host_ticks = i * 1_000_000_000_u64;
            let media = i as f64;
            clock.update(host(host_ticks), media);
        }

        let mt = clock.media_time_at(host(11_000_000_000)).unwrap();
        assert!((mt - 11.0).abs() < 0.1, "expected ~11.0, got {mt}");
    }

    #[test]
    fn reset_clears_state() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(1_000_000_000), 1.0);
        assert!(clock.media_time_at(host(2_000_000_000)).is_some());

        clock.reset();
        assert!(clock.media_time_at(host(2_000_000_000)).is_none());
    }

    #[test]
    fn reset_restores_initial_rate() {
        let mut clock = AffineClock::new(1e-9, 0.5, 0.5);
        clock.update(host(0), 0.0);
        // Drive media at twice the initial rate so the EMA drifts toward 2e-9.
        for i in 1..=30_u64 {
            clock.update(host(i * 1_000_000_000), 2.0 * i as f64);
        }
        assert!(
            clock.rate > 1.5e-9,
            "precondition: rate should have drifted up"
        );

        clock.reset();

        assert_eq!(clock.rate, 1e-9, "reset must restore the initial rate");
    }

    #[test]
    fn reanchor_snaps_known_discontinuity() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(0), 0.0);
        clock.update(host(1_000_000_000), 1.0);

        clock.reanchor(host(2_000_000_000), 10.0);

        assert!((clock.media_time_at(host(2_000_000_000)).unwrap() - 10.0).abs() < 1e-12);
        assert!((clock.media_time_at(host(3_000_000_000)).unwrap() - 11.0).abs() < 1e-12);
    }

    #[test]
    fn update_or_reanchor_snaps_large_error() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        assert_eq!(
            clock.update_or_reanchor(host(0), 0.0, 0.25),
            AffineClockUpdate::Initialized
        );

        assert_eq!(
            clock.update_or_reanchor(host(1_000_000_000), 1.0, 0.25),
            AffineClockUpdate::Smoothed
        );
        assert_eq!(
            clock.update_or_reanchor(host(2_000_000_000), 10.0, 0.25),
            AffineClockUpdate::Reanchored
        );

        assert!((clock.media_time_at(host(2_000_000_000)).unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn set_rate_applies_commanded_rate_without_drift_learning() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(0), 0.0);
        clock.update(host(1_000_000_000), 1.0);

        clock.set_rate(2e-9);

        assert!((clock.media_time_at(host(1_000_000_000)).unwrap() - 1.0).abs() < 1e-12);
        assert!((clock.media_time_at(host(2_000_000_000)).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn ignores_non_finite_first_observation() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        // Garbage before any good observation must not initialize the clock.
        clock.update(host(1_000_000_000), f64::NAN);
        clock.update(host(1_000_000_000), f64::INFINITY);
        assert!(clock.media_time_at(host(2_000_000_000)).is_none());

        // A finite observation still initializes normally.
        clock.update(host(1_000_000_000), 1.0);
        assert!(clock.media_time_at(host(2_000_000_000)).is_some());
    }

    #[test]
    fn ignores_non_finite_observation_after_init() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(0), 0.0);
        clock.update(host(1_000_000_000), 1.0);
        let before = clock.media_time_at(host(2_000_000_000)).unwrap();

        // Garbage observations leave the mapping untouched.
        clock.update(host(2_000_000_000), f64::NAN);
        clock.update(host(3_000_000_000), f64::INFINITY);
        clock.update(host(4_000_000_000), f64::NEG_INFINITY);
        assert_eq!(clock.media_time_at(host(2_000_000_000)).unwrap(), before);

        // A subsequent good observation is still applied.
        clock.update(host(2_000_000_000), 2.0);
        assert!((clock.media_time_at(host(2_000_000_000)).unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ignores_out_of_order_observations() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(1_000_000_000), 1.0);
        clock.update(host(2_000_000_000), 2.0);
        let before = clock.media_time_at(host(3_000_000_000)).unwrap();

        assert_eq!(
            clock.update_or_reanchor(host(1_500_000_000), 99.0, 0.25),
            AffineClockUpdate::Ignored
        );

        assert_eq!(clock.media_time_at(host(3_000_000_000)).unwrap(), before);
    }

    #[test]
    fn ignores_same_host_observations_after_init() {
        let mut clock = AffineClock::new(1e-9, 0.1, 0.1);
        clock.update(host(1_000_000_000), 1.0);
        let before = clock.media_time_at(host(2_000_000_000)).unwrap();

        assert_eq!(
            clock.update_or_reanchor(host(1_000_000_000), 4.0, 0.25),
            AffineClockUpdate::Ignored
        );

        assert_eq!(clock.media_time_at(host(2_000_000_000)).unwrap(), before);
    }
}
