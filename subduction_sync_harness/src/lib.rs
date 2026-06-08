// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reusable sync metrics and grading for demo harnesses.

#![no_std]

extern crate alloc;

use alloc::string::String;
use frameclock::TimingConfidence;

/// Runtime pathology toggles for stress tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PathologyToggles {
    /// Simulated decode jitter is enabled.
    pub decode_jitter: bool,
    /// Simulated GPU/render stall is enabled.
    pub gpu_stall: bool,
    /// Simulated timer jitter is enabled.
    pub timer_jitter: bool,
    /// Emulated refresh-rate switching is enabled.
    pub vary_refresh: bool,
}

/// Per-frame metrics sample fed into [`SyncTracker::observe`].
#[derive(Clone, Copy, Debug)]
pub struct SyncSample {
    /// Timing capability for this frame.
    pub confidence: TimingConfidence,
    /// Signed phase error between media and overlay timelines, in ms.
    pub phase_error_ms: f64,
    /// Miss determined from explicit present deadline semantics.
    pub hard_miss: bool,
    /// Miss determined from fallback/pacing heuristics.
    pub soft_miss: bool,
    /// Frame delta in milliseconds.
    pub frame_delta_ms: f64,
}

/// Letter grade for synchronization quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncGrade {
    /// Tight sync and low miss rate.
    A,
    /// Good sync with moderate misses.
    B,
    /// Degraded but usable.
    C,
    /// Poor sync.
    D,
}

impl SyncGrade {
    /// Returns a short label for HUD rendering.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
        }
    }
}

/// Aggregated report returned by [`SyncTracker::observe`].
#[derive(Clone, Copy, Debug)]
pub struct SyncReport {
    /// Current grade.
    pub grade: SyncGrade,
    /// Misses per 1000 observed frames.
    pub miss_rate_per_1000: f64,
    /// Current frame's signed phase error in milliseconds.
    pub phase_error_ms: f64,
    /// Total frames observed.
    pub total_frames: u64,
    /// Total misses observed.
    pub missed_frames: u64,
}

/// Rolling sync tracker with fixed-size frame-delta history.
#[derive(Debug)]
pub struct SyncTracker<const N: usize> {
    deltas_ms: [f64; N],
    cursor: usize,
    total_frames: u64,
    missed_frames: u64,
}

impl<const N: usize> Default for SyncTracker<N> {
    fn default() -> Self {
        Self::new(16.67)
    }
}

impl<const N: usize> SyncTracker<N> {
    /// Creates a tracker with `seed_delta_ms` prefilled in the ring buffer.
    #[must_use]
    pub const fn new(seed_delta_ms: f64) -> Self {
        Self {
            deltas_ms: [seed_delta_ms; N],
            cursor: 0,
            total_frames: 0,
            missed_frames: 0,
        }
    }

    /// Observes one frame and returns an updated report.
    #[must_use]
    pub fn observe(&mut self, sample: SyncSample) -> SyncReport {
        self.total_frames = self.total_frames.saturating_add(1);
        self.deltas_ms[self.cursor % N] = sample.frame_delta_ms;
        self.cursor = (self.cursor + 1) % N;

        if sample.hard_miss || sample.soft_miss {
            self.missed_frames = self.missed_frames.saturating_add(1);
        }

        let miss_rate = if self.total_frames == 0 {
            0.0
        } else {
            self.missed_frames as f64 * 1000.0 / self.total_frames as f64
        };

        let grade = grade_for(sample.confidence, sample.phase_error_ms.abs(), miss_rate);

        SyncReport {
            grade,
            miss_rate_per_1000: miss_rate,
            phase_error_ms: sample.phase_error_ms,
            total_frames: self.total_frames,
            missed_frames: self.missed_frames,
        }
    }

    /// Returns ring-buffer frame deltas oldest→newest.
    #[must_use]
    pub fn frame_deltas(&self) -> [f64; N] {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            let idx = (self.cursor + i) % N;
            out[i] = self.deltas_ms[idx];
            i += 1;
        }
        out
    }

    /// Returns an ASCII sparkline over `frame_deltas()`.
    #[must_use]
    pub fn sparkline_ascii(&self, min_ms: f64, max_ms: f64) -> String {
        const LEVELS: &[u8] = b" .:-=+*#%@";
        let mut out = String::with_capacity(N);
        let mut i = 0;
        while i < N {
            let idx = (self.cursor + i) % N;
            let v = self.deltas_ms[idx].clamp(min_ms, max_ms);
            let t = (v - min_ms) / (max_ms - min_ms);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "index is clamped to ASCII level count"
            )]
            let level = (t * (LEVELS.len() as f64 - 1.0) + 0.5) as usize;
            out.push(LEVELS[level] as char);
            i += 1;
        }
        out
    }
}

fn grade_for(
    conf: TimingConfidence,
    phase_error_abs_ms: f64,
    miss_rate_per_1000: f64,
) -> SyncGrade {
    let (a_phase, b_phase, c_phase, a_miss, b_miss, c_miss) = match conf {
        TimingConfidence::Predictive => (16.0, 32.0, 50.0, 1.0, 5.0, 15.0),
        TimingConfidence::Estimated => (24.0, 45.0, 70.0, 3.0, 10.0, 25.0),
        TimingConfidence::PacingOnly => (35.0, 65.0, 100.0, 10.0, 30.0, 80.0),
    };

    if phase_error_abs_ms < a_phase && miss_rate_per_1000 < a_miss {
        SyncGrade::A
    } else if phase_error_abs_ms < b_phase && miss_rate_per_1000 < b_miss {
        SyncGrade::B
    } else if phase_error_abs_ms < c_phase && miss_rate_per_1000 < c_miss {
        SyncGrade::C
    } else {
        SyncGrade::D
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_rate_accumulates() {
        let mut t = SyncTracker::<8>::new(16.67);
        let mut i = 0;
        while i < 10 {
            let report = t.observe(SyncSample {
                confidence: TimingConfidence::PacingOnly,
                phase_error_ms: 10.0,
                hard_miss: i < 2,
                soft_miss: false,
                frame_delta_ms: 16.7,
            });
            if i == 9 {
                assert!((report.miss_rate_per_1000 - 200.0).abs() < 1e-6);
            }
            i += 1;
        }
    }

    #[test]
    fn predictive_thresholds_are_stricter() {
        let mut t = SyncTracker::<4>::new(16.67);
        let p = t.observe(SyncSample {
            confidence: TimingConfidence::Predictive,
            phase_error_ms: 40.0,
            hard_miss: false,
            soft_miss: false,
            frame_delta_ms: 16.7,
        });
        assert_eq!(p.grade, SyncGrade::C);

        let e = t.observe(SyncSample {
            confidence: TimingConfidence::Estimated,
            phase_error_ms: 40.0,
            hard_miss: false,
            soft_miss: false,
            frame_delta_ms: 16.7,
        });
        assert_eq!(e.grade, SyncGrade::B);
    }
}
