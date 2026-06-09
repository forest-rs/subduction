// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame demand and demand-ordering policy.
//!
//! This module owns the policy that ranks why a frame is requested. It
//! explicitly does not own display timing, event-loop wakeups, timers, or
//! renderer submission.

use core::ops::{BitOr, BitOrAssign};

/// Dominant scheduling class for a [`FrameDemand`] set.
///
/// Variants are ordered from least urgent to most urgent. The scheduler and
/// higher-level frame drivers use this ordering to decide whether newly arrived
/// demand should preempt an already queued frame plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrameDemandClass {
    /// No frame is currently needed.
    #[default]
    None = 0,
    /// Deferrable visual work where power and batching matter more than
    /// immediate latency.
    Background = 1,
    /// Smooth visual work such as animation or media playback.
    Animation = 2,
    /// Continuous user input such as scrolling, resize, pointer movement, or
    /// gestures.
    ContinuousInput = 3,
    /// Latency-sensitive one-shot input such as key presses, clicks, or IME.
    Input = 4,
}

/// Why the host is requesting frame work.
///
/// Demand is a compact bit set because several causes can be pending at once.
/// The scheduler derives a policy from the strongest pending demand: input is
/// latency-first, continuous input is latency-sensitive but allowed to choose a
/// sustainable cadence, animation prefers even pacing, and background work can
/// be deferred.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FrameDemand(u8);

impl FrameDemand {
    /// No frame is currently needed.
    ///
    /// Hosts should normally avoid calling
    /// [`Scheduler::plan()`](crate::scheduler::Scheduler::plan) when demand is
    /// empty. Passing `NONE` is reserved for code that intentionally wants a
    /// passive pacing plan for diagnostics or backend bookkeeping.
    pub const NONE: Self = Self(0);
    /// Smooth visual work such as animation or media playback.
    pub const ANIMATION: Self = Self(1 << 0);
    /// Latency-sensitive one-shot input such as key presses, clicks, or IME.
    pub const INPUT: Self = Self(1 << 1);
    /// Continuous user input such as scrolling, resize, pointer movement, or
    /// gestures.
    pub const CONTINUOUS_INPUT: Self = Self(1 << 2);
    /// Deferrable visual work where power and batching matter more than
    /// immediate latency.
    pub const BACKGROUND: Self = Self(1 << 3);

    /// Returns an empty demand set.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self::NONE
    }

    /// Creates a demand set from raw bits, discarding unknown bits.
    #[inline]
    #[must_use]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & 0x0f)
    }

    /// Returns the raw demand bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns whether no demand bits are set.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether all bits in `other` are set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns the strongest scheduling class present in this demand set.
    #[inline]
    #[must_use]
    pub const fn dominant_class(self) -> FrameDemandClass {
        if self.contains(Self::INPUT) {
            FrameDemandClass::Input
        } else if self.contains(Self::CONTINUOUS_INPUT) {
            FrameDemandClass::ContinuousInput
        } else if self.contains(Self::ANIMATION) {
            FrameDemandClass::Animation
        } else if self.contains(Self::BACKGROUND) {
            FrameDemandClass::Background
        } else {
            FrameDemandClass::None
        }
    }

    /// Returns whether this demand is strong enough to replace `planned`.
    ///
    /// This uses the same ordering as the scheduler. Hosts and adapters should
    /// use this instead of duplicating local demand-priority tables.
    #[inline]
    #[must_use]
    pub fn preempts(self, planned: Self) -> bool {
        self.dominant_class() > planned.dominant_class()
    }

    /// Adds demand bits.
    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl BitOr for FrameDemand {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for FrameDemand {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_class_uses_strongest_demand() {
        assert_eq!(FrameDemand::NONE.dominant_class(), FrameDemandClass::None);
        assert_eq!(
            FrameDemand::BACKGROUND.dominant_class(),
            FrameDemandClass::Background
        );
        assert_eq!(
            (FrameDemand::BACKGROUND | FrameDemand::ANIMATION).dominant_class(),
            FrameDemandClass::Animation
        );
        assert_eq!(
            (FrameDemand::ANIMATION | FrameDemand::CONTINUOUS_INPUT).dominant_class(),
            FrameDemandClass::ContinuousInput
        );
        assert_eq!(
            (FrameDemand::INPUT | FrameDemand::CONTINUOUS_INPUT | FrameDemand::ANIMATION)
                .dominant_class(),
            FrameDemandClass::Input
        );
    }

    #[test]
    fn preemption_follows_demand_order() {
        assert!(FrameDemandClass::Input > FrameDemandClass::ContinuousInput);
        assert!(FrameDemandClass::ContinuousInput > FrameDemandClass::Animation);
        assert!(FrameDemandClass::Animation > FrameDemandClass::Background);
        assert!(FrameDemandClass::Background > FrameDemandClass::None);

        assert!(FrameDemand::INPUT.preempts(FrameDemand::ANIMATION));
        assert!(FrameDemand::CONTINUOUS_INPUT.preempts(FrameDemand::ANIMATION));
        assert!(FrameDemand::ANIMATION.preempts(FrameDemand::BACKGROUND));
        assert!(!FrameDemand::ANIMATION.preempts(FrameDemand::CONTINUOUS_INPUT));
        assert!(!FrameDemand::BACKGROUND.preempts(FrameDemand::ANIMATION));
        assert!(!FrameDemand::NONE.preempts(FrameDemand::BACKGROUND));
    }
}
