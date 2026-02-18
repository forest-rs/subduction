// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Internal frame-tick queueing primitives.
#![allow(
    dead_code,
    reason = "queue wiring is staged and consumed by later backend integration work"
)]

use crate::queue::BoundedQueue;
use subduction_core::timing::FrameTick;

/// Internal bounded queue for frame ticks.
///
/// Overflow policy is `drop_oldest` to retain the freshest pacing signal.
#[derive(Debug, Clone)]
pub(crate) struct TickQueue {
    inner: BoundedQueue<FrameTick>,
}

impl TickQueue {
    pub(crate) const DEFAULT_CAPACITY: usize = 8;

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: BoundedQueue::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, tick: FrameTick) {
        self.inner.push(tick);
    }

    pub(crate) fn pop(&mut self) -> Option<FrameTick> {
        self.inner.pop()
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub(crate) fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }
}

impl Default for TickQueue {
    fn default() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::TickQueue;
    use subduction_core::output::OutputId;
    use subduction_core::time::HostTime;
    use subduction_core::timing::{FrameTick, TimingConfidence};

    fn test_tick(frame_index: u64) -> FrameTick {
        FrameTick {
            now: HostTime(frame_index),
            predicted_present: None,
            refresh_interval: None,
            confidence: TimingConfidence::PacingOnly,
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    #[test]
    fn overflow_drops_oldest_tick() {
        let mut queue = TickQueue::with_capacity(2);
        queue.push(test_tick(1));
        queue.push(test_tick(2));
        queue.push(test_tick(3));

        assert_eq!(queue.pop().map(|tick| tick.frame_index), Some(2));
        assert_eq!(queue.pop().map(|tick| tick.frame_index), Some(3));
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn empty_state_tracks_push_and_pop() {
        let mut queue = TickQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.push(test_tick(7));
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let _ = queue.pop();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }
}
