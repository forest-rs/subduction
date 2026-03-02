// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Presentation feedback contracts and queueing.

use crate::queue::BoundedQueue;
use subduction_core::output::OutputId;
use subduction_core::time::HostTime;

/// Unique identity for one `wl_surface.commit` submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionId(pub u64);

/// Per-feedback-object accumulation state while the feedback is in flight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingFeedback {
    pub(crate) sync_output: Option<OutputId>,
}

/// Converts a `wp_presentation_feedback.presented` timestamp to [`HostTime`].
///
/// `tv_nsec` is clamped to `≤999_999_999` to prevent overflow. Arithmetic is
/// saturating so edge-case compositor data never causes a panic.
pub(crate) fn presentation_time_to_host_time(
    tv_sec_hi: u32,
    tv_sec_lo: u32,
    tv_nsec: u32,
) -> HostTime {
    let seconds = u64::from(tv_sec_hi) << 32 | u64::from(tv_sec_lo);
    let nanos = seconds
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::from(tv_nsec.min(999_999_999)));
    HostTime(nanos)
}

/// Per-commit presentation feedback event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PresentEvent {
    /// The submission was presented.
    Presented {
        /// Identity of the commit this event corresponds to.
        id: SubmissionId,
        /// Actual presentation timestamp in backend `HostTime` ticks.
        actual_present: HostTime,
        /// Observed refresh interval in host ticks, if known.
        refresh_interval: Option<u64>,
        /// Output where the frame was shown, if known.
        output: Option<OutputId>,
        /// Raw protocol flags.
        flags: u32,
    },
    /// The compositor discarded the submission.
    Discarded {
        /// Identity of the commit this event corresponds to.
        id: SubmissionId,
    },
}

/// Bounded FIFO queue for [`PresentEvent`] values.
///
/// Overflow policy is `drop_oldest`: when full, pushing a new event removes
/// the oldest queued event first. This keeps newest feedback available to the
/// host under backpressure.
#[derive(Debug, Clone)]
pub struct PresentEventQueue {
    inner: BoundedQueue<PresentEvent>,
}

impl PresentEventQueue {
    /// Default queue capacity used by [`Default`].
    pub const DEFAULT_CAPACITY: usize = 64;

    /// Creates a queue with an explicit capacity.
    ///
    /// `capacity == 0` is promoted to `1`.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: BoundedQueue::with_capacity(capacity),
        }
    }

    /// Enqueues one presentation event.
    pub fn push(&mut self, event: PresentEvent) {
        self.inner.push(event);
    }

    /// Pops the oldest queued event, if any.
    pub fn pop(&mut self) -> Option<PresentEvent> {
        self.inner.pop()
    }

    /// Returns the current queue length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when no events are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of events dropped due to queue overflow.
    #[must_use]
    pub fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }
}

impl Default for PresentEventQueue {
    fn default() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::{PresentEvent, PresentEventQueue, SubmissionId, presentation_time_to_host_time};
    use subduction_core::output::OutputId;
    use subduction_core::time::HostTime;

    #[test]
    fn queue_overflow_drops_oldest_event() {
        let mut queue = PresentEventQueue::with_capacity(2);
        queue.push(PresentEvent::Discarded {
            id: SubmissionId(1),
        });
        queue.push(PresentEvent::Discarded {
            id: SubmissionId(2),
        });
        queue.push(PresentEvent::Discarded {
            id: SubmissionId(3),
        });

        assert_eq!(
            queue.pop(),
            Some(PresentEvent::Discarded {
                id: SubmissionId(2)
            })
        );
        assert_eq!(
            queue.pop(),
            Some(PresentEvent::Discarded {
                id: SubmissionId(3)
            })
        );
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn presented_event_round_trips_payload() {
        let event = PresentEvent::Presented {
            id: SubmissionId(9),
            actual_present: HostTime(123),
            refresh_interval: Some(16_666_667),
            output: Some(OutputId(4)),
            flags: 7,
        };

        let mut queue = PresentEventQueue::with_capacity(4);
        queue.push(event);
        assert_eq!(queue.pop(), Some(event));
    }

    #[test]
    fn zero_capacity_is_promoted_to_one() {
        let mut queue = PresentEventQueue::with_capacity(0);
        queue.push(PresentEvent::Discarded {
            id: SubmissionId(1),
        });
        queue.push(PresentEvent::Discarded {
            id: SubmissionId(2),
        });

        assert_eq!(queue.len(), 1);
        assert_eq!(
            queue.pop(),
            Some(PresentEvent::Discarded {
                id: SubmissionId(2)
            })
        );
        assert_eq!(queue.dropped_count(), 1);
    }

    // --- presentation_time_to_host_time tests ---

    #[test]
    fn timestamp_packing_normal_values() {
        // 1 second + 500_000_000 ns
        let t = presentation_time_to_host_time(0, 1, 500_000_000);
        assert_eq!(t, HostTime(1_500_000_000));
    }

    #[test]
    fn timestamp_packing_large_tv_sec_hi() {
        // tv_sec_hi = 1 means seconds = 1 << 32 = 4_294_967_296
        let t = presentation_time_to_host_time(1, 0, 0);
        assert_eq!(t, HostTime(4_294_967_296 * 1_000_000_000));
    }

    #[test]
    fn timestamp_packing_zero() {
        let t = presentation_time_to_host_time(0, 0, 0);
        assert_eq!(t, HostTime(0));
    }

    #[test]
    fn timestamp_packing_saturates_on_overflow() {
        // u32::MAX across all fields should saturate rather than panic.
        let t = presentation_time_to_host_time(u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(t, HostTime(u64::MAX));
    }

    #[test]
    fn tv_nsec_clamped_above_max() {
        // tv_nsec > 999_999_999 is clamped.
        let clamped = presentation_time_to_host_time(0, 0, 1_500_000_000);
        let expected = presentation_time_to_host_time(0, 0, 999_999_999);
        assert_eq!(clamped, expected);
    }

    #[test]
    fn tv_nsec_exact_max_is_not_clamped() {
        let t = presentation_time_to_host_time(0, 0, 999_999_999);
        assert_eq!(t, HostTime(999_999_999));
    }

    #[test]
    fn refresh_zero_produces_none() {
        // Verify the conversion convention: refresh == 0 → None.
        let refresh: u32 = 0;
        let interval: Option<u64> = if refresh == 0 {
            None
        } else {
            Some(u64::from(refresh))
        };
        assert_eq!(interval, None);
    }

    #[test]
    fn refresh_nonzero_produces_some() {
        let refresh: u32 = 16_666_667;
        let interval: Option<u64> = if refresh == 0 {
            None
        } else {
            Some(u64::from(refresh))
        };
        assert_eq!(interval, Some(16_666_667));
    }
}
