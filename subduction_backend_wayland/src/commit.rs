// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Commit wiring and presentation feedback tracking.

use crate::presentation::SubmissionId;
use wayland_client::backend::WaylandError;

/// Internal bookkeeping for the commit-sequencing path.
#[derive(Debug)]
pub(crate) struct CommitState {
    next_submission_id: u64,
    pending_count: u32,
    max_pending: u32,
}

impl CommitState {
    const DEFAULT_MAX_PENDING: u32 = 64;

    pub(crate) fn new() -> Self {
        Self {
            next_submission_id: 0,
            pending_count: 0,
            max_pending: Self::DEFAULT_MAX_PENDING,
        }
    }

    /// Allocates the next monotonically increasing [`SubmissionId`].
    pub(crate) fn allocate_id(&mut self) -> SubmissionId {
        let id = SubmissionId(self.next_submission_id);
        self.next_submission_id += 1;
        id
    }

    /// Increments the count of pending presentation feedback objects.
    pub(crate) fn increment_pending(&mut self) {
        self.pending_count += 1;
    }

    /// Decrements the count of pending presentation feedback objects,
    /// saturating at zero.
    pub(crate) fn decrement_pending(&mut self) {
        self.pending_count = self.pending_count.saturating_sub(1);
    }

    /// Returns `true` when the pending feedback count has reached the limit.
    pub(crate) fn is_at_limit(&self) -> bool {
        self.pending_count >= self.max_pending
    }

    /// Returns the current number of pending feedback objects.
    #[allow(dead_code, reason = "used by tests and future diagnostics")]
    pub(crate) fn pending_count(&self) -> u32 {
        self.pending_count
    }
}

/// Error returned by [`WaylandState::commit_frame`](crate::WaylandState::commit_frame).
#[derive(Debug)]
pub enum CommitFrameError {
    /// No surface registered via [`set_surface`](crate::WaylandState::set_surface).
    NoSurface,
    /// Flush failed after committing — requests may not have reached the compositor.
    Flush(WaylandError),
}

/// User data attached to backend-issued `wp_presentation_feedback` objects.
///
/// Public because embedded-mode hosts need it as a type parameter in
/// [`delegate_dispatch!`](wayland_client::delegate_dispatch).
#[derive(Debug, Clone, Copy)]
pub struct FeedbackData {
    pub(crate) submission_id: SubmissionId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_id_monotonically_increasing() {
        let mut state = CommitState::new();
        let a = state.allocate_id();
        let b = state.allocate_id();
        let c = state.allocate_id();
        assert_eq!(a, SubmissionId(0));
        assert_eq!(b, SubmissionId(1));
        assert_eq!(c, SubmissionId(2));
    }

    #[test]
    fn pending_count_tracks_correctly() {
        let mut state = CommitState::new();
        assert_eq!(state.pending_count(), 0);
        assert!(!state.is_at_limit());

        state.increment_pending();
        assert_eq!(state.pending_count(), 1);

        state.increment_pending();
        assert_eq!(state.pending_count(), 2);
    }

    #[test]
    fn pending_count_respects_limit() {
        let mut state = CommitState {
            next_submission_id: 0,
            pending_count: 0,
            max_pending: 2,
        };

        state.increment_pending();
        assert!(!state.is_at_limit());
        state.increment_pending();
        assert!(state.is_at_limit());
    }

    #[test]
    fn decrement_pending_saturates_at_zero() {
        let mut state = CommitState::new();
        assert_eq!(state.pending_count(), 0);
        state.decrement_pending();
        assert_eq!(state.pending_count(), 0);

        state.increment_pending();
        state.increment_pending();
        state.decrement_pending();
        assert_eq!(state.pending_count(), 1);
        state.decrement_pending();
        assert_eq!(state.pending_count(), 0);
        state.decrement_pending();
        assert_eq!(state.pending_count(), 0);
    }
}
