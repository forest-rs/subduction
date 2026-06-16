// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend-private presentation-feedback bookkeeping.
//!
//! The presentation-feedback contracts (`PresentEvent`, `PresentEventQueue`,
//! `SubmissionId`, and the timestamp conversion) live in `frameclock_wayland`.
//! This module holds only the dispatch-side state the backend accumulates
//! while a `wp_presentation_feedback` object is in flight.

use frameclock::OutputId;

/// Per-feedback-object accumulation state while the feedback is in flight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingFeedback {
    pub(crate) sync_output: Option<OutputId>,
}
