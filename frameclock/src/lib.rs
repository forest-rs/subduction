// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Display-frame timing, scheduling, feedback, and diagnostics.
//!
//! `frameclock` turns platform display callbacks into explicit frame plans.
//! Backends provide timing facts such as a callback time, a predicted
//! presentation time, and a commit deadline. The scheduler converts those facts
//! into a [`FramePlan`] whose [`sample_time`](FramePlan::sample_time) is the
//! time applications should use for animations, simulations, media overlays,
//! and other frame-dependent state.
//!
//! The crate intentionally does not own windows, event loops, layer trees,
//! renderers, swapchains, or native presentation resources. Those belong in
//! platform adapters and higher-level engines.
//!
//! # Core Flow
//!
//! ```text
//! platform tick -> FrameTick + PresentHints
//!               -> Scheduler::plan()
//!               -> FramePlan
//!               -> build/submit frame
//!               -> PresentFeedback
//!               -> Scheduler::observe()
//! ```
//!
//! # Crate Features
//!
//! - `std` (disabled by default): reserved for future standard-library
//!   integration. The current API is `no_std`.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod diagnostics;
pub mod output;
pub mod scheduler;
pub mod time;
pub mod timeline;
pub mod timing;

pub use diagnostics::{
    Diagnostics, DiagnosticsSink, FramePlanEvent, FrameTickEvent, NoopDiagnostics,
    PresentFeedbackEvent, SchedulerStateEvent, SubmitEvent,
};
pub use output::OutputId;
pub use scheduler::{DegradationPolicy, Scheduler, SchedulerConfig, SchedulerState};
pub use time::{Duration, HostTime, Timebase};
pub use timeline::AffineClock;
pub use timing::{
    FramePlan, FrameTick, PendingFeedback, PresentFeedback, PresentHints, TimingConfidence,
};
