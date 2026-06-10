// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Display-frame timing, scheduling, feedback, and diagnostics.
//!
//! `frameclock` turns platform display callbacks, display timing, and frame
//! demand into explicit frame plans. Backends provide timing facts such as a
//! callback time, a predicted presentation time, and a commit deadline. Hosts
//! provide demand and display constraints. The scheduler converts those facts
//! into a [`FramePlan`] whose [`frame_start`](FramePlan::frame_start) says when
//! to begin frame work and whose [`sample_time`](FramePlan::sample_time) says
//! what presentation time animations, simulations, media overlays, and other
//! frame-dependent state should target.
//!
//! The crate intentionally does not own windows, event loops, layer trees,
//! renderers, swapchains, or native presentation resources. Those belong in
//! platform adapters and higher-level engines.
//!
//! # Core Flow
//!
//! ```text
//! platform tick -> FrameOpportunity
//!               -> FrameDriver::begin_frame()
//!               -> ActiveFrame
//!               -> build/submit frame
//!               -> FrameSubmission
//!               -> FrameDriver::submit_frame()
//!               -> FrameTimingSummary
//! ```
//!
//! # Crate Features
//!
//! - `std` (disabled by default): reserved for future standard-library
//!   integration. The current API is `no_std`.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod demand;
pub mod diagnostics;
pub mod driver;
pub mod output;
pub mod scheduler;
pub mod time;
pub mod timeline;
pub mod timing;

pub use demand::{FrameDemand, FrameDemandClass};
pub use diagnostics::{
    Diagnostics, DiagnosticsSink, FramePlanEvent, FrameTickEvent, FrameTimingBasis,
    FrameTimingSummary, FrameTimingSummaryBuilder, NoopDiagnostics, PresentFeedbackEvent,
    SchedulerStateEvent, SubmitEvent,
};
pub use driver::{ActiveFrame, FrameDriver, FrameOpportunity, FrameSubmission, PlannedFrame};
pub use output::OutputId;
pub use scheduler::{DegradationPolicy, Scheduler, SchedulerConfig, SchedulerState};
pub use time::{Duration, HostTime, Timebase};
pub use timeline::AffineClock;
pub use timing::{
    DisplayTiming, FramePlan, FrameRequest, FrameTick, PendingFeedback, PresentFeedback,
    PresentHints, TimingConfidence,
};
