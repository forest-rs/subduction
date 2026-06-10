// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Display-frame timing, scheduling, feedback, and diagnostics.
//!
//! `frameclock` turns platform display callbacks, display timing, and frame
//! demand into explicit frame plans. Backends provide timing facts such as a
//! callback time, a predicted presentation time, and a commit deadline. Hosts
//! provide demand and display constraints. [`FrameDriver`] owns the retained
//! frame lifecycle: it queues demand, decides when a frame is ready to prepare,
//! observes submission feedback, and returns a [`FrameTimingSummary`].
//!
//! The crate intentionally does not own windows, event loops, layer trees,
//! renderers, swapchains, or native presentation resources. Those belong in
//! platform adapters and higher-level engines.
//!
//! # API Surfaces
//!
//! The root module re-exports the frame-planning vocabulary used by both
//! retained [`FrameDriver`] hosts and lower-level [`Scheduler`] integrations:
//! demand, opportunities, active frames, submissions, scheduler configuration,
//! display timing, feedback, host time, timebase conversion, output ids, and
//! frame summaries.
//!
//! The modules group the same responsibilities more explicitly:
//!
//! - [`diagnostics`] exposes explicit event structs,
//!   [`DiagnosticsSink`](diagnostics::DiagnosticsSink), and
//!   [`FrameTimingSummaryBuilder`](diagnostics::FrameTimingSummaryBuilder) for
//!   telemetry adapters and tests.
//! - [`scheduler`], [`timing`], [`time`], [`timeline`], [`driver`], and
//!   [`demand`] expose the same public types grouped by responsibility.
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
//! A minimal host loop has this shape:
//!
//! ```rust,ignore
//! use frameclock::{
//!     DisplayTiming, Duration, FrameDemand, FrameDriver, FrameOpportunity, FrameSubmission,
//!     FrameTick, HostTime, OutputId, PresentHints, SchedulerConfig, TimingConfidence,
//! };
//!
//! let mut driver = FrameDriver::new(SchedulerConfig::pacing_only());
//!
//! // Input, animation, timers, or layout invalidation add demand.
//! driver.request(FrameDemand::ANIMATION);
//!
//! // A platform callback or event-loop redraw opportunity becomes a
//! // FrameOpportunity.
//! let tick = FrameTick {
//!     now: HostTime(1_000_000),
//!     predicted_present: None,
//!     refresh_interval: Some(16_666_667),
//!     confidence: TimingConfidence::PacingOnly,
//!     frame_index: 1,
//!     output: OutputId(0),
//!     prev_actual_present: None,
//! };
//! let hints = PresentHints {
//!     desired_present: None,
//!     latest_commit: HostTime(17_000_000),
//! };
//! let opportunity = FrameOpportunity::new(
//!     tick,
//!     hints,
//!     DisplayTiming::from_tick(&tick, Duration(16_666_667)),
//! );
//!
//! if let Some(frame) = driver.begin_frame(opportunity) {
//!     let sample_time = frame.sample_time();
//!     // Prepare app/model/render state for `sample_time`, submit renderer work,
//!     // then report submission facts back to frameclock.
//!     let summary = driver.submit_frame(
//!         frame,
//!         FrameSubmission::new(HostTime(2_000_000), None),
//!     );
//!     _ = summary;
//! }
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
pub use diagnostics::{FrameTimingBasis, FrameTimingSummary};
pub use driver::{ActiveFrame, FrameDriver, FrameOpportunity, FrameSubmission, PlannedFrame};
pub use output::OutputId;
pub use scheduler::{DegradationPolicy, Scheduler, SchedulerConfig, SchedulerState};
pub use time::{Duration, HostTime, Timebase};
pub use timeline::AffineClock;
pub use timing::{
    DisplayTiming, FramePlan, FrameRequest, FrameTick, PendingFeedback, PresentFeedback,
    PresentHints, TimingConfidence,
};
