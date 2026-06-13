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
//! The root module re-exports the common retained-host integration surface:
//! frame demand, timing facts needed to build a [`FrameOpportunity`],
//! [`FrameDriver`] lifecycle types, [`SchedulerConfig`], host time, output ids,
//! and frame summaries.
//!
//! More specialized APIs live under their modules:
//!
//! - [`scheduler`] exposes [`Scheduler`](scheduler::Scheduler),
//!   [`DegradationPolicy`](scheduler::DegradationPolicy), and adaptation state
//!   for custom low-level integrations.
//! - [`diagnostics`] exposes explicit event structs,
//!   [`DiagnosticsSink`](diagnostics::DiagnosticsSink), and
//!   [`FrameTimingSummaryBuilder`](diagnostics::FrameTimingSummaryBuilder) for
//!   telemetry adapters and tests.
//! - [`time`] exposes [`Timebase`](time::Timebase) for backend clock
//!   conversion. Media timeline mapping lives in the `mediaclock` crate.
//! - [`timing`] and [`driver`] expose lower-level lifecycle and presentation
//!   feedback types such as [`PresentFeedback`](timing::PresentFeedback),
//!   [`PendingFeedback`](timing::PendingFeedback), and
//!   [`PlannedFrame`](driver::PlannedFrame).
//!
//! # Core Flow
//!
//! ```text
//! platform tick -> FrameOpportunity
//!               -> FrameDriver::begin_frame()
//!               -> FrameBegin { result: FrameBeginResult::Ready(ActiveFrame), ... }
//!               -> build frame
//!               -> FrameDriver::submit_frame() or FrameDriver::discard_frame()
//!               -> FrameTimingSummary
//! ```
//!
//! A minimal host loop has this shape:
//!
//! ```rust,ignore
//! use frameclock::{
//!     Duration, FrameBeginResult, FrameDemand, FrameDriver, FrameOpportunity,
//!     FrameSubmission, HostTime, OutputId, SchedulerConfig,
//! };
//!
//! let mut driver = FrameDriver::new(SchedulerConfig::pacing_only());
//!
//! // Input, animation, timers, or layout invalidation add demand.
//! driver.request(FrameDemand::ANIMATION);
//!
//! // A platform callback or event-loop redraw opportunity becomes a
//! // FrameOpportunity. Pacing-only hosts do not have a predicted present time.
//! let opportunity = FrameOpportunity::pacing_only(
//!     HostTime(1_000_000),
//!     Duration(16_666_667),
//!     1,
//!     OutputId(0),
//! );
//!
//! let begin = driver.begin_frame(opportunity);
//! if let Some(summary) = begin.resolved_feedback {
//!     // A previous deferred submission resolved on this tick.
//!     _ = summary;
//! }
//!
//! match begin.result {
//!     FrameBeginResult::Ready(frame) => {
//!         let sample_time = frame.sample_time();
//!         // Prepare app/model/render state for `sample_time`, submit renderer work,
//!         // then report submission facts back to frameclock. If the frame cannot
//!         // be submitted, call `discard_frame` instead.
//!         let submit = driver.submit_frame(
//!             frame,
//!             FrameSubmission::new(HostTime(2_000_000), None),
//!         );
//!         _ = submit.summary;
//!     }
//!     FrameBeginResult::WaitUntil(frame_start) => {
//!         // Mirror `frame_start` into the host timer queue and wait.
//!         _ = frame_start;
//!     }
//!     FrameBeginResult::Expired(summary) => {
//!         // The queued plan missed its commit deadline before it was released.
//!         // Record the dropped-frame summary and request fresh demand if needed.
//!         _ = summary;
//!     }
//!     FrameBeginResult::Idle => {
//!         // Wait for input, animation, timers, or other app demand.
//!     }
//! }
//! ```
//!
//! `FrameDemand` is host-owned pending work and the semantic cause carried into
//! the resulting frame plan. Request `INPUT` for discrete user action,
//! `CONTINUOUS_INPUT` while an interaction is active, `ANIMATION` while a visual
//! timeline is running, and `BACKGROUND` for deferrable visual work. After
//! [`FrameBeginResult::Ready`] returns, use `frame.plan().demand` to choose the
//! workload for that frame: interactive frames can skip optional refinement,
//! animation frames can use normal visual work, and background frames can batch
//! or defer.
//!
//! `FrameDemandClass` is the derived ordering used by
//! [`FrameDemand::dominant_class`] and [`FrameDemand::preempts`]; use it for
//! diagnostics or adapter policy that needs to match frameclock's demand order.
//!
//! `DisplayTiming` should come from the backend/platform facts for the
//! opportunity's current target output. If a window or surface moves to another
//! display, or the platform reports a changed display mode, build the next
//! [`FrameOpportunity`] with timing for that new output instead of reusing stale
//! app-global display timing.
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
pub mod timing;

pub use demand::FrameDemand;
pub use diagnostics::FrameTimingSummary;
pub use driver::{
    ActiveFrame, FrameBegin, FrameBeginResult, FrameDriver, FrameSubmission, FrameSubmitResult,
    PresentationObservation,
};
pub use output::OutputId;
pub use scheduler::SchedulerConfig;
pub use time::{Duration, HostTime};
pub use timing::{DisplayTiming, FrameOpportunity, FrameTick, PresentHints};
