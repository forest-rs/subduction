// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained layer tree and backend presenter contract for compositing.
//!
//! `subduction_core` provides the foundational data structures for managing
//! retained compositing layer trees. It is `no_std` compatible (with `alloc`)
//! and uses array-based struct-of-arrays storage with index handles for
//! cache-friendly traversal.
//!
//! Display-frame timing and scheduling live in the [`frameclock`] crate. This
//! crate keeps compatibility re-exports for `scheduler`, `time`, and `timing`
//! while local callers migrate to direct `frameclock` imports. Media timeline
//! clocks live in the `mediaclock` crate.
//!
//! # Architecture
//!
//! The crate is organized around a frame loop that turns platform display
//! callbacks into incremental scene updates:
//!
//! ```text
//!   Backend (tick source)
//!       │
//!       ▼
//!   frameclock::FrameTick ──► frameclock::scheduler::Scheduler::plan()
//!                                           │
//!                                           ▼
//!                         frameclock::timing::FramePlan
//!                                           │
//!                 ┌─────────────────────────┘
//!                 ▼
//!   LayerStore::evaluate() ──► FrameChanges ──► Presenter::apply()
//!                                                    │
//!                 ┌──────────────────────────────────┘
//!                 ▼
//!   frameclock::timing::PresentFeedback ──► frameclock::scheduler::Scheduler::observe()
//! ```
//!
//! **[`layer`]** — Struct-of-arrays layer tree with generational handles.
//! Properties (transform, opacity, clip, content) are set by the caller;
//! world transforms and effective opacities are computed by evaluation.
//!
//! **[`dirty`]** — Multi-channel dirty tracking via `invalidation`.
//! Property mutations automatically mark the appropriate channel. TRANSFORM
//! and OPACITY propagate to descendants; CLIP and CONTENT are local-only;
//! TOPOLOGY triggers a traversal rebuild.
//!
//! **[`timing`]** — Compatibility re-export of `frameclock::timing`.
//!
//! **[`scheduler`]** — Compatibility re-export of `frameclock::scheduler`.
//!
//! **[`backend`]** — The [`Presenter`](backend::Presenter) trait that
//! platform backends implement to apply frame changes to native trees.
//!
//! **[`transform`]** — 3D affine transform type for layer positioning.
//!
//! **[`output`]** — Layer-root presentation policy such as the backdrop style,
//! plus a compatibility re-export of `frameclock::OutputId`.
//!
//! **[`trace`]** — [`TraceSink`](trace::TraceSink) trait, Subduction frame-loop
//! phase summaries, and rich layer/damage events. Timing diagnostics are
//! compatibility re-exports of the `frameclock` event types.
//!
//! # Crate features
//!
//! - `std` (disabled by default): Enables `std` support in dependencies and
//!   in `frameclock`.
//! - `trace` (disabled by default): Enables `Tracer` method bodies (one branch
//!   per call site).
//! - `trace-rich` (disabled by default, implies `trace`): Gates per-layer
//!   change and damage-rect events.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

pub mod backend;
pub mod dirty;
pub mod layer;
pub mod output;
pub mod trace;
pub mod transform;

/// Compatibility re-export for frame scheduling.
pub mod scheduler {
    pub use frameclock::scheduler::*;
}

/// Compatibility re-export for host-time types.
pub mod time {
    pub use frameclock::time::*;
}

/// Compatibility re-export for frame timing and feedback types.
pub mod timing {
    pub use frameclock::timing::*;
}
