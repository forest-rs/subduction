// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Core types and layer tree for timing-synchronized compositing.
//!
//! `subduction_core` provides the foundational data structures for managing
//! trees of compositing layers with high-precision timing. It is `no_std`
//! compatible (with `alloc`) and uses array-based struct-of-arrays storage with index
//! handles for cache-friendly traversal.
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
//!   FrameTick ──► Scheduler::plan() ──► FramePlan
//!                                           │
//!                 ┌─────────────────────────┘
//!                 ▼
//!   LayerStore::evaluate() ──► FrameChanges ──► Presenter::apply()
//!                                                    │
//!                 ┌──────────────────────────────────┘
//!                 ▼
//!   PresentFeedback ──► Scheduler::observe()
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
//! **[`timing`]** — Capability-graded timing model. Types flow from backend
//! tick sources through the scheduler and back as feedback.
//!
//! **[`scheduler`]** — Adaptive pipeline-depth scheduler that converts ticks
//! into frame plans and adjusts based on observed build costs and deadline
//! hits/misses.
//!
//! **[`backend`]** — The [`Presenter`](backend::Presenter) trait that
//! platform backends implement to apply frame changes to native trees.
//!
//! **[`clock`]** — `AffineClock` for smoothed time mapping (A/V sync).
//!
//! **[`transform`]** — 3D affine transform type for layer positioning.
//!
//! **[`trace`]** — [`TraceSink`](trace::TraceSink) trait and event types for
//! frame-loop instrumentation, with zero-overhead [`Tracer`](trace::Tracer)
//! wrapper.
//!
//! # Crate features
//!
//! - `std` (disabled by default): Enables `std` support in dependencies.
//! - `trace` (disabled by default): Enables `Tracer` method bodies (one branch
//!   per call site).
//! - `trace-rich` (disabled by default, implies `trace`): Gates per-layer
//!   change and damage-rect events.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

pub mod backend;
pub mod clock;
pub mod dirty;
pub mod layer;
pub mod output;
pub mod scheduler;
pub mod time;
pub mod timing;
pub mod trace;
pub mod transform;
