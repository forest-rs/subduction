// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dirty-tracking channel constants.
//!
//! Subduction uses multi-channel dirty tracking (via [`invalidation`]) to
//! efficiently propagate invalidation through the layer tree. Each channel
//! represents an independent category of change.
//!
//! # Propagation semantics
//!
//! Channels differ in whether dirtiness propagates to descendants:
//!
//! - **Propagating** — [`TRANSFORM`] and [`OPACITY`] use
//!   [`EagerPolicy`](invalidation::EagerPolicy) and have dependency
//!   edges from child to parent. Marking a parent dirty automatically marks
//!   all descendants, because world transforms, effective opacities, and
//!   effective hidden state are inherited properties. (Hidden-flag changes
//!   are routed through [`TRANSFORM`] so that the same drain pass
//!   recomputes both world transforms and `effective_hidden`.)
//!
//! - **Local-only** — [`CLIP`], [`CONTENT`], [`BOUNDS`], and [`BLEND_MODE`]
//!   are marked with the default policy. Only the explicitly marked layer
//!   appears in the drain output, since clip shapes, surface content, bounds,
//!   and blend modes are per-layer properties.
//!
//! - **Structural** — [`TOPOLOGY`] is marked on topology mutations
//!   (add/remove child, create/destroy layer). It triggers a traversal-order
//!   rebuild during evaluation but does not propagate to descendants.
//!
//! # Consumption
//!
//! Callers never need to query dirty state directly. Each
//! [`LayerStore::evaluate`](crate::layer::LayerStore::evaluate) call drains
//! all channels and surfaces the results as
//! [`FrameChanges`](crate::layer::FrameChanges), which backends
//! [consume](crate::backend::Presenter::apply) to apply incremental updates.

use invalidation::Channel;

/// Transform or hidden flag changed — requires world transform and effective
/// hidden recomputation for descendants.
pub const TRANSFORM: Channel = Channel::new(0);

/// Opacity changed — requires effective opacity recomputation for descendants.
pub const OPACITY: Channel = Channel::new(1);

/// Clip shape changed — no propagation needed.
pub const CLIP: Channel = Channel::new(2);

/// Surface content changed — no propagation needed.
pub const CONTENT: Channel = Channel::new(3);

/// Tree topology changed — triggers traversal order rebuild.
pub const TOPOLOGY: Channel = Channel::new(4);

/// Layer bounds changed — no propagation needed.
pub const BOUNDS: Channel = Channel::new(5);

/// Blend mode changed — no propagation needed.
pub const BLEND_MODE: Channel = Channel::new(6);
