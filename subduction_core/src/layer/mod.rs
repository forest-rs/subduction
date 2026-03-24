// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer tree data model.
//!
//! A *layer* is a node in a compositing tree. Each layer has:
//!
//! - An identity ([`LayerId`]) — a generational handle that becomes stale when
//!   the layer is destroyed, preventing use-after-free bugs at the API level.
//! - Topology — parent, first-child, and sibling links forming an ordered tree.
//! - **Local properties** set by the caller: [`transform`](LayerStore::set_transform),
//!   [`opacity`](LayerStore::set_opacity), [`clip`](LayerStore::set_clip),
//!   [`content`](LayerStore::set_content), [`bounds`](LayerStore::set_bounds),
//!   and [`flags`](LayerStore::set_flags).
//! - **Computed properties** produced by [`evaluate`](LayerStore::evaluate):
//!   `world_transform` (product of ancestor local transforms) and
//!   `effective_opacity` (product of ancestor local opacities).
//!
//! Layers are stored in struct-of-arrays layout with index-based handles
//! for cache-friendly traversal.
//!
//! # Dirty tracking
//!
//! Property mutations automatically mark the corresponding dirty channel
//! (see [`dirty`](crate::dirty)). The channels map to property categories:
//!
//! - **TRANSFORM** / **OPACITY** — propagate to all descendants, since
//!   world transforms and effective opacities are inherited.
//! - **CLIP** / **CONTENT** / **BOUNDS** — local-only; only the modified layer is marked.
//! - **TOPOLOGY** — structural changes (add/remove child, create/destroy
//!   layer) that trigger a traversal-order rebuild.

mod clip;
mod evaluate;
mod hit_test;
mod id;
mod store;
mod traverse;

pub use clip::ClipShape;
pub use evaluate::FrameChanges;
pub use hit_test::HitEntry;
pub use id::{INVALID, LayerId, SurfaceId};
pub use store::{LayerFlags, LayerStore};
pub use traverse::Children;
