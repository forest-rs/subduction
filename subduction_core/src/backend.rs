// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend contract for platform integrations.
//!
//! Subduction splits platform-specific work into *backend* crates. Each
//! backend provides the following piece:
//!
//! - **Presenter** — Implements the [`Presenter`] trait to apply evaluated
//!   frame changes to a platform-native tree, such as `CALayer`, DOM elements,
//!   DirectComposition visuals, or Wayland subsurfaces.
//!
//! Frame timing is owned by `frameclock` and adapter crates such as
//! `frameclock_apple` and `frameclock_web`. Those crates produce
//! `frameclock::FrameTick` values, compute present hints, expose host-time
//! helpers, and drive retained `frameclock::FrameDriver` state. Application
//! code wires a timing adapter together with a `Presenter`.
//!
//! # Crate boundaries
//!
//! `subduction_core` owns the layer data model, evaluation, and this presenter
//! contract. Backend crates depend on `subduction_core` and provide native
//! presentation glue. Application code depends on `subduction_core`, one
//! presenter backend, and one timing adapter, then wires them together in a
//! frame loop.

use crate::layer::{FrameChanges, LayerStore};

/// Applies evaluated frame changes to a platform-native presentation tree.
///
/// Both `CALayer`-based and DOM-based presenters implement this trait,
/// enabling generic frame loops and test doubles.
///
/// # Frame loop pseudocode
///
/// A typical frame callback wires a `frameclock` adapter to a presenter like
/// this:
///
/// ```rust,ignore
/// fn on_frame(tick: FrameTick) {
///     frame_clock.request(FrameDemand::ANIMATION);
///     let FrameBeginResult::Ready(frame) = frame_clock.begin_frame(tick) else {
///         return;
///     };
///     let sample_time = frame.sample_time();
///
///     // Animate: update layer properties using sample_time.
///     store.set_transform(layer, animated_transform(sample_time));
///
///     // Evaluate: drain dirty channels, recompute world properties
///     let changes = store.evaluate();
///
///     // Present: apply incremental changes to the native tree
///     presenter.apply(&store, &changes);
///
///     // Feedback: report submission through the timing adapter.
///     frame_clock.submit_frame(frame, FrameSubmission::new(now(), actual));
/// }
/// ```
pub trait Presenter {
    /// Applies the given [`FrameChanges`] to the backing presentation tree,
    /// reading current property values from `store` as needed.
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges);
}
