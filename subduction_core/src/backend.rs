// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend contract for platform integrations.
//!
//! Subduction splits platform-specific work into *backend* crates. Each
//! backend provides the following pieces:
//!
//! - **Tick source** — Produces [`FrameTick`] values via a platform mechanism
//!   (e.g. `CADisplayLink` callback, `requestAnimationFrame`). This is
//!   backend-specific and not abstracted by a trait because the setup and
//!   lifecycle differ fundamentally across platforms.
//!
//! - **Time** — `now() -> HostTime` and `timebase() -> Timebase` free
//!   functions that read the platform's monotonic clock.
//!
//! - **Hint computation** — A `compute_present_hints(&FrameTick, Duration)
//!   -> PresentHints` free function. This is stateless and varies by
//!   platform (Apple uses predicted present times; web has pacing-only
//!   timing), so it stays as a free function rather than a trait method.
//!
//! - **Presenter** — Implements the [`Presenter`] trait to apply frame
//!   changes to a platform-native tree (e.g. `CALayer`, DOM elements).
//!
//! - **Feedback** — Uses [`PresentFeedback::new`] to report timing
//!   observations back to the [`Scheduler`](crate::scheduler::Scheduler),
//!   preserving `missed_deadline: None` when the backend lacks enough timing
//!   information to classify a frame honestly.
//!
//! # Crate boundaries
//!
//! `subduction_core` owns the data model, evaluation, scheduling, and this
//! contract module. Backend crates depend on `subduction_core` and provide
//! platform glue. Application code depends on both and wires them together
//! in a frame loop.
//!
//! [`FrameTick`]: crate::timing::FrameTick
//! [`PresentFeedback::new`]: crate::timing::PresentFeedback::new
//! [`PresentHints`]: crate::timing::PresentHints

use crate::layer::{FrameChanges, LayerStore};

/// Applies evaluated frame changes to a platform-native presentation tree.
///
/// Both `CALayer`-based and DOM-based presenters implement this trait,
/// enabling generic frame loops and test doubles.
///
/// # Frame loop pseudocode
///
/// A typical frame callback wires the pieces together like this:
///
/// ```rust,ignore
/// fn on_frame(tick: FrameTick) {
///     let hints = compute_present_hints(&tick, safety);
///     let plan = scheduler.plan(&tick, &hints);
///
///     // Animate: update layer properties using plan.semantic_time
///     store.set_transform(layer, animated_transform(plan.semantic_time));
///
///     // Evaluate: drain dirty channels, recompute world properties
///     let changes = store.evaluate();
///
///     // Present: apply incremental changes to the native tree
///     presenter.apply(&store, &changes);
///
///     // Feedback: report timing observations for adaptation
///     let feedback = PresentFeedback::new(&hints, build_start, now(), actual);
///     scheduler.observe(&feedback);
/// }
/// ```
pub trait Presenter {
    /// Applies the given [`FrameChanges`] to the backing presentation tree,
    /// reading current property values from `store` as needed.
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges);
}
