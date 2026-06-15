// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland timing adapters for [`frameclock`].
//!
//! This crate owns Wayland-specific timing adaptation. It selects and reads
//! the compositor-aligned presentation clock as [`HostTime`], converts
//! `wl_surface.frame` callback completions into [`FrameTick`] values via
//! [`TickerState`], and carries `wp_presentation` feedback facts as
//! [`PresentEvent`] values.
//!
//! It intentionally does not own `wl_surface` objects, event queues, buffers,
//! registries, or protocol dispatch. Protocol I/O belongs to hosts and backend
//! crates; this crate owns the timing bookkeeping those hosts feed and poll.
//!
//! # Core Flow
//!
//! ```text
//! wl_surface.frame done           -> TickerState -> FrameTick
//! wp_presentation_feedback events -> PresentEvent -> PresentEventQueue
//! wp_presentation.clock_id        -> Clock -> HostTime reads
//! ```
//!
//! A host's frame-callback dispatch path has this shape:
//!
//! ```rust,ignore
//! use frameclock::OutputId;
//! use frameclock_wayland::{Clock, TickerState};
//!
//! let mut ticker = TickerState::new();
//! let clock = Clock::Monotonic;
//!
//! // Claim the single in-flight slot before sending a wl_surface.frame request:
//! if ticker.mark_callback_requested() {
//!     // send the wl_surface.frame request
//! }
//!
//! // When the matching wl_callback.done event arrives:
//! ticker.on_callback_done(clock, OutputId(0));
//!
//! // After dispatch, drain the queued ticks:
//! while let Some(tick) = ticker.poll_tick() {
//!     // Build a FrameOpportunity and plan the frame.
//!     _ = tick;
//! }
//! ```
//!
//! All `HostTime` values are nanosecond ticks. When the compositor advertises
//! `wp_presentation`, map its `clock_id` event to a [`Clock`] with
//! [`Clock::from_presentation_clock_id`] and read timing facts from that clock
//! so feedback timestamps and tick times stay in one time domain.
//!
//! This crate keeps its implementation `no_std` (with `alloc`), but reading
//! clocks requires an operating system. It is intended to be validated on
//! Linux targets, not on generic no-std targets such as `x86_64-unknown-none`.
//!
//! [`HostTime`]: frameclock::HostTime
//! [`FrameTick`]: frameclock::FrameTick

#![no_std]

extern crate alloc;

mod presentation;
mod queue;
mod tick;
mod time;

pub use presentation::{
    PresentEvent, PresentEventQueue, SubmissionId, presentation_time_to_host_time,
};
pub use tick::TickerState;
pub use time::{Clock, now, timebase};
