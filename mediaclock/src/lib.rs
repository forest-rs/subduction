// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Media timeline clocks and synchronization helpers.
//!
//! `mediaclock` maps `frameclock` host times into media timelines such as video
//! PTS or an audio-master playback clock. It is meant to sit above frame pacing:
//! `frameclock` decides when a frame should be prepared and presented, while
//! `mediaclock` helps decide which media time that planned frame should show.
//!
//! The crate intentionally does not own decoders, audio devices, native media
//! players, renderers, event loops, or presentation feedback. Platform adapters
//! and host applications feed it timing observations and apply the returned
//! media-time decisions.
//!
//! # Core Flow
//!
//! ```text
//! frameclock FramePlan sample/target host time
//!              -> MediaTimeline
//!              -> media seconds / PTS
//!              -> host chooses media content
//! ```
//!
//! Use [`MediaTimeline`] for ordinary playback timelines, including playback
//! rate changes and pause/resume boundaries. Use [`AffineClock`] directly when a
//! host needs only the lower-level smoothed affine mapping.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod affine;
mod timeline;

pub use affine::{AffineClock, AffineClockUpdate};
pub use timeline::MediaTimeline;
