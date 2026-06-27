// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Windows timing adapters for [`frameclock`].
//!
//! This crate owns Windows-specific timing adaptation. It reads the QPC
//! (`QueryPerformanceCounter`) host clock as [`HostTime`] / [`Timebase`] and
//! builds [`FrameTick`] values for `VSync`-paced hosts via [`make_tick`].
//!
//! It intentionally does not own `HWND`s, message loops, `DwmFlush` pacing
//! threads, or present-hint policy. Those belong to hosts and backend crates
//! such as `subduction_backend_windows`; this crate owns the clock reads and
//! tick bookkeeping those hosts feed and poll.
//!
//! # Core Flow
//!
//! ```text
//! QueryPerformanceCounter / QueryPerformanceFrequency -> now / timebase -> HostTime
//! VSync-paced tick                                    -> make_tick      -> FrameTick
//! ```
//!
//! [`HostTime`]: frameclock::HostTime
//! [`FrameTick`]: frameclock::FrameTick
//! [`Timebase`]: frameclock::time::Timebase

#![expect(
    unsafe_code,
    reason = "Windows timing adapters require QueryPerformanceCounter/Frequency FFI"
)]

mod tick;
mod time;

pub use tick::make_tick;
pub use time::{now, timebase};
