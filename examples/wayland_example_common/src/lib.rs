// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared helpers for Wayland examples.
//!
//! Provides xdg-shell window creation, `wl_shm` buffer allocation, and
//! Wayland dispatch wiring that the individual examples share.

#![expect(unsafe_code, reason = "wl_shm requires unsafe memfd/mmap")]

mod shm;
mod xdg;

pub use shm::{ShmBuffer, ShmPool};
pub use xdg::{ExampleState, XdgState, XdgWindow, create_window};

/// Creates a new `ShmPool` + `ShmBuffer` for the root surface background.
///
/// The buffer is a solid dark color at `(width, height)` dimensions.
/// Returns the pool and buffer. The caller must attach the buffer to the
/// root surface and commit.
pub fn create_background(
    shm: &wayland_client::protocol::wl_shm::WlShm,
    width: u32,
    height: u32,
    qh: &wayland_client::QueueHandle<ExampleState>,
) -> (ShmPool, ShmBuffer) {
    let size = width as usize * height as usize * 4;
    let mut pool = ShmPool::new(shm, size, qh);
    let buf = pool
        .alloc_solid(width, height, 25, 25, 36, 255, qh)
        .expect("bg pool exhausted");
    (pool, buf)
}
