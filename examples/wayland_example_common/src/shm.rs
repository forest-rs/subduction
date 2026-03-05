// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal `wl_shm` buffer helpers for examples.
//!
//! Creates solid-color ARGB8888 buffers via a shared `wl_shm_pool` backed by
//! a memfd. Pixel data is written once at creation time.

use std::io::Write;
use std::os::fd::AsFd;

use wayland_client::protocol::{wl_buffer, wl_shm, wl_shm_pool};
use wayland_client::{Connection, Dispatch, QueueHandle};

/// A `wl_shm_pool` backed by a memfd, supporting bump allocation.
#[derive(Debug)]
pub struct ShmPool {
    pool: wl_shm_pool::WlShmPool,
    file: std::fs::File,
    size: usize,
    offset: i32,
}

/// A `wl_buffer` allocated from a [`ShmPool`].
#[derive(Debug)]
pub struct ShmBuffer {
    /// The Wayland buffer proxy.
    pub buffer: wl_buffer::WlBuffer,
}

impl ShmPool {
    /// Creates a pool of `size` bytes.
    pub fn new<D>(shm: &wl_shm::WlShm, size: usize, qh: &QueueHandle<D>) -> Self
    where
        D: Dispatch<wl_shm_pool::WlShmPool, ()> + 'static,
    {
        let file = create_memfd(size);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "pool sizes in examples are small enough to fit in i32"
        )]
        let pool = shm.create_pool(file.as_fd(), size as i32, qh, ());
        Self {
            pool,
            file,
            size,
            offset: 0,
        }
    }

    /// Allocates a solid-color ARGB8888 buffer.
    ///
    /// Writes pixel data into the pool and returns a `wl_buffer`.
    /// Returns `None` if the pool is exhausted.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "buffer dimensions are small example sizes that fit in i32"
    )]
    pub fn alloc_solid<D>(
        &mut self,
        width: u32,
        height: u32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
        qh: &QueueHandle<D>,
    ) -> Option<ShmBuffer>
    where
        D: Dispatch<wl_buffer::WlBuffer, ()> + 'static,
    {
        let w = width as i32;
        let h = height as i32;
        let stride = w * 4;
        let buf_bytes = (stride * h) as usize;
        if self.offset as usize + buf_bytes > self.size {
            return None;
        }

        // Write ARGB8888 pixel data at the current offset.
        let pixel = [b, g, r, a]; // BGRA byte order = ARGB8888 in little-endian
        let row: Vec<u8> = pixel.repeat(width as usize);
        use std::io::Seek;
        self.file
            .seek(std::io::SeekFrom::Start(self.offset as u64))
            .expect("seek failed");
        for _ in 0..height {
            self.file.write_all(&row).expect("write failed");
        }

        let buffer = self.pool.create_buffer(
            self.offset,
            w,
            h,
            stride,
            wl_shm::Format::Argb8888,
            qh,
            (),
        );
        self.offset += buf_bytes as i32;
        Some(ShmBuffer { buffer })
    }
}

fn create_memfd(size: usize) -> std::fs::File {
    use std::os::fd::FromRawFd;

    // SAFETY: `memfd_create` with a static name is a simple syscall.
    let fd = unsafe {
        libc::memfd_create(
            c"subduction-shm".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    assert!(fd >= 0, "memfd_create failed");
    // SAFETY: fd is valid and exclusively owned.
    let file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.set_len(size as u64).expect("ftruncate failed");
    file
}

// --- No-op dispatch impls for protocol objects ---

/// Dispatch state marker for `wl_shm` events.
pub(crate) struct ShmDispatch;

impl<D> Dispatch<wl_shm::WlShm, (), D> for ShmDispatch
where
    D: Dispatch<wl_shm::WlShm, ()>,
{
    fn event(
        _state: &mut D,
        _proxy: &wl_shm::WlShm,
        _event: wl_shm::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
    }
}

impl<D> Dispatch<wl_shm_pool::WlShmPool, (), D> for ShmDispatch
where
    D: Dispatch<wl_shm_pool::WlShmPool, ()>,
{
    fn event(
        _state: &mut D,
        _proxy: &wl_shm_pool::WlShmPool,
        _event: wl_shm_pool::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
    }
}

impl<D> Dispatch<wl_buffer::WlBuffer, (), D> for ShmDispatch
where
    D: Dispatch<wl_buffer::WlBuffer, ()>,
{
    fn event(
        _state: &mut D,
        _proxy: &wl_buffer::WlBuffer,
        _event: wl_buffer::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
    }
}
