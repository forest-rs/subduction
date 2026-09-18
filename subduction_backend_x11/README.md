<div align="center">

# Subduction Backend X11

**X11 backend for subduction.**

</div>

`subduction_backend_x11` presents the subduction layer tree to an X11 window.
It assumes a server with the Present extension. It presents directly to a
top-level window and reads that window's vblank clock through Present; a
compositing manager is not required.

- **`X11PresentSource`** turns a window's Present completions into `frameclock`
  ticks (via `frameclock_x11`) and per-submission feedback. It is pure
  bookkeeping and references no `x11rb` type, so any event loop can drive it.
- **`X11Compositor`** is the presenting `Presenter`: it owns the X11 window and
  a wgpu surface, and composites the subduction layer tree onto it with
  `subduction_backend_wgpu`. Available on non-Apple Unix targets.

The compositor presents through wgpu's own swapchain. Per-pixmap Present
feedback only flows to a presenter that submits `PresentPixmap` itself, so in
this configuration the `X11PresentSource` feedback queue stays empty.

## Minimum supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE] or <http://www.apache.org/licenses/LICENSE-2.0>), or
- MIT license ([LICENSE-MIT] or <http://opensource.org/licenses/MIT>),

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[LICENSE-APACHE]: https://github.com/forest-rs/subduction/blob/main/LICENSE-APACHE
[LICENSE-MIT]: https://github.com/forest-rs/subduction/blob/main/LICENSE-MIT
