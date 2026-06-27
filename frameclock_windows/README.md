<div align="center">

# Frameclock Windows

**Windows timing adapters for `frameclock`.**

</div>

`frameclock_windows` connects Windows frame timing to `frameclock`. It reads
the QPC (`QueryPerformanceCounter`) host clock as `HostTime` / `Timebase`,
and builds `FrameTick` values for `VSync`-paced hosts.

The crate intentionally does not own `HWND`s, message loops, `DwmFlush`
pacing threads, or present-hint policy. Those belong to hosts and backend
crates such as `subduction_backend_windows`; this crate owns the clock reads
and tick bookkeeping those hosts feed and poll.

## Core Flow

```text
QueryPerformanceCounter / QueryPerformanceFrequency -> now / timebase -> HostTime
VSync-paced tick                                    -> make_tick      -> FrameTick
```

Use `now` and `timebase` to read the QPC clock as a `HostTime` /
`Timebase` pair (`nanos = ticks * timebase.numer / timebase.denom`). Call
`make_tick` from a `VSync`-paced tick handler (for example, one driven by
`DwmFlush` or a swapchain frame-latency waitable) to build a `FrameTick` from
the refresh interval, frame index, and previous actual present time.

## Timing Model

QPC frequency is read once and cached for the lifetime of the process via
`QueryPerformanceFrequency`. `now` reads `QueryPerformanceCounter` directly,
so all `HostTime` values are raw QPC ticks, not nanoseconds; convert with
`timebase` when nanosecond values are required.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
