<div align="center">

# Frameclock Wayland

**Wayland timing adapters for `frameclock`.**

</div>

`frameclock_wayland` connects Wayland frame timing to `frameclock`. It selects
and reads the compositor-aligned presentation clock as `HostTime`, converts
`wl_surface.frame` callback completions into `FrameTick` values, and carries
`wp_presentation` feedback facts as `PresentEvent` values for scheduler
feedback.

The crate intentionally does not own `wl_surface` objects, event queues,
buffers, registries, or protocol dispatch. Protocol I/O belongs to hosts and
backend crates such as `subduction_backend_wayland`; this crate owns the
timing bookkeeping those hosts feed and poll. Present-hint computation and a
retained `FrameDriver` wrapper are left to future implementation here.

## Core Flow

```text
wl_surface.frame done           -> TickerState -> FrameTick
wp_presentation_feedback events -> PresentEvent -> PresentEventQueue
wp_presentation.clock_id        -> Clock -> HostTime reads
```

Use `TickerState` as the frame-callback bookkeeping for one surface: call
`mark_callback_requested` to claim the single in-flight slot before sending a
`wl_surface.frame` request (it returns `false` if a callback is already in
flight), call `on_callback_done` when the matching `wl_callback.done` event
arrives, and drain resulting ticks with `poll_tick`. A `TickerState` models a single paced
surface/output stream — create one per `wl_surface`, pass it a stable
`OutputId`, and feed it only that surface's presentation feedback. Hosts that
multiplex several surfaces on one queue should keep a `TickerState` per stream
and correlate feedback to the right stream by `SubmissionId` themselves.

Use `presentation_time_to_host_time` to convert
`wp_presentation_feedback.presented` timestamps, store the most recent value
via `TickerState::set_last_observed_actual_present` so the next tick carries
`FrameTick::prev_actual_present`, and queue per-commit `PresentEvent`s in a
`PresentEventQueue` correlated by `SubmissionId`.

Use `Clock::from_presentation_clock_id` to map the `wp_presentation.clock_id`
event to a `Clock`, and read all timing facts from that clock so feedback
timestamps and tick times stay in one time domain.

## Timing Model

`now`, `Clock::now`, and all converted timestamps are nanosecond ticks;
`timebase` returns the identity nanosecond `Timebase`.

Wayland frame callbacks carry no predicted present time or refresh interval,
so emitted `FrameTick`s are pacing-only facts. Actual presentation evidence
arrives separately through `wp_presentation` feedback: the previous frame's
actual present time is surfaced as `FrameTick::prev_actual_present`, and the
full per-commit event stream is available as `PresentEvent` values for hosts
that resolve feedback by `SubmissionId`.

When the compositor advertises `wp_presentation`, its `clock_id` event names
the clock domain of all feedback timestamps. Hosts should switch their reads
to that clock (`Clock::Presentation`) so `HostTime` comparisons remain valid.
`Clock::Monotonic` is the fallback when `wp_presentation` is missing or the
advertised clock is unknown.

Compositors stop delivering frame callbacks while a surface is occluded or
minimised, so the tick stream can stall. Hosts should treat tick starvation
as normal Wayland behaviour: idle, apply a timeout, or fall back to a
timer-based tick source.

## no_std

This crate keeps its implementation `no_std` (with `alloc`), but reading
clocks requires an operating system. It is validated on Linux targets instead
of the workspace's generic `x86_64-unknown-none` no-std target.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
