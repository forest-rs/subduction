<div align="center">

# Frameclock Web

**Browser timing adapters for `frameclock`.**

</div>

`frameclock_web` connects browser frame callbacks to `frameclock`. It converts
`requestAnimationFrame` timestamps and `performance.now()` into `frameclock`
host-time ticks, provides a `RafLoop` tick source, and offers `WebFrameClock`
as a retained wrapper around `FrameDriver`.

The crate intentionally does not own DOM presentation, WebGL, WebGPU,
application state, renderer submission, or browser event routing.

## Core Flow

```text
requestAnimationFrame -> FrameTick
                      -> WebFrameClock::begin_frame()
                      -> FrameBeginResult::Ready(ActiveFrame)
                      -> host render
                      -> WebFrameClock::submit_frame() or WebFrameClock::discard_frame()
                      -> FrameTimingSummary
```

Use `RafLoop` when an application wants this crate to register and maintain a
browser `requestAnimationFrame` loop. Each callback receives a `FrameTick` in
browser host time.

Use `WebFrameClock` when an application wants retained frame lifecycle state:
pending demand, queued frame-start plans, stronger-demand preemption,
submission summaries, and dropped-frame summaries. Hosts still decide when a
frame is needed, what to render, and where the rendered output is submitted.

Browser RAF does not expose a portable predicted present timestamp, commit
deadline, or current display refresh interval. `WebFrameClock` therefore
creates pacing-only `FrameOpportunity` values and uses a fallback refresh
interval for display timing. The default fallback is `DEFAULT_REFRESH_INTERVAL`,
a 60 Hz interval in microsecond ticks.

```rust,ignore
use frameclock::{
    FrameBeginResult, FrameDemand, FrameSubmission, OutputId, SchedulerConfig,
};
use frameclock_web::{DEFAULT_REFRESH_INTERVAL, RafLoop, WebFrameClock};

let mut clock = WebFrameClock::new(
    SchedulerConfig::pacing_only(),
    DEFAULT_REFRESH_INTERVAL,
);

let raf = RafLoop::new(
    move |tick| {
        clock.request(FrameDemand::ANIMATION);

        match clock.begin_frame(tick) {
            FrameBeginResult::Ready(frame) => {
                let sample_time = frame.sample_time();
                // Prepare and submit browser rendering work for sample_time.
                let summary =
                    clock.submit_frame(frame, FrameSubmission::new(frameclock_web::now(), None));
                _ = (sample_time, summary);
            }
            FrameBeginResult::WaitUntil(frame_start) => {
                // Mirror frame_start into the host's timer/redraw machinery.
                _ = frame_start;
            }
            FrameBeginResult::Expired(summary) => {
                // Record the dropped-frame summary and request fresh demand if needed.
                _ = summary;
            }
            FrameBeginResult::Idle => {}
        }
    },
    OutputId(0),
);

raf.start();
```

## API Surfaces

The root module exposes the browser integration surface:

- `RafLoop` for `requestAnimationFrame` callbacks.
- `WebFrameClock` for retained `FrameDriver` integration.
- `now` and `timebase` for browser host-time conversion.
- `present_hints`, `compute_present_hints`, and `display_timing` for hosts that
  need lower-level timing facts.
- `DEFAULT_REFRESH_INTERVAL` for conservative pacing fallback.

`frameclock_web` keeps platform-specific browser code out of `frameclock`
proper. Core scheduling policy, frame demand ordering, frame summaries, and
diagnostics stay in `frameclock`.

## Timing Model

`TIMEBASE` uses microsecond ticks: `1 tick = 1_000 ns`. This matches browser
`DOMHighResTimeStamp` values after converting milliseconds to microseconds.

`RafLoop` increments `FrameTick::frame_index` once per delivered RAF callback.
Applications that bypass `RafLoop` and create their own ticks should keep the
same per-output monotonic ownership rule: the frame index identifies a
delivered browser frame opportunity for one output or surface.

Because RAF is pacing-only, `PresentHints::desired_present` is `None` and
`PresentHints::latest_commit` is the RAF tick time. Hosts that can get richer
browser timing from media APIs, such as video frame callbacks, should build
their own `FrameOpportunity` or use a future media-specific adapter instead of
forcing that data through plain RAF.

## Feature Flags

This crate currently has no feature flags.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
