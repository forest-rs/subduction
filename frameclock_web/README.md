<div align="center">

# Frameclock Web

**Browser timing adapters for `frameclock`.**

</div>

`frameclock_web` connects browser frame callbacks to `frameclock`. It converts
`requestAnimationFrame` timestamps and `performance.now()` into `frameclock`
host-time ticks, provides a `RafLoop` tick source, and builds
`FrameOpportunity` values for hosts that own `FrameDriver`.

The crate intentionally does not own DOM presentation, WebGL, WebGPU,
application state, renderer submission, or browser event routing.

## Core Flow

```text
requestAnimationFrame -> FrameTick
                      -> frameclock_web::frame_opportunity()
                      -> FrameDriver::begin_frame()
                      -> FrameBegin { result: FrameBeginResult::Ready(ActiveFrame), ... }
                      -> host render
                      -> FrameDriver::submit_frame() or FrameDriver::discard_frame()
                      -> FrameTimingSummary
```

Use `RafLoop` when an application wants this crate to register and maintain a
browser `requestAnimationFrame` loop. Each callback receives a `FrameTick` in
browser host time.

Use `FrameDriver` when an application wants retained frame lifecycle state:
pending demand, queued frame-start plans, stronger-demand preemption,
submission summaries, and dropped-frame summaries. This crate supplies only
browser timing facts. Hosts still decide when a frame is needed, what to render,
and where the rendered output is submitted.

Browser RAF does not expose a portable predicted present timestamp, commit
deadline, or current display refresh interval. `frame_opportunity` creates
pacing-only `FrameOpportunity` values and uses a fallback refresh interval for
display timing. The default fallback is
`DEFAULT_REFRESH_INTERVAL`, a 60 Hz interval in microsecond ticks.

```rust,ignore
use frameclock::{
    FrameBeginResult, FrameDemand, FrameDriver, FrameSubmission, OutputId,
    SchedulerConfig,
};
use frameclock_web::{DEFAULT_REFRESH_INTERVAL, RafLoop};

let mut driver = FrameDriver::new(SchedulerConfig::pacing_only());

let raf = RafLoop::new(
    move |tick| {
        driver.request(FrameDemand::ANIMATION);

        let opportunity = frameclock_web::frame_opportunity(tick, DEFAULT_REFRESH_INTERVAL);
        let begin = driver.begin_frame(opportunity);
        match begin.result {
            FrameBeginResult::Ready(frame) => {
                let sample_time = frame.sample_time();
                // Prepare and submit browser rendering work for sample_time.
                let submit = driver.submit_frame(
                    frame,
                    FrameSubmission::new(frameclock_web::now(), None),
                );
                _ = (sample_time, submit.summary);
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
- `now` and `timebase` for browser host-time conversion.
- `present_hints`, `display_timing`, and `frame_opportunity` for hosts that own
  `FrameDriver` directly.
- `DEFAULT_REFRESH_INTERVAL` for conservative pacing fallback.

`frameclock_web` keeps platform-specific browser code out of `frameclock`
proper. Core scheduling policy, frame demand ordering, frame summaries, and
diagnostics stay in `frameclock`.

## Timing Model

`TIMEBASE` uses microsecond ticks: `1 tick = 1_000 ns`. This matches browser
`DOMHighResTimeStamp` values after converting milliseconds to microseconds.

`RafLoop` emits pacing facts for each delivered RAF callback. It does not assign
content-frame identity; retained hosts get `FramePlan::frame_index` from
`FrameDriver`, while low-level `Scheduler` integrations pass their own frame id
to `Scheduler::plan` and `FrameTickEvent::new`.

Because RAF is pacing-only, `PresentHints::desired_present` is `None` and
`PresentHints::latest_commit` is one fallback refresh interval after the RAF
tick time. Hosts that can get richer browser timing from media APIs, such as
video frame callbacks, should build their own `FrameOpportunity` or use a
future media-specific adapter instead of forcing that data through plain RAF.

## Feature Flags

This crate currently has no feature flags.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
