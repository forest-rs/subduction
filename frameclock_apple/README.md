<div align="center">

# Frameclock Apple

**Apple display-link timing adapters for `frameclock`.**

</div>

`frameclock_apple` connects Apple display-link callbacks to `frameclock`. It
converts `CADisplayLink` and `CVDisplayLink` timing into `FrameTick` values,
exposes Mach absolute time as `HostTime`, computes predictive present hints,
and offers `AppleFrameClock` as a retained wrapper around `FrameDriver`.

The crate intentionally does not own `CALayer` trees, `CAMetalLayer`
presentation, renderers, windows, app lifecycle, or event-loop policy.

## Core Flow

```text
CADisplayLink / CVDisplayLink -> FrameTick
                              -> AppleFrameClock::begin_frame()
                              -> FrameBegin { result: FrameBeginResult::Ready(ActiveFrame), ... }
                              -> host render
                              -> AppleFrameClock::submit_frame() or AppleFrameClock::discard_frame()
                              -> FrameTimingSummary
```

Use `DisplayLink` when an application wants this crate to create an Apple
display-link tick source. The default `ca-display-link` feature exposes a
main-thread `CADisplayLink` wrapper. The `cv-display-link` feature exposes the
legacy `CVDisplayLink` wrapper and forwarding types for sending ticks to a
single-threaded host scheduler.

Use `AppleFrameClock` when an application wants retained frame lifecycle state:
pending demand, queued frame-start plans, stronger-demand preemption,
submission summaries, and dropped-frame summaries. Hosts still decide when a
frame is needed, what to render, when to acquire native presentation resources,
and how to submit to Core Animation or Metal.

```rust,ignore
use frameclock::{
    DisplayTiming, Duration, FrameBeginResult, FrameDemand, OutputId, SchedulerConfig,
};
use frameclock_apple::{AppleFrameClock, DisplayLink};
use objc2::MainThreadMarker;

let mut clock = AppleFrameClock::new(
    SchedulerConfig::predictive(),
    DisplayTiming::fixed(Duration(16_666_667)),
);
let output = OutputId(0);
let mtm = MainThreadMarker::new().unwrap();

let display_link = DisplayLink::new(
    move |tick| {
        clock.request(FrameDemand::ANIMATION);

        let begin = clock.begin_frame(tick);
        if let Some(summary) = begin.resolved_feedback {
            // Previous deferred submit resolved with this tick's actual-present fact.
            _ = summary;
        }

        match begin.result {
            FrameBeginResult::Ready(frame) => {
                let sample_time = frame.sample_time();
                // Prepare and submit Apple rendering work for sample_time.
                let submit = clock.submit_frame_now(frame);
                _ = (sample_time, submit.awaiting_actual_present);
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
    output,
    mtm,
);

display_link.start();
```

## API Surfaces

The root module exposes the Apple integration surface:

- `DisplayLink` for the enabled Apple display-link implementation.
- `AppleFrameClock` for retained `FrameDriver` integration.
- `now` and `timebase` for Mach host-time conversion.
- `present_hints`, `compute_present_hints`, and `display_timing` for hosts that
  need lower-level timing facts.
- `preferred_frame_rate_range` and `PreferredFrameRateRange` for translating a
  selected frame interval into a Core Animation-style ProMotion cadence request.
- `TickForwarder`, `TickSender`, and `DisplayLinkError` when the
  `cv-display-link` feature is enabled without `ca-display-link`.

`frameclock_apple` keeps Apple FFI and thread-model details out of
`frameclock` proper. Core scheduling policy, frame demand ordering, frame
summaries, and diagnostics stay in `frameclock`.

## Timing Model

`now` and emitted `FrameTick` values use Mach absolute time ticks. Use
`timebase` when a host needs to convert those ticks to nanoseconds for logs,
tracing, or external diagnostics.

`CADisplayLink` ticks carry `targetTimestamp` as `predicted_present`,
`duration` as `refresh_interval`, and the previous callback's `timestamp` as
`prev_actual_present` after the first tick. `CVDisplayLink` ticks carry the
output host time as `predicted_present`.

`AppleFrameClock` computes predictive `PresentHints` from the display-link
prediction when it is fresh. If a display-link callback arrives after its
predicted present time, the stale prediction is ignored and the frame is planned
with pacing-only hints from the callback time. Scheduler safety margin remains
inside `frameclock` planning; it is not baked into Apple platform hints.

`CADisplayLink.timestamp` reports the previous callback's actual display time.
`AppleFrameClock::submit_frame_now` therefore records a deferred submission;
the next `begin_frame` returns the completed `FrameTimingSummary` in
`FrameBegin::resolved_feedback` when the tick carries `prev_actual_present`.
Hosts with immediate or unavailable feedback can still call
`AppleFrameClock::submit_frame` with an explicit `FrameSubmission`.

Display timing belongs to the output that produced the tick. Hosts should
refresh output identity and target-output `DisplayTiming` when a window moves
between displays or the platform reports a different display mode. For
ProMotion/VRR displays, pass `DisplayTiming::variable` for the current output
and use `preferred_frame_rate_range` or
`DisplayLink::set_preferred_frame_interval` to translate a planned
`FramePlan::frame_interval` into a Core Animation frame-rate range.

## Feature Flags

- `ca-display-link`: enables `CADisplayLink` support and is enabled by
  default.
- `cv-display-link`: enables legacy `CVDisplayLink` support. This feature is
  intended for hosts that need a Core Video display link and a forwarding path
  back to a single-threaded scheduler.

This crate keeps its own implementation `no_std`, but the selected Objective-C
framework bindings currently require `std`. It is validated on supported Apple
targets instead of the workspace's generic `x86_64-unknown-none` no-std target.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
