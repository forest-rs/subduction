<div align="center">

# Frameclock

**Display-frame timing, scheduling, feedback, and diagnostics.**

</div>

`frameclock` is a small `no_std` crate for turning platform display callbacks,
display timing, and frame demand into explicit frame plans. Platform adapters
provide timing facts; hosts provide demand and display constraints;
`frameclock` decides when frame work should begin, which time a frame should
sample, what deadline it has, and how scheduler policy should adapt after
presentation feedback.

The crate intentionally does not own windows, event loops, layer trees,
renderers, swapchains, or platform presentation resources.

## Core Flow

```text
platform tick -> FrameOpportunity
              -> FrameDriver::begin_frame()
              -> FrameBeginResult::Ready(ActiveFrame)
              -> build frame
              -> FrameDriver::submit_frame() or FrameDriver::discard_frame()
              -> FrameTimingSummary
```

Use `FrameDemand` as the host-owned reason a frame is needed. Request
`INPUT` for discrete user action, `CONTINUOUS_INPUT` while scrolling/resizing or
dragging, `ANIMATION` while a visual timeline is active, and `BACKGROUND` for
deferrable visual work. With `FrameDriver`, call `request(demand)` when those
causes arrive; with the low-level scheduler, pass the demand to
`Scheduler::plan(opportunity, demand)`.

Use `DisplayTiming` to describe the current target output's fixed or variable
refresh constraints. It should come from the backend/platform facts for the
same output or surface as the `FrameTick`; refresh it when a window moves
between displays or the platform reports a changed display mode.

Use `FramePlan::frame_start` to arm an event-loop wake or request redraw when
the start time is already due. Use `FramePlan::sample_time` for animation and
simulation sampling. Use `FramePlan::target_present` only when code
specifically needs the predicted display time.

`FrameDemand::NONE` means there is no app-visible frame work to schedule.
Ordinary render loops should stay idle instead of calling `Scheduler::plan`
with empty demand. Passing `NONE` is reserved for hosts that intentionally want
a passive pacing plan for diagnostics or backend bookkeeping.
`FrameDemandClass` is the derived ordering/classification used by
`FrameDemand::dominant_class` and `FrameDemand::preempts`; it is mainly for
diagnostics and adapter policy that needs to match frameclock's demand order.

`FrameDriver` is the retained lifecycle helper for hosts that need to queue
demand and a future frame-start plan between event-loop turns. It owns pending
demand, stronger-demand preemption, queued `PlannedFrame`s, feedback
observation, and `FrameTimingSummary` construction. Hosts still own timers,
redraw requests, renderer submission, and native presentation resources. Use
`FrameDriver::next_frame_start` as one wake source to merge with app timers.
After submitting or discarding an `ActiveFrame`, hosts should request another
redraw when `FrameDriver::has_pending_demand()` is still true.
`FrameTick::frame_index` is host-owned per output and identifies one planned
content frame. Hosts using `FrameDriver` normally increment it after an
`ActiveFrame` is submitted or discarded, not every time a frame-start wake
fires while a plan is queued.

The lower-level `Scheduler` remains available for custom integrations. Event
structs and `FrameTimingSummaryBuilder` live under `frameclock::diagnostics`
for telemetry adapters and tests, but normal host code should not need to
assemble summary events by hand.

## API Surfaces

The root module re-exports the common retained-host integration surface:

- `FrameDriver`, `FrameOpportunity`, `ActiveFrame`, and `FrameSubmission`
- `FrameDemand`
- `SchedulerConfig`
- `FrameTick`, `PresentHints`, and `DisplayTiming`
- `HostTime`, `Duration`, and `OutputId`
- `FrameTimingSummary`

Specialized APIs live under their modules:

- `frameclock::diagnostics` for event structs, `DiagnosticsSink`, and
  `FrameTimingSummaryBuilder`.
- `frameclock::scheduler` for `Scheduler`, `DegradationPolicy`, and adaptation
  state used by custom low-level integrations.
- `frameclock::timing` for presentation feedback, pending feedback, frame
  plans, and lower-level timing details.
- `frameclock::time` for `Timebase` and clock conversion.
- `frameclock::timeline` for `AffineClock`.
- `frameclock::driver` and `frameclock::demand` for lower-level lifecycle and
  demand policy types.

```rust,ignore
use frameclock::{
    Duration, FrameBeginResult, FrameDemand, FrameDriver, FrameOpportunity,
    FrameSubmission, HostTime, OutputId, SchedulerConfig,
};

let mut driver = FrameDriver::new(SchedulerConfig::pacing_only());
driver.request(FrameDemand::ANIMATION);

let opportunity = FrameOpportunity::pacing_only(
    HostTime(1_000_000),
    Duration(16_666_667),
    1,
    OutputId(0),
);

match driver.begin_frame(opportunity) {
    FrameBeginResult::Ready(frame) => {
        let sample_time = frame.sample_time();
        // Prepare app/model/render state for sample_time, then submit renderer
        // work. If the frame cannot be submitted, call `discard_frame` instead.
        let summary = driver.submit_frame(
            frame,
            FrameSubmission::new(HostTime(2_000_000), None),
        );
        _ = (sample_time, summary);
    }
    FrameBeginResult::WaitUntil(frame_start) => {
        // Mirror frame_start into the host timer queue.
        _ = frame_start;
    }
    FrameBeginResult::Expired(summary) => {
        // Record the dropped-frame summary and request fresh demand if needed.
        _ = summary;
    }
    FrameBeginResult::Idle => {}
}
```

## Display Timing And VRR

`DisplayTiming::fixed(interval)` is the right model when a backend has only a
current refresh interval, or when the platform does not expose direct control
over variable presentation timing.

Display timing is not app-global state. It belongs to the current target output
for a frame opportunity. If a window is dragged from a 60 Hz display to a 120 Hz
display, or if a platform display link reports a new interval/range, the next
`FrameOpportunity` should carry timing for that new output.

`DisplayTiming::variable(min_interval, max_interval, granularity)` describes a
display range. The `granularity` argument is intentionally conservative:

- `Some(step)` means the backend knows direct display intervals can be selected
  at that step.
- `None` means the backend knows a VRR range exists, but does not know the
  direct interval granularity or cannot request arbitrary direct presentation
  durations.

When granularity is unknown, `frameclock` chooses stable multiples of
`min_interval`, like fixed-rate pacing, instead of inventing arbitrary intervals
inside the VRR range. Platform adapters should pass an explicit granularity only
when the presentation API can honestly honor it.

## Diagnostics

`frameclock` exposes a neutral `DiagnosticsSink` trait and event structs for
ticks, plans, submits, feedback, scheduler state, and compact per-frame timing
summaries. Adapter crates can map those events to Spoor, Tracy, or other
instrumentation systems without adding those dependencies to the core crate.
`FrameTimingSummary::timing_basis` classifies whether each summary is based on
actual present feedback, predicted present timing, submission timing, or
pacing-only timing.

## Migration Notes

`frameclock` owns the timing pieces that previously lived under
`subduction_core::{clock, scheduler, time, timing}`. `subduction_core` keeps
compatibility re-exports for now, but new code should import these types from
`frameclock` directly.

The split also tightens names around timing semantics:

- `FramePlan::semantic_time` is now `FramePlan::sample_time`.
- `FramePlan::present_time` is now `FramePlan::target_present`.
- `FramePlan::frame_start` is now the scheduler-selected time to wake or start
  app-side frame work before `FramePlan::commit_deadline`.
- `Scheduler::plan` now takes a `FrameOpportunity` plus `FrameDemand` so
  display timing facts and demand remain explicit policy inputs.
- `FrameDemand::dominant_class` and `FrameDemand::preempts` expose the demand
  ordering used by the scheduler.
- `FrameDriver` owns pending demand and queued frame-start plans for hosts that
  need retained frame scheduling state. `PlannedFrame` retains the originating
  `FrameTick`, selected `FramePlan`, and matching `PresentHints`.
- Platform-named scheduler presets are now capability-named:
  `SchedulerConfig::predictive()`, `SchedulerConfig::estimated()`, and
  `SchedulerConfig::pacing_only()`.

## Feature Flags

- `std`: reserved for future standard-library integrations. The current API is
  `no_std`.

## Minimum Supported Rust Version

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
