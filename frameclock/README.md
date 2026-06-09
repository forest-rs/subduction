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
platform tick -> FrameTick + PresentHints
              -> FrameRequest + FrameDemand + DisplayTiming
              -> Scheduler::plan()
              -> FramePlan
              -> build/submit frame
              -> PresentFeedback
              -> Scheduler::observe()
```

Use `FrameDemand` to distinguish latency-sensitive input from continuous input,
animation, and deferrable background work. Use `DisplayTiming` to describe the
target output's fixed or variable refresh constraints. Use
`FramePlan::frame_start` to arm an event-loop wake or request redraw when the
start time is already due. Use `FramePlan::sample_time` for animation and
simulation sampling. Use `FramePlan::target_present` only when code
specifically needs the predicted display time.

`FrameDemand::NONE` means there is no app-visible frame work to schedule.
Ordinary render loops should stay idle instead of calling `Scheduler::plan`
with empty demand. Passing `NONE` is reserved for hosts that intentionally want
a passive pacing plan for diagnostics or backend bookkeeping.

`FrameDriver` is the retained helper for hosts that need to queue demand and a
future frame-start plan between event-loop turns. It owns pending demand,
stronger-demand preemption, and the queued `PlannedFrame`; hosts still own
timers, redraw requests, and renderer submission. Use
`FrameDriver::next_frame_start` as one wake source to merge with app timers.
Returned `PlannedFrame`s retain the originating `FrameTick`, the selected
`FramePlan`, and matching `PresentHints` so diagnostics and frame summaries can
refer to the same facts the scheduler used. After consuming a ready
`PlannedFrame`, hosts should request another redraw when
`FrameDriver::has_pending_demand()` is still true.

## Display Timing And VRR

`DisplayTiming::fixed(interval)` is the right model when a backend has only a
current refresh interval, or when the platform does not expose direct control
over variable presentation timing.

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
- `Scheduler::plan` now takes a `FrameRequest` so demand and display timing are
  explicit policy inputs.
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
