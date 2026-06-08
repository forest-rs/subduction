<div align="center">

# Frameclock

**Display-frame timing, scheduling, feedback, and diagnostics.**

</div>

`frameclock` is a small `no_std` crate for turning platform display callbacks
into explicit frame plans. Platform adapters provide timing facts; `frameclock`
decides which time a frame should sample, what deadline it has, and how
scheduler policy should adapt after presentation feedback.

The crate intentionally does not own windows, event loops, layer trees,
renderers, swapchains, or platform presentation resources.

## Core Flow

```text
platform tick -> FrameTick + PresentHints
              -> Scheduler::plan()
              -> FramePlan
              -> build/submit frame
              -> PresentFeedback
              -> Scheduler::observe()
```

Use `FramePlan::sample_time` for animation and simulation sampling. Use
`FramePlan::target_present` only when code specifically needs the predicted
display time.

## Diagnostics

`frameclock` exposes a neutral `DiagnosticsSink` trait and event structs for
ticks, plans, submits, feedback, and scheduler state. Adapter crates can map
those events to Spoor, Tracy, or other instrumentation systems without adding
those dependencies to the core crate.

## Migration Notes

`frameclock` owns the timing pieces that previously lived under
`subduction_core::{clock, scheduler, time, timing}`. `subduction_core` keeps
compatibility re-exports for now, but new code should import these types from
`frameclock` directly.

The split also tightens names around timing semantics:

- `FramePlan::semantic_time` is now `FramePlan::sample_time`.
- `FramePlan::present_time` is now `FramePlan::target_present`.
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
