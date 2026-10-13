<div align="center">

# Frameclock Footfall

**Records `frameclock`'s frame timing in Footfall traces.**

</div>

`frameclock_footfall` turns what a `FrameDriver` learns about each frame into
[Footfall](https://github.com/forest-rs/footfall) observations. A trace can
then say which frames missed the display, by how much, and what their work did
differently.

The crate is `no_std` and holds no state beyond the platform's timebase.
Without an installed Footfall backend every call does nothing.

## Core Flow

```text
platform tick       -> Recorder::sync(now)
FrameBeginResult::Ready(frame)
                    -> Recorder::begin_frame(&frame)
                    -> build frame
                    -> Recorder::end_frame(&frame)
                    -> FrameDriver::submit_frame() / report_frame() / ...
FrameTimingSummary  -> Recorder::record(&summary)
```

Pass every summary the driver returns, including the ones that arrive frames
later from `report_frame` or `FrameBegin::resolved_feedback`. Frame points
carry the frame's number, so they find their frame whenever they arrive. The
summary is the only source of frame points: the plan's deadline and target are
known at begin, but recording them then would only duplicate it.

## What Is Recorded

| `FrameTimingSummary` | Footfall |
| --- | --- |
| `frame_index` | the number on the frame markers and every frame point, in the recorder's `FrameStream` |
| `predicted_present` | `FramePointKind::Predicted`, with a `presentation_timing` arg |
| `commit_deadline` | `FramePointKind::Deadline`, with the plan's scheduler state as args |
| `target_present` | `FramePointKind::Target` |
| `sample_time` | `FramePointKind::Sampled` |
| `submitted_at` | `FramePointKind::Submitted`, observed, with a `work_ns` arg when reported |
| `expected_present` | `FramePointKind::Expected` |
| `actual_present` | `FramePointKind::Presented`, observed for `ActualPresent` and estimated otherwise |
| `drop_reason` | `FramePointKind::Dropped`, with the matching `DropReason` |
| `pipeline_depth`, `safety_margin_ticks`, `frame_interval`, `scheduler_state.present_latency_ticks` | args on the deadline point: `pipeline_depth`, `safety_margin_ns`, `frame_interval_ns`, `present_latency_ns` |
| `work` | also the `frameclock_submission_work` histogram, in nanoseconds |

The scheduler state rides on the frame's own deadline point rather than in
gauges set when the summary arrives, which may be frames later, so it
describes the frame it was planned for. `FramePacing::args` returns it.

A program with several frame drivers, such as one per window, gives each
recorder its own stream with `Recorder::with_stream`, and a recreated driver,
whose frame indices start again, a new generation of it.

Every drop is recorded with its reason, including frames the host discarded
or superseded itself. Footfall's analysis keeps those out of its miss counts,
while still counting the work they wasted. `NotPresented` stays
`DropReason::NotPresented`, a miss: frameclock does not say whether a later
frame replaced the frame or it was lost.

Frame points stay in host ticks under `HOST_CLOCK`, with the platform
adapter's timebase as the tick period. Footfall's analysis converts them
through the clock syncs `Recorder::sync` records, and `Trace::pacing` judges
each frame on time, late, dropped, skipped, or submitted late from them.
`Pacing::prediction_error` compares the predicted, target, and expected times
with when frames were shown.

## Footfall Version

This crate uses Footfall 0.1.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
