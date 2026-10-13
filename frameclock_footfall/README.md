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
carry the frame's number, so they find their frame whenever they arrive.

## What Is Recorded

| `FrameTimingSummary` | Footfall |
| --- | --- |
| `frame_index` | the number on the frame markers and every frame point |
| `commit_deadline` | `FramePointKind::Deadline` |
| `target_present` | `FramePointKind::Target` |
| `actual_present` | `FramePointKind::Presented` |
| `drop_reason`, except `Discarded` and `Superseded` | `FramePoint::dropped`, and a `frameclock_dropped_<reason>` counter |
| `drop_reason` `Discarded` or `Superseded` | the counter only: the host chose not to show the frame |
| `pipeline_depth` | `frameclock_pipeline_depth` gauge |
| `safety_margin_ticks` | `frameclock_safety_margin` gauge, in nanoseconds |
| `frame_interval` | `frameclock_frame_interval` gauge, in nanoseconds |
| `scheduler_state.present_latency_ticks` | `frameclock_present_latency` gauge, in nanoseconds |
| `work` | `frameclock_submission_work` histogram, in nanoseconds |

Frame points stay in host ticks under `HOST_CLOCK`, with the platform
adapter's timebase as the tick period. Footfall's analysis converts them
through the clock syncs `Recorder::sync` records, and `Trace::pacing` judges
each frame on time, late, dropped, or submitted late from them.

## Footfall Version

Footfall isn't on crates.io yet. Until it is, the workspace pins it by git
revision.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
