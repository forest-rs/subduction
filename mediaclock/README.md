<div align="center">

# Mediaclock

**Media timeline clocks and synchronization helpers.**

</div>

`mediaclock` is a small `no_std` crate for mapping `frameclock` host times into
external media timelines. Frame scheduling answers when a visual frame should be
prepared and presented; `mediaclock` answers which media time that planned frame
should represent.

The crate intentionally does not own decoders, audio devices, `<video>`
elements, `AVPlayer`, renderers, event loops, or presentation feedback. Those
belong to platform adapters and host applications.

## Core Flow

```text
frameclock FramePlan sample/target host time
             -> MediaTimeline
             -> media seconds / PTS
             -> host chooses media content
```

Use `MediaTimeline` for ordinary playback timelines. Feed observations from a
media backend with `observe`, call `set_paused` when playback pauses or resumes,
call `reanchor` for known discontinuities such as seek or loop boundaries, and
query `media_time_at` with a frameclock `HostTime`.

Use `AffineClock` directly when a host needs a lower-level smoothed affine
mapping without playback-rate and discontinuity policy.

```rust,ignore
use frameclock::HostTime;
use mediaclock::MediaTimeline;

let mut timeline = MediaTimeline::new(1e-9);

// Observation from a media backend: at host time 1s, media PTS was 1s.
timeline.observe(HostTime(1_000_000_000), 1.0);

// Later, choose media content for a frameclock-planned host time.
let media_time = timeline.media_time_at(HostTime(2_000_000_000));
assert!((media_time.unwrap() - 2.0).abs() < 1e-6);
```

## Relationship To Frameclock

`frameclock` owns display-frame pacing: demand, display timing, deadlines,
feedback, and frame summaries. `mediaclock` owns media-time mapping above that
host-time layer. A video renderer usually asks `frameclock` for the frame's
sample or target-present time, then asks `mediaclock` which media time should be
visible at that host time.

## Minimum Supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license,

at your option.
