# ADR-0001: Frame Timing in Footfall

## Status

Accepted.

## Context

`frameclock` knows, for each frame, its commit deadline, its target present
time, when it was shown or that it was dropped, and how the scheduler adapted.
Footfall records CPU work as frame markers and scopes. Its analysis judges
frame pacing from numbered frames and frame points recorded in another clock.
This crate connects the two.

Much of what frameclock knows arrives late: a GPU renderer learns when a frame
was shown frames after submitting it, through `FrameDriver::report_frame` or
`FrameBegin::resolved_feedback`.

## Decision

- **The summary is the one source of frame points.** `Recorder::record` maps a
  `FrameTimingSummary` and nothing else, whenever it arrives. Frame points
  carry the frame number, so arriving late doesn't matter. The plan's deadline
  and target are known at begin, but recording them then as well would only
  duplicate the summary.
- **The host delimits the frame's work.** A summary has no build start, and
  arrives after submission. So `Recorder::begin_frame` and
  `Recorder::end_frame` emit Footfall's numbered frame markers from the
  `ActiveFrame`, called by the host around its own work. A frame whose work
  ends after its deadline then shows as submitted late when no presentation
  time is known, which is what frameclock calls a pacing overrun.
- **Times stay in host ticks.** Frame points are readings of `HOST_CLOCK`, with
  the platform adapter's `Timebase` as Footfall's `TickPeriod`. The host calls
  `Recorder::sync` right after reading its clock. Footfall's analysis does the
  converting, and it reports any points it could not place.
- **Every time frameclock knows is a point.** The platform's prediction at
  frame start (`predicted_present`), the commit deadline, the target, the
  present time expected at submit (`expected_present`), and the actual present
  time each become a frame point, so Footfall can judge every prediction
  against what happened.
- **Every drop carries its reason.** Each `FrameDropReason` becomes
  `FramePoint::dropped` with the matching Footfall `DropReason`;
  `NotPresented` is `Replaced`. Frames the host discarded or superseded itself
  are recorded too. Deciding which drops are misses is the analysis's job, not
  the adapter's: Footfall counts only some reasons as misses, and counts the
  work on every unshown frame as wasted.
- **Scheduler state goes in counters.** Pipeline depth, safety margin, frame
  interval and learned present latency are gauges. Submission work, such as
  GPU time, is a histogram. Durations are converted to nanoseconds. Counter
  descriptors are private, and their names are documented.
- **No state.** The recorder holds only the timebase and a context handle, so
  it can live in a `static` and be shared.

## Consequences

- Frameclock's timing basis and presentation timing aren't recorded. Analysis
  infers the basis from which points exist.
- Gauges are stamped when the summary arrives, not when the frame ran.
- Footfall is pinned by git revision until it is published. The crate depends
  only on `footfall_core`, which is `no_std` and has no dependencies.
