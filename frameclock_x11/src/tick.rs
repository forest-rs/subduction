// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Present-driven frame-tick queueing and ticker state machine.

use crate::queue::BoundedQueue;
use crate::time::{Clock, ust_to_host_time};
use frameclock::{FrameTick, HostTime, OutputId};

const MICROS_TO_NANOS: u64 = 1_000;

/// Internal bounded queue for frame ticks.
///
/// Overflow policy is `drop_oldest` to retain the freshest pacing signal.
#[derive(Debug, Clone)]
pub(crate) struct TickQueue {
    inner: BoundedQueue<FrameTick>,
}

impl TickQueue {
    pub(crate) const DEFAULT_CAPACITY: usize = 8;

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: BoundedQueue::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, tick: FrameTick) {
        self.inner.push(tick);
    }

    pub(crate) fn pop(&mut self) -> Option<FrameTick> {
        self.inner.pop()
    }

    #[cfg(test)]
    pub(crate) fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }
}

impl Default for TickQueue {
    fn default() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }
}

/// One observed Present completion: the vblank timestamp and its counter.
#[derive(Clone, Copy, Debug)]
struct PresentSample {
    /// Unadjusted system time of the vblank, in microseconds.
    ust_micros: u64,
    /// Media stream counter (vblank count) of the vblank.
    msc: u64,
}

/// Pure-logic state machine that turns Present completions into [`FrameTick`]s.
///
/// The Present extension answers a `PresentNotifyMSC` request with a
/// `PresentCompleteNotify` event carrying the vblank's `ust` (unadjusted
/// system time, microseconds) and `msc` (media stream counter). One such event
/// is both the frame-start wake and the actual-present timestamp, so each
/// completion emits a fully-populated tick.
///
/// Protocol I/O is handled externally; this type contains only the bookkeeping.
/// Hosts call [`mark_notify_requested`](Self::mark_notify_requested) before
/// sending a `PresentNotifyMSC` request, [`on_complete_notify`](Self::on_complete_notify)
/// when the matching `PresentCompleteNotify` event arrives, and drain ticks with
/// [`poll_tick`](Self::poll_tick).
///
/// # One stream per window
///
/// A `PresentTicker` models a single window's vblank stream. Create one instance
/// per presented window, drive it only with that window's Present completions,
/// and pass a stable [`OutputId`] for the stream via
/// [`with_output`](Self::with_output).
///
/// # Refresh interval
///
/// The refresh interval is derived from consecutive completions:
/// `(ust_now - ust_prev) / (msc_now - msc_prev)`, converted to nanoseconds. It
/// is `None` until a second completion establishes a delta, leaving early ticks
/// pacing-only. Because `msc` counts *every* vblank whether or not a frame was
/// presented into it, this yields the true display cadence even when the app
/// renders below the refresh rate.
#[derive(Debug)]
pub struct PresentTicker {
    queue: TickQueue,
    notify_in_flight: bool,
    last_present: Option<PresentSample>,
    refresh_interval: Option<u64>,
    output: OutputId,
}

impl PresentTicker {
    /// Creates an empty ticker for [`OutputId`] `0` with no notify in flight.
    #[must_use]
    pub fn new() -> Self {
        Self::with_output(OutputId(0))
    }

    /// Creates an empty ticker whose ticks carry `output`.
    #[must_use]
    pub fn with_output(output: OutputId) -> Self {
        Self {
            queue: TickQueue::default(),
            notify_in_flight: false,
            last_present: None,
            refresh_interval: None,
            output,
        }
    }

    /// Returns the [`OutputId`] stamped on emitted ticks.
    #[must_use]
    pub fn output(&self) -> OutputId {
        self.output
    }

    /// Sets the [`OutputId`] stamped on future ticks.
    ///
    /// Update this only when the window actually moves between outputs; the
    /// observed refresh interval is intentionally retained across the change,
    /// since a stale cadence is a better predictor than none until the new
    /// output reports its own.
    pub fn set_output(&mut self, output: OutputId) {
        self.output = output;
    }

    /// Records that a `PresentCompleteNotify` event has arrived.
    ///
    /// If a notify is in flight, builds a [`FrameTick`] from the current time
    /// read from `clock`, the vblank's `ust_micros`/`msc`, and the observed
    /// refresh interval, enqueues it, and clears the in-flight flag. If no notify
    /// is in flight, debug-asserts and returns.
    ///
    /// `ust_micros` and `msc` are the `ust` and `msc` fields of the
    /// `PresentCompleteNotify` event. Only completions of kind `NotifyMSC` (the
    /// answer to a `PresentNotifyMSC` request) should be fed here; per-pixmap
    /// completions belong to a presentation-feedback path, not the tick stream.
    pub fn on_complete_notify(&mut self, clock: Clock, ust_micros: u64, msc: u64) {
        debug_assert!(
            self.notify_in_flight,
            "on_complete_notify called without an in-flight notify"
        );
        if !self.notify_in_flight {
            return;
        }

        if let Some(prev) = self.last_present
            && let Some(interval) = refresh_interval_from_samples(prev, ust_micros, msc)
        {
            self.refresh_interval = Some(interval);
        }

        let actual_present = ust_to_host_time(ust_micros);
        let now = clock.now();
        let predicted_present = predict_next_present(actual_present, self.refresh_interval, now);

        let tick = FrameTick {
            now,
            predicted_present,
            refresh_interval: self.refresh_interval,
            output: self.output,
            prev_actual_present: Some(actual_present),
        };

        self.queue.push(tick);
        self.last_present = Some(PresentSample { ust_micros, msc });
        self.notify_in_flight = false;
    }

    /// Pops the next queued [`FrameTick`], if any.
    pub fn poll_tick(&mut self) -> Option<FrameTick> {
        self.queue.pop()
    }

    /// Returns whether a `PresentNotifyMSC` notify is currently in flight.
    #[must_use]
    pub fn is_notify_in_flight(&self) -> bool {
        self.notify_in_flight
    }

    /// Claims the single in-flight notify slot before a `PresentNotifyMSC`
    /// request is sent.
    ///
    /// Only one notify may be in flight at a time. Returns `true` when the slot
    /// was newly claimed and the caller should send the `PresentNotifyMSC`
    /// request. Returns `false` when a notify is already in flight; in that case
    /// the ticker state is left unchanged and the caller must not arm another.
    /// The slot is released when the matching
    /// [`on_complete_notify`](Self::on_complete_notify) runs.
    #[must_use = "a false return means a notify is already in flight and no new request should be armed"]
    pub fn mark_notify_requested(&mut self) -> bool {
        if self.notify_in_flight {
            return false;
        }
        self.notify_in_flight = true;
        true
    }

    /// Returns the most recently observed refresh interval in host ticks, if a
    /// second completion has established the display cadence.
    #[must_use]
    pub fn observed_refresh_interval(&self) -> Option<u64> {
        self.refresh_interval
    }
}

impl Default for PresentTicker {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the per-vblank refresh interval (host ticks) from two completions.
///
/// Returns `None` when the counter did not advance (a duplicate or reordered
/// completion) or the resulting interval would be zero, so a bad sample never
/// overwrites a good cadence.
fn refresh_interval_from_samples(prev: PresentSample, ust_micros: u64, msc: u64) -> Option<u64> {
    let msc_delta = msc.checked_sub(prev.msc).filter(|delta| *delta > 0)?;
    let ust_delta_micros = ust_micros.checked_sub(prev.ust_micros)?;
    let per_frame = ust_delta_micros.saturating_mul(MICROS_TO_NANOS) / msc_delta;
    (per_frame > 0).then_some(per_frame)
}

/// Predicts the next vblank at or after `now` from the last observed present.
///
/// Returns `None` when no refresh interval has been established, leaving the
/// tick pacing-only. Otherwise it advances the last actual-present time by whole
/// refresh intervals until it reaches `now`, landing on the display's vblank
/// grid.
fn predict_next_present(
    last_actual: HostTime,
    refresh_interval: Option<u64>,
    now: HostTime,
) -> Option<HostTime> {
    let refresh = refresh_interval.filter(|interval| *interval > 0)?;
    let elapsed = now.ticks().saturating_sub(last_actual.ticks());
    let intervals = elapsed.div_ceil(refresh);
    let advance = intervals.checked_mul(refresh)?;
    last_actual.ticks().checked_add(advance).map(HostTime)
}

#[cfg(test)]
mod tests {
    use super::{PresentTicker, TickQueue, predict_next_present};
    use crate::time::{Clock, ust_to_host_time};
    use frameclock::{HostTime, OutputId};

    #[test]
    fn poll_tick_returns_none_when_empty() {
        let mut ticker = PresentTicker::new();
        assert!(ticker.poll_tick().is_none());
    }

    #[test]
    fn first_completion_is_pacing_only_with_actual_present() {
        let mut ticker = PresentTicker::new();

        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 1_000_000, 100);

        let tick = ticker.poll_tick().expect("should have a tick");
        // No prior sample, so no cadence and no prediction yet.
        assert_eq!(tick.refresh_interval, None);
        assert_eq!(tick.predicted_present, None);
        // The vblank's ust (microseconds) becomes the actual-present ticks.
        assert_eq!(tick.prev_actual_present, Some(ust_to_host_time(1_000_000)));
        assert_eq!(tick.output, OutputId(0));
    }

    #[test]
    fn second_completion_establishes_refresh_and_prediction() {
        let mut ticker = PresentTicker::new();

        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 1_000_000, 100);
        let _ = ticker.poll_tick();

        // One vblank later: +16_667 µs, +1 msc -> ~16.667 ms refresh.
        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 1_016_667, 101);

        let tick = ticker.poll_tick().expect("should have a tick");
        assert_eq!(tick.refresh_interval, Some(16_667_000));
        assert_eq!(ticker.observed_refresh_interval(), Some(16_667_000));
        // Prediction lands on the vblank grid at or after `now`.
        let predicted = tick.predicted_present.expect("prediction available");
        assert!(predicted >= tick.now);
    }

    #[test]
    fn refresh_is_per_vblank_even_when_frames_are_skipped() {
        let mut ticker = PresentTicker::new();

        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 2_000_000, 200);
        let _ = ticker.poll_tick();

        // Three vblanks elapsed in one completion: +50_000 µs over +3 msc.
        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 2_050_000, 203);
        let tick = ticker.poll_tick().expect("should have a tick");

        // 50_000 µs / 3 vblanks = 16_666_666 ns per vblank.
        assert_eq!(tick.refresh_interval, Some(16_666_666));
    }

    #[test]
    fn duplicate_msc_does_not_overwrite_cadence() {
        let mut ticker = PresentTicker::new();

        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 3_000_000, 300);
        let _ = ticker.poll_tick();
        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 3_016_667, 301);
        let _ = ticker.poll_tick();
        assert_eq!(ticker.observed_refresh_interval(), Some(16_667_000));

        // A completion with the same msc yields no delta and is ignored.
        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 3_016_667, 301);
        let _ = ticker.poll_tick();
        assert_eq!(ticker.observed_refresh_interval(), Some(16_667_000));
    }

    #[test]
    fn notify_in_flight_transitions() {
        let mut ticker = PresentTicker::new();

        assert!(!ticker.is_notify_in_flight());
        assert!(ticker.mark_notify_requested());
        assert!(ticker.is_notify_in_flight());
        ticker.on_complete_notify(Clock::Monotonic, 1_000_000, 1);
        assert!(!ticker.is_notify_in_flight());
    }

    #[test]
    fn mark_notify_requested_rejects_double_request() {
        let mut ticker = PresentTicker::new();

        assert!(ticker.mark_notify_requested());
        assert!(!ticker.mark_notify_requested());
        assert!(ticker.is_notify_in_flight());

        ticker.on_complete_notify(Clock::Monotonic, 1_000_000, 1);
        assert!(ticker.mark_notify_requested());
    }

    #[test]
    fn output_is_stamped_on_ticks() {
        let mut ticker = PresentTicker::with_output(OutputId(4));
        assert_eq!(ticker.output(), OutputId(4));

        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 1_000_000, 1);
        let tick = ticker.poll_tick().unwrap();
        assert_eq!(tick.output, OutputId(4));

        ticker.set_output(OutputId(7));
        assert!(ticker.mark_notify_requested());
        ticker.on_complete_notify(Clock::Monotonic, 1_016_667, 2);
        let tick = ticker.poll_tick().unwrap();
        assert_eq!(tick.output, OutputId(7));
    }

    #[test]
    fn queue_overflow_drops_oldest_through_ticker() {
        let mut ticker = PresentTicker::new();

        // Default capacity is 8; push 9 ticks without polling and verify one drop.
        for frame in 0..9 {
            assert!(ticker.mark_notify_requested());
            ticker.on_complete_notify(Clock::Monotonic, 1_000_000 + frame * 16_667, 100 + frame);
        }

        assert_eq!(ticker.queue.dropped_count(), 1);
    }

    // --- Present prediction tests ---

    #[test]
    fn predict_next_present_rounds_up_to_next_vblank() {
        // last + 4*refresh = 1640 is the first vblank at or after now = 1500.
        let predicted = predict_next_present(HostTime(1000), Some(160), HostTime(1500));
        assert_eq!(predicted, Some(HostTime(1640)));
    }

    #[test]
    fn predict_next_present_on_exact_vblank_returns_now() {
        let predicted = predict_next_present(HostTime(1000), Some(160), HostTime(1480));
        assert_eq!(predicted, Some(HostTime(1480)));
    }

    #[test]
    fn predict_next_present_without_refresh_is_none() {
        assert_eq!(
            predict_next_present(HostTime(1000), None, HostTime(1500)),
            None
        );
        assert_eq!(
            predict_next_present(HostTime(1000), Some(0), HostTime(1500)),
            None
        );
    }

    #[test]
    fn tick_queue_overflow_drops_oldest() {
        let mut queue = TickQueue::with_capacity(2);
        for now in 1..=3 {
            queue.push(frameclock::FrameTick {
                now: HostTime(now),
                predicted_present: None,
                refresh_interval: None,
                output: OutputId(0),
                prev_actual_present: None,
            });
        }
        assert_eq!(queue.pop().map(|tick| tick.now), Some(HostTime(2)));
        assert_eq!(queue.pop().map(|tick| tick.now), Some(HostTime(3)));
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.dropped_count(), 1);
    }
}
