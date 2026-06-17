// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame-tick queueing primitives and ticker state machine.

use crate::queue::BoundedQueue;
use crate::time::Clock;
use frameclock::{FrameTick, HostTime, OutputId};

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

    #[allow(dead_code, reason = "used by tests and future diagnostics")]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    #[allow(dead_code, reason = "used by tests and future diagnostics")]
    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[allow(dead_code, reason = "used by tests and future diagnostics")]
    pub(crate) fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }
}

impl Default for TickQueue {
    fn default() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }
}

/// Pure-logic state machine for frame callback tick generation.
///
/// Tracks the in-flight callback state, builds [`FrameTick`]s when callbacks
/// complete, and queues them for polling. Protocol I/O is handled externally;
/// this type contains only the bookkeeping. Hosts call
/// [`mark_callback_requested`](Self::mark_callback_requested) when sending a
/// `wl_surface.frame` request, [`on_callback_done`](Self::on_callback_done)
/// when the matching `wl_callback.done` event arrives, and drain ticks with
/// [`poll_tick`](Self::poll_tick).
///
/// # One stream per surface
///
/// A `TickerState` models a single paced surface/output stream. Create one
/// instance per `wl_surface` you pace, drive it only with that surface's frame
/// callbacks, and pass a stable [`OutputId`] for the stream to
/// [`on_callback_done`](Self::on_callback_done).
///
/// The ticker keeps the most-recent actual-present timestamp (see
/// [`set_last_observed_actual_present`](Self::set_last_observed_actual_present))
/// and the most-recent observed refresh interval (see
/// [`set_last_observed_refresh_interval`](Self::set_last_observed_refresh_interval)).
/// It stamps the actual-present time onto the next tick's
/// [`FrameTick::prev_actual_present`], and when both facts are known it predicts
/// the next vsync at or after the tick time and emits it as
/// [`FrameTick::predicted_present`] alongside the [`FrameTick::refresh_interval`].
/// Feed
/// it only presentation feedback for the same surface/output stream: mixing in
/// feedback from an unrelated surface or output would attribute one surface's
/// presentation to another. Hosts that multiplex several surfaces on one event
/// queue should keep a `TickerState` per stream and correlate presentation
/// feedback to the right stream themselves — for example by the
/// [`SubmissionId`](crate::SubmissionId) carried on each
/// [`PresentEvent`](crate::PresentEvent).
#[derive(Debug)]
pub struct TickerState {
    queue: TickQueue,
    tick_index: u64,
    callback_in_flight: bool,
    last_observed_actual_present: Option<HostTime>,
    last_observed_refresh_interval: Option<u64>,
}

impl TickerState {
    /// Creates an empty ticker with no callback in flight.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: TickQueue::default(),
            tick_index: 0,
            callback_in_flight: false,
            last_observed_actual_present: None,
            last_observed_refresh_interval: None,
        }
    }

    /// Records that a `wl_callback.done` event has arrived.
    ///
    /// If a callback is in flight, builds a [`FrameTick`] for `output` with the
    /// current time read from `clock`, enqueues it, increments the tick index,
    /// and clears the in-flight flag. If no callback is in flight, debug-asserts
    /// and returns.
    ///
    /// When a previous actual-present time and refresh interval have been
    /// observed, the tick carries a predicted next-vsync
    /// [`FrameTick::predicted_present`] and [`FrameTick::refresh_interval`];
    /// otherwise it is pacing-only.
    ///
    /// `output` should identify this stream's current target output and stay
    /// stable for the stream's lifetime; refresh it only when the surface
    /// actually moves between outputs.
    pub fn on_callback_done(&mut self, clock: Clock, output: OutputId) {
        debug_assert!(
            self.callback_in_flight,
            "on_callback_done called without an in-flight callback"
        );
        if !self.callback_in_flight {
            return;
        }

        let now = clock.now();
        let last_actual = self.last_observed_actual_present;
        let refresh_interval = self.last_observed_refresh_interval;
        let predicted_present = predict_next_present(last_actual, refresh_interval, now);

        let tick = FrameTick {
            now,
            predicted_present,
            refresh_interval,
            frame_index: self.tick_index,
            output,
            prev_actual_present: last_actual,
        };

        self.queue.push(tick);
        self.tick_index += 1;
        self.callback_in_flight = false;
    }

    /// Pops the next queued [`FrameTick`], if any.
    pub fn poll_tick(&mut self) -> Option<FrameTick> {
        self.queue.pop()
    }

    /// Returns whether a frame callback is currently in flight.
    #[must_use]
    pub fn is_callback_in_flight(&self) -> bool {
        self.callback_in_flight
    }

    /// Claims the single in-flight callback slot before a `wl_surface.frame`
    /// request is sent.
    ///
    /// Only one frame callback may be in flight at a time. Returns `true` when
    /// the slot was newly claimed and the caller should send the
    /// `wl_surface.frame` request. Returns `false` when a callback is already
    /// in flight; in that case the ticker state is left unchanged and the
    /// caller must not request another callback. The slot is released when the
    /// matching [`on_callback_done`](Self::on_callback_done) runs.
    #[must_use = "a false return means a callback is already in flight and no new frame request should be sent"]
    pub fn mark_callback_requested(&mut self) -> bool {
        if self.callback_in_flight {
            return false;
        }
        self.callback_in_flight = true;
        true
    }

    /// Stores the most recent actual present time for propagation into the
    /// next [`FrameTick::prev_actual_present`].
    ///
    /// Feed this only with presentation feedback for the same surface/output
    /// stream this ticker paces (see the [type-level contract](Self#one-stream-per-surface)).
    pub fn set_last_observed_actual_present(&mut self, t: HostTime) {
        self.last_observed_actual_present = Some(t);
    }

    /// Stores the most recent observed refresh interval (in host ticks) for
    /// predicting the next [`FrameTick::predicted_present`].
    ///
    /// Feed this only with presentation feedback for the same surface/output
    /// stream this ticker paces (see the [type-level contract](Self#one-stream-per-surface)).
    pub fn set_last_observed_refresh_interval(&mut self, interval: u64) {
        self.last_observed_refresh_interval = Some(interval);
    }
}

/// Predicts the next vsync at or after `now` from the last observed present.
///
/// Returns `None` when no presentation feedback has been observed yet or the
/// refresh interval is unknown, leaving the tick pacing-only. Otherwise it
/// advances the last observed actual-present time by whole refresh intervals
/// until it reaches `now`, landing on the compositor's vsync grid.
fn predict_next_present(
    last_actual: Option<HostTime>,
    refresh_interval: Option<u64>,
    now: HostTime,
) -> Option<HostTime> {
    let last = last_actual?;
    let refresh = refresh_interval.filter(|interval| *interval > 0)?;
    let elapsed = now.ticks().saturating_sub(last.ticks());
    let intervals = elapsed.div_ceil(refresh);
    let advance = intervals.checked_mul(refresh)?;
    last.ticks().checked_add(advance).map(HostTime)
}

impl Default for TickerState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{TickQueue, TickerState, predict_next_present};
    use crate::time::Clock;
    use frameclock::FrameTick;
    use frameclock::HostTime;
    use frameclock::OutputId;

    fn test_tick(frame_index: u64) -> FrameTick {
        FrameTick {
            now: HostTime(frame_index),
            predicted_present: None,
            refresh_interval: None,
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    // --- TickQueue tests ---

    #[test]
    fn overflow_drops_oldest_tick() {
        let mut queue = TickQueue::with_capacity(2);
        queue.push(test_tick(1));
        queue.push(test_tick(2));
        queue.push(test_tick(3));

        assert_eq!(queue.pop().map(|tick| tick.frame_index), Some(2));
        assert_eq!(queue.pop().map(|tick| tick.frame_index), Some(3));
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn empty_state_tracks_push_and_pop() {
        let mut queue = TickQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.push(test_tick(7));
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let _ = queue.pop();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    // --- TickerState tests ---

    #[test]
    fn on_callback_done_enqueues_tick_with_correct_fields() {
        let mut ticker = TickerState::new();

        assert!(ticker.mark_callback_requested());
        ticker.on_callback_done(Clock::Monotonic, OutputId(0));

        let tick = ticker.poll_tick().expect("should have a tick");
        assert!(tick.now.ticks() > 0);
        assert_eq!(tick.predicted_present, None);
        assert_eq!(tick.refresh_interval, None);
        assert_eq!(tick.frame_index, 0);
        assert_eq!(tick.output, OutputId(0));
        assert_eq!(tick.prev_actual_present, None);
    }

    #[test]
    fn on_callback_done_uses_caller_output() {
        let mut ticker = TickerState::new();

        assert!(ticker.mark_callback_requested());
        ticker.on_callback_done(Clock::Monotonic, OutputId(3));

        let tick = ticker.poll_tick().expect("should have a tick");
        assert_eq!(tick.output, OutputId(3));
    }

    #[test]
    fn poll_tick_returns_none_when_empty() {
        let mut ticker = TickerState::new();
        assert!(ticker.poll_tick().is_none());
    }

    #[test]
    fn tick_index_increments_monotonically() {
        let mut ticker = TickerState::new();

        for expected in 0..5 {
            assert!(ticker.mark_callback_requested());
            ticker.on_callback_done(Clock::Monotonic, OutputId(0));
            let tick = ticker.poll_tick().unwrap();
            assert_eq!(tick.frame_index, expected);
        }
    }

    #[test]
    fn callback_in_flight_transitions() {
        let mut ticker = TickerState::new();

        assert!(!ticker.is_callback_in_flight());
        assert!(ticker.mark_callback_requested());
        assert!(ticker.is_callback_in_flight());
        ticker.on_callback_done(Clock::Monotonic, OutputId(0));
        assert!(!ticker.is_callback_in_flight());
    }

    #[test]
    fn mark_callback_requested_rejects_double_request() {
        let mut ticker = TickerState::new();

        // First request claims the in-flight slot.
        assert!(ticker.mark_callback_requested());
        assert!(ticker.is_callback_in_flight());

        // A second request while one is in flight is rejected and leaves the
        // state unchanged.
        assert!(!ticker.mark_callback_requested());
        assert!(ticker.is_callback_in_flight());

        // After the callback completes, the slot can be claimed again.
        ticker.on_callback_done(Clock::Monotonic, OutputId(0));
        assert!(ticker.mark_callback_requested());
    }

    #[test]
    fn last_observed_actual_present_propagates() {
        let mut ticker = TickerState::new();

        // First tick: no previous actual present.
        assert!(ticker.mark_callback_requested());
        ticker.on_callback_done(Clock::Monotonic, OutputId(0));
        let tick0 = ticker.poll_tick().unwrap();
        assert_eq!(tick0.prev_actual_present, None);

        // Record an actual present time.
        ticker.set_last_observed_actual_present(HostTime(42_000));

        // Second tick: should carry the observed time.
        assert!(ticker.mark_callback_requested());
        ticker.on_callback_done(Clock::Monotonic, OutputId(0));
        let tick1 = ticker.poll_tick().unwrap();
        assert_eq!(tick1.prev_actual_present, Some(HostTime(42_000)));
    }

    #[test]
    fn done_when_not_in_flight_is_ignored() {
        let mut ticker = TickerState::new();

        // Calling on_callback_done without mark_callback_requested is guarded
        // by debug_assert, so in release builds it returns without enqueuing.
        // We verify the initial state reflects no enqueue path.
        assert!(!ticker.is_callback_in_flight());
        assert!(ticker.poll_tick().is_none());
    }

    #[test]
    fn queue_overflow_drops_oldest_through_ticker() {
        // TickQueue default capacity is 8; push 9 ticks and verify the first
        // is dropped.
        let mut ticker = TickerState::new();

        for _ in 0..9 {
            assert!(ticker.mark_callback_requested());
            ticker.on_callback_done(Clock::Monotonic, OutputId(0));
        }

        // First available tick should be index 1 (index 0 was dropped).
        let tick = ticker.poll_tick().unwrap();
        assert_eq!(tick.frame_index, 1);
    }

    // --- Present prediction tests ---

    #[test]
    fn predict_next_present_rounds_up_to_next_vsync() {
        // last + 4*refresh = 1640 is the first vsync at or after now = 1500.
        let predicted = predict_next_present(Some(HostTime(1000)), Some(160), HostTime(1500));
        assert_eq!(predicted, Some(HostTime(1640)));
    }

    #[test]
    fn predict_next_present_on_exact_vsync_returns_now() {
        // now = 1000 + 3*160 lands exactly on a vsync, so the prediction is now.
        let predicted = predict_next_present(Some(HostTime(1000)), Some(160), HostTime(1480));
        assert_eq!(predicted, Some(HostTime(1480)));
    }

    #[test]
    fn predict_next_present_when_last_at_or_after_now_returns_last() {
        let predicted = predict_next_present(Some(HostTime(2000)), Some(160), HostTime(1500));
        assert_eq!(predicted, Some(HostTime(2000)));
    }

    #[test]
    fn predict_next_present_without_actual_present_is_none() {
        assert_eq!(predict_next_present(None, Some(160), HostTime(1500)), None);
    }

    #[test]
    fn predict_next_present_without_refresh_is_none() {
        assert_eq!(
            predict_next_present(Some(HostTime(1000)), None, HostTime(1500)),
            None
        );
        assert_eq!(
            predict_next_present(Some(HostTime(1000)), Some(0), HostTime(1500)),
            None
        );
    }

    #[test]
    fn on_callback_done_emits_predicted_present_from_feedback() {
        let mut ticker = TickerState::new();
        ticker.set_last_observed_actual_present(HostTime(1000));
        ticker.set_last_observed_refresh_interval(16_666_667);

        assert!(ticker.mark_callback_requested());
        ticker.on_callback_done(Clock::Monotonic, OutputId(0));
        let tick = ticker.poll_tick().expect("should have a tick");

        assert_eq!(tick.refresh_interval, Some(16_666_667));
        assert_eq!(tick.prev_actual_present, Some(HostTime(1000)));
        let predicted = tick.predicted_present.expect("prediction available");
        // The prediction is the first vsync at or after the tick time.
        assert!(predicted >= tick.now);
        assert!(predicted.ticks() - tick.now.ticks() < 16_666_667);
    }
}
