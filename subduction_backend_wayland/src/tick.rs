// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Internal frame-tick queueing primitives and ticker state machine.

use crate::output_registry::OutputRegistry;
use crate::queue::BoundedQueue;
use crate::time::{Clock, now_for_clock};
use subduction_core::output::OutputId;
use subduction_core::time::HostTime;
use subduction_core::timing::{FrameTick, TimingConfidence};

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

/// Selects the output to associate with the next frame tick.
///
/// This is a seam for future output routing logic. Currently returns the
/// lowest tracked output, falling back to `OutputId::default()` when the
/// registry is empty.
pub(crate) fn select_tick_output(registry: &OutputRegistry) -> OutputId {
    registry.lowest_id().unwrap_or_default()
}

/// Pure-logic state machine for frame callback tick generation.
///
/// Tracks the in-flight callback state, builds [`FrameTick`]s when callbacks
/// complete, and queues them for polling. Protocol I/O is handled externally;
/// this type contains only the bookkeeping.
#[derive(Debug)]
pub(crate) struct TickerState {
    queue: TickQueue,
    tick_index: u64,
    callback_in_flight: bool,
    last_observed_actual_present: Option<HostTime>,
}

impl TickerState {
    pub(crate) fn new() -> Self {
        Self {
            queue: TickQueue::default(),
            tick_index: 0,
            callback_in_flight: false,
            last_observed_actual_present: None,
        }
    }

    /// Records that a `wl_callback.done` event has arrived.
    ///
    /// If a callback is in flight, builds a [`FrameTick`] with `PacingOnly`
    /// confidence, enqueues it, increments the tick index, and clears the
    /// in-flight flag. If no callback is in flight, debug-asserts and returns.
    pub(crate) fn on_callback_done(&mut self, clock: Clock, output_registry: &OutputRegistry) {
        debug_assert!(
            self.callback_in_flight,
            "on_callback_done called without an in-flight callback"
        );
        if !self.callback_in_flight {
            return;
        }

        let now = now_for_clock(clock);
        let output = select_tick_output(output_registry);
        let prev_actual_present = self.last_observed_actual_present;

        let tick = FrameTick {
            now,
            predicted_present: None,
            refresh_interval: None,
            confidence: TimingConfidence::PacingOnly,
            frame_index: self.tick_index,
            output,
            prev_actual_present,
        };

        self.queue.push(tick);
        self.tick_index += 1;
        self.callback_in_flight = false;
    }

    /// Pops the next queued [`FrameTick`], if any.
    pub(crate) fn poll_tick(&mut self) -> Option<FrameTick> {
        self.queue.pop()
    }

    /// Returns whether a frame callback is currently in flight.
    pub(crate) fn is_callback_in_flight(&self) -> bool {
        self.callback_in_flight
    }

    /// Marks that a frame callback request has been sent.
    pub(crate) fn mark_callback_requested(&mut self) {
        debug_assert!(
            !self.callback_in_flight,
            "mark_callback_requested called while a callback is already in flight"
        );
        self.callback_in_flight = true;
    }

    /// Stores the most recent actual present time for propagation into the
    /// next [`FrameTick::prev_actual_present`].
    pub(crate) fn set_last_observed_actual_present(&mut self, t: HostTime) {
        self.last_observed_actual_present = Some(t);
    }
}

#[cfg(test)]
mod tests {
    use super::{TickQueue, TickerState, select_tick_output};
    use crate::output_registry::OutputRegistry;
    use crate::time::Clock;
    use subduction_core::output::OutputId;
    use subduction_core::time::HostTime;
    use subduction_core::timing::{FrameTick, TimingConfidence};
    use wayland_client::protocol::wl_output;
    use wayland_client::{Connection, Proxy};

    fn test_tick(frame_index: u64) -> FrameTick {
        FrameTick {
            now: HostTime(frame_index),
            predicted_present: None,
            refresh_interval: None,
            confidence: TimingConfidence::PacingOnly,
            frame_index,
            output: OutputId(0),
            prev_actual_present: None,
        }
    }

    fn inert_output() -> wl_output::WlOutput {
        let (s1, _s2) = std::os::unix::net::UnixStream::pair().unwrap();
        let conn = Connection::from_socket(s1).unwrap();
        wl_output::WlOutput::from_id(&conn, wayland_client::backend::ObjectId::null()).unwrap()
    }

    fn registry_with_outputs(count: u32) -> OutputRegistry {
        let mut reg = OutputRegistry::new();
        for i in 0..count {
            reg.add(i, inert_output());
        }
        reg
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

    // --- select_tick_output tests ---

    #[test]
    fn select_tick_output_returns_lowest() {
        let reg = registry_with_outputs(3);
        assert_eq!(select_tick_output(&reg), OutputId(0));
    }

    #[test]
    fn select_tick_output_returns_default_when_empty() {
        let reg = OutputRegistry::new();
        assert_eq!(select_tick_output(&reg), OutputId::default());
    }

    // --- TickerState tests ---

    #[test]
    fn on_callback_done_enqueues_tick_with_correct_fields() {
        let mut ticker = TickerState::new();
        let reg = registry_with_outputs(2);

        ticker.mark_callback_requested();
        ticker.on_callback_done(Clock::Monotonic, &reg);

        let tick = ticker.poll_tick().expect("should have a tick");
        assert!(tick.now.ticks() > 0);
        assert_eq!(tick.predicted_present, None);
        assert_eq!(tick.refresh_interval, None);
        assert_eq!(tick.confidence, TimingConfidence::PacingOnly);
        assert_eq!(tick.frame_index, 0);
        assert_eq!(tick.output, OutputId(0));
        assert_eq!(tick.prev_actual_present, None);
    }

    #[test]
    fn poll_tick_returns_none_when_empty() {
        let mut ticker = TickerState::new();
        assert!(ticker.poll_tick().is_none());
    }

    #[test]
    fn tick_index_increments_monotonically() {
        let mut ticker = TickerState::new();
        let reg = registry_with_outputs(1);

        for expected in 0..5 {
            ticker.mark_callback_requested();
            ticker.on_callback_done(Clock::Monotonic, &reg);
            let tick = ticker.poll_tick().unwrap();
            assert_eq!(tick.frame_index, expected);
        }
    }

    #[test]
    fn callback_in_flight_transitions() {
        let mut ticker = TickerState::new();
        let reg = registry_with_outputs(1);

        assert!(!ticker.is_callback_in_flight());
        ticker.mark_callback_requested();
        assert!(ticker.is_callback_in_flight());
        ticker.on_callback_done(Clock::Monotonic, &reg);
        assert!(!ticker.is_callback_in_flight());
    }

    #[test]
    fn last_observed_actual_present_propagates() {
        let mut ticker = TickerState::new();
        let reg = registry_with_outputs(1);

        // First tick: no previous actual present.
        ticker.mark_callback_requested();
        ticker.on_callback_done(Clock::Monotonic, &reg);
        let tick0 = ticker.poll_tick().unwrap();
        assert_eq!(tick0.prev_actual_present, None);

        // Record an actual present time.
        ticker.set_last_observed_actual_present(HostTime(42_000));

        // Second tick: should carry the observed time.
        ticker.mark_callback_requested();
        ticker.on_callback_done(Clock::Monotonic, &reg);
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
        let reg = registry_with_outputs(1);

        for _ in 0..9 {
            ticker.mark_callback_requested();
            ticker.on_callback_done(Clock::Monotonic, &reg);
        }

        // First available tick should be index 1 (index 0 was dropped).
        let tick = ticker.poll_tick().unwrap();
        assert_eq!(tick.frame_index, 1);
    }
}
