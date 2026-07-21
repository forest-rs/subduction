// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Present-extension timing and feedback source.
//!
//! [`X11PresentSource`] turns a window's Present completions into `frameclock`
//! ticks and per-submission feedback. It is pure bookkeeping over
//! [`frameclock_x11`] and references no `x11rb` type; a host decodes the
//! Present events and feeds their fields in.
//!
//! Two Present event streams feed it:
//!
//! - **`NotifyMSC` completions** — the vblank stream armed with `PresentNotifyMSC`
//!   (see [`mark_notify_requested`](X11PresentSource::mark_notify_requested) and
//!   [`on_notify_complete`](X11PresentSource::on_notify_complete)). These become
//!   [`FrameTick`]s.
//! - **Per-pixmap completions** — the feedback for buffers submitted with
//!   `PresentPixmap` (see [`on_pixmap_complete`](X11PresentSource::on_pixmap_complete)
//!   and [`on_idle`](X11PresentSource::on_idle)). These become [`PresentEvent`]s
//!   correlated by the [`SubmissionId`] a presenter stamps on each submission.

use frameclock::{FrameTick, OutputId};
use frameclock_x11::{Clock, PresentEvent, PresentEventQueue, PresentTicker, SubmissionId};

/// Turns a window's Present completions into `frameclock` ticks and feedback.
///
/// Keep one per presented window. Drive the vblank stream by calling
/// [`mark_notify_requested`](Self::mark_notify_requested) before arming a
/// `PresentNotifyMSC` request and [`on_notify_complete`](Self::on_notify_complete)
/// when the matching completion arrives, then drain [`poll_tick`](Self::poll_tick).
/// A presenter that issues `PresentPixmap` allocates a [`SubmissionId`] with
/// [`next_submission`](Self::next_submission), and reports the buffer's fate
/// through [`on_pixmap_complete`](Self::on_pixmap_complete) /
/// [`on_idle`](Self::on_idle), draining [`poll_present_event`](Self::poll_present_event).
#[derive(Debug)]
pub struct X11PresentSource {
    ticker: PresentTicker,
    feedback: PresentEventQueue,
    clock: Clock,
    output: OutputId,
    next_serial: u32,
}

impl X11PresentSource {
    /// Creates a source for [`OutputId`] `0`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_output(OutputId(0))
    }

    /// Creates a source whose ticks and feedback carry `output`.
    #[must_use]
    pub fn with_output(output: OutputId) -> Self {
        Self {
            ticker: PresentTicker::with_output(output),
            feedback: PresentEventQueue::default(),
            clock: Clock::Monotonic,
            output,
            next_serial: 0,
        }
    }

    /// Claims the single in-flight vblank-notify slot before arming
    /// `PresentNotifyMSC`.
    ///
    /// Returns `true` when the caller should send the request; `false` when a
    /// notify is already in flight. Delegates to
    /// [`PresentTicker::mark_notify_requested`].
    #[must_use = "a false return means a notify is already in flight and no new request should be armed"]
    pub fn mark_notify_requested(&mut self) -> bool {
        self.ticker.mark_notify_requested()
    }

    /// Records a `NotifyMSC` completion, enqueuing a [`FrameTick`].
    ///
    /// `ust` and `msc` are the fields of the `PresentCompleteNotify` event of
    /// kind `NotifyMSC`.
    pub fn on_notify_complete(&mut self, ust: u64, msc: u64) {
        self.ticker.on_complete_notify(self.clock, ust, msc);
    }

    /// Pops the next queued [`FrameTick`], if any.
    pub fn poll_tick(&mut self) -> Option<FrameTick> {
        self.ticker.poll_tick()
    }

    /// Allocates the next [`SubmissionId`] to stamp on a `PresentPixmap`.
    ///
    /// The returned value is the `serial` a presenter passes to `PresentPixmap`
    /// and matches against the `serial` echoed by the completion and idle events.
    pub fn next_submission(&mut self) -> SubmissionId {
        let serial = self.next_serial;
        self.next_serial = self.next_serial.wrapping_add(1);
        SubmissionId(serial)
    }

    /// Records a per-pixmap `PresentCompleteNotify` (kind `Pixmap`) as feedback.
    ///
    /// `serial` is the submission's [`SubmissionId`] serial; `mode` is the raw
    /// `CompleteMode` (`Copy`, `Flip`, `Skip`, …).
    pub fn on_pixmap_complete(&mut self, serial: u32, ust: u64, msc: u64, mode: u8) {
        self.feedback.push(PresentEvent::presented(
            SubmissionId(serial),
            ust,
            msc,
            Some(self.output),
            mode,
        ));
    }

    /// Records a `PresentIdleNotify`: the submission's buffer is reusable.
    pub fn on_idle(&mut self, serial: u32) {
        self.feedback.push(PresentEvent::Idle {
            id: SubmissionId(serial),
        });
    }

    /// Pops the next queued per-submission [`PresentEvent`], if any.
    pub fn poll_present_event(&mut self) -> Option<PresentEvent> {
        self.feedback.pop()
    }

    /// Returns the most recently observed refresh interval in host ticks, once a
    /// second `NotifyMSC` completion has established the cadence.
    #[must_use]
    pub fn observed_refresh_interval(&self) -> Option<u64> {
        self.ticker.observed_refresh_interval()
    }
}

impl Default for X11PresentSource {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::X11PresentSource;
    use frameclock_x11::{PresentEvent, SubmissionId};

    #[test]
    fn notify_completions_produce_ticks() {
        let mut source = X11PresentSource::new();
        assert!(source.mark_notify_requested());
        source.on_notify_complete(1_000_000, 100);
        let tick = source.poll_tick().expect("a notify completion is a tick");
        assert_eq!(
            tick.prev_actual_present,
            Some(frameclock::HostTime(1_000_000_000))
        );
    }

    #[test]
    fn single_notify_in_flight() {
        let mut source = X11PresentSource::new();
        assert!(source.mark_notify_requested());
        assert!(!source.mark_notify_requested());
        source.on_notify_complete(1_000_000, 1);
        assert!(source.mark_notify_requested());
    }

    #[test]
    fn submissions_get_monotonic_ids() {
        let mut source = X11PresentSource::new();
        assert_eq!(source.next_submission(), SubmissionId(0));
        assert_eq!(source.next_submission(), SubmissionId(1));
        assert_eq!(source.next_submission(), SubmissionId(2));
    }

    #[test]
    fn pixmap_completion_and_idle_are_feedback() {
        let mut source = X11PresentSource::new();
        let id = source.next_submission();
        source.on_pixmap_complete(id.0, 2_000_000, 200, 1);
        source.on_idle(id.0);

        let presented = source.poll_present_event().expect("a presented event");
        match presented {
            PresentEvent::Presented {
                id: got, msc, mode, ..
            } => {
                assert_eq!(got, id);
                assert_eq!(msc, 200);
                assert_eq!(mode, 1);
            }
            other => panic!("expected Presented, got {other:?}"),
        }
        assert_eq!(source.poll_present_event(), Some(PresentEvent::Idle { id }));
        assert_eq!(source.poll_present_event(), None);
    }

    #[test]
    fn refresh_interval_emerges_after_second_completion() {
        let mut source = X11PresentSource::new();
        assert!(source.mark_notify_requested());
        source.on_notify_complete(1_000_000, 100);
        assert_eq!(source.observed_refresh_interval(), None);
        assert!(source.mark_notify_requested());
        source.on_notify_complete(1_016_667, 101);
        assert_eq!(source.observed_refresh_interval(), Some(16_667_000));
    }
}
