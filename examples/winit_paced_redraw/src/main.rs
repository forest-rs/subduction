// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal winit paced redraw loop using `frameclock`.
//!
//! This example does not render anything. It shows where `frameclock` planning
//! fits in a winit application: send redraw demand to [`FrameDriver`], convert
//! redraw opportunities into [`FrameTick`]s, wake at the plan's frame-start time,
//! submit work, and feed pacing feedback back into the scheduler.
//!
//! A real renderer would replace the synthetic submit and feedback below with
//! surface acquisition, command submission, and platform or swapchain timing
//! feedback. The `frameclock` calls should stay in the same places.
//!
//! The example also uses `understory_timing` to keep the host wake queue
//! separate from the per-surface frame scheduler. `frameclock` contributes the
//! frame-start wake for this surface; the host merges that wake with its other
//! timers.

use std::time::Duration as StdDuration;

use frameclock::{
    ActiveFrame, DisplayTiming, Duration, FrameDemand, FrameDriver, FrameOpportunity, FramePlan,
    FrameSubmission, FrameTick, HostTime, OutputId, PresentHints, SchedulerConfig,
    TimingConfidence,
};
use understory_timing::{TimerId, TimerInstant, TimerQueue};
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const REFRESH_INTERVAL: Duration = Duration(16_666_667);
const DEMO_ANIMATION_DURATION: Duration = Duration(5_000_000_000);
const SYNTHETIC_BUILD_COST: Duration = Duration(500_000);

struct App {
    started_at: Instant,
    window: Option<WindowState>,
}

struct WindowState {
    window: Window,
    model: DemoModel,
    renderer: SyntheticRenderer,
    surface_clock: SurfaceFrameClock,
    timers: TimerQueue<Wake>,
    frame_start_wake: Option<TimerId>,
}

struct SurfaceFrameClock {
    driver: FrameDriver,
    frame_index: u64,
    output: OutputId,
}

struct DemoModel {
    animation_active: bool,
    sampled_position: u64,
}

struct SyntheticRenderer;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Wake {
    // App timers, animation pulses, and delayed UI work would be additional
    // variants in the same queue. The frame-start wake is only a host wake; it
    // does not create new app-visible frame demand.
    FrameStart(OutputId),
}

fn host_time(started_at: Instant) -> HostTime {
    let nanos = started_at.elapsed().as_nanos();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "example runtime is far shorter than u64 nanoseconds"
    )]
    {
        HostTime(nanos as u64)
    }
}

impl App {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            window: None,
        }
    }
}

impl WindowState {
    fn new(window: Window, output: OutputId) -> Self {
        Self {
            window,
            model: DemoModel::new(),
            renderer: SyntheticRenderer,
            surface_clock: SurfaceFrameClock::new(output),
            timers: TimerQueue::new(),
            frame_start_wake: None,
        }
    }

    /// Adds app-visible demand and asks winit to deliver `RedrawRequested`.
    fn request_frame(&mut self, demand: FrameDemand) {
        self.surface_clock.driver.request(demand);
        self.window.request_redraw();
    }

    fn drain_timers(&mut self, now: HostTime) {
        while let Some(timer) = self.timers.pop_expired(now.ticks()) {
            let id = timer.id();
            match *timer.target() {
                Wake::FrameStart(output) => {
                    if output == self.surface_clock.output && self.frame_start_wake == Some(id) {
                        self.frame_start_wake = None;
                        // A frame-start wake means a previously planned frame
                        // is due. It wakes winit without adding new app
                        // demand.
                        self.window.request_redraw();
                    }
                }
            }
        }
    }

    fn cancel_frame_start_wake(&mut self) {
        if let Some(id) = self.frame_start_wake.take() {
            self.timers.cancel(id);
        }
    }

    fn schedule_frame_start(&mut self, frame_start: HostTime) {
        self.cancel_frame_start_wake();

        self.frame_start_wake = Some(self.timers.schedule_once_at(
            Wake::FrameStart(self.surface_clock.output),
            frame_start.ticks(),
        ));
    }

    /// Mirrors the frame driver's queued frame-start time into the host timer.
    ///
    /// This keeps timer ownership at the app/window level. `frameclock`
    /// decides the start time; the host decides how that wake is merged with
    /// input, animation, layout, and other app timers.
    fn sync_frame_start_wake(&mut self) {
        if let Some(frame_start) = self.surface_clock.driver.next_frame_start() {
            self.schedule_frame_start(frame_start);
        } else {
            self.cancel_frame_start_wake();
        }
    }

    fn arm_next_wake(&self, started_at: Instant, event_loop: &ActiveEventLoop) {
        match self.timers.next_deadline() {
            Some(deadline) => {
                let wake_instant = instant_for(started_at, deadline);
                if wake_instant <= Instant::now() {
                    event_loop.set_control_flow(ControlFlow::Poll);
                } else {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(wake_instant));
                }
            }
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn frame_tick(&self, now: HostTime) -> FrameTick {
        // Plain winit does not expose the future present time here, so this
        // tick carries refresh-rate information but no predicted present.
        FrameTick {
            now,
            predicted_present: None,
            refresh_interval: Some(REFRESH_INTERVAL.ticks()),
            confidence: TimingConfidence::PacingOnly,
            frame_index: self.surface_clock.frame_index,
            output: self.surface_clock.output,
            prev_actual_present: None,
        }
    }

    fn present_hints(&self, now: HostTime) -> PresentHints {
        // Hints express app-side intent and constraints. With predictive
        // display timing, desired_present would normally be the display target.
        // In this pacing-only example, latest_commit gives the scheduler a
        // conservative "submit by around the next refresh" boundary.
        PresentHints {
            desired_present: None,
            latest_commit: now + REFRESH_INTERVAL,
        }
    }

    fn begin_frame(&mut self, now: HostTime) -> Option<ActiveFrame> {
        let tick = self.frame_tick(now);
        let hints = self.present_hints(now);
        self.surface_clock.driver.begin_frame(FrameOpportunity::new(
            tick,
            hints,
            DisplayTiming::from_tick(&tick, REFRESH_INTERVAL),
        ))
    }

    fn redraw(&mut self, started_at: Instant, event_loop: &ActiveEventLoop) {
        // Frameclock uses an explicit monotonic host timeline. A real app would
        // usually centralize this conversion so renderer and platform feedback
        // report times on the same clock.
        let now = host_time(started_at);

        // RedrawRequested is winit's signal that the app may build or plan a
        // frame. The driver owns pending demand and queued frame plans. If the
        // selected frame start is still in the future, it returns `None`; the
        // host mirrors `next_frame_start()` into its timer queue and sleeps.
        let Some(frame) = self.begin_frame(now) else {
            self.sync_frame_start_wake();
            self.arm_next_wake(started_at, event_loop);
            return;
        };
        let plan = frame.plan();
        self.cancel_frame_start_wake();

        // Update application state and models here. Time-varying content samples
        // `plan.sample_time`, not wall-clock "now", so CPU work targets the
        // frame that is expected to be displayed.
        self.model.update_for_frame(plan.sample_time);

        // Render here. A real renderer would acquire the surface texture, encode
        // commands, and submit before `plan.commit_deadline`. If the backend
        // provides a real `plan.target_present`, presentation-aware renderers can
        // use it to pick content or configure platform-specific present timing.
        let submission = self.renderer.render(&plan, now);

        // Submitting through the driver feeds scheduler feedback internally and
        // returns the frameclock-owned timing summary a devtools view would use.
        let summary = self.surface_clock.driver.submit_frame(frame, submission);

        if self.surface_clock.frame_index.is_multiple_of(60) {
            self.window.set_title(&format!(
                "Frameclock + winit: sample={}ms x={}",
                plan.sample_time.ticks() / 1_000_000,
                self.model.sampled_position,
            ));
            println!(
                "frame={:04} start={} sample={} target={:?} deadline={} depth={} overrun={:?}",
                plan.frame_index,
                plan.frame_start.ticks(),
                plan.sample_time.ticks(),
                plan.target_present.map(HostTime::ticks),
                plan.commit_deadline.ticks(),
                plan.pipeline_depth,
                summary.pacing_overrun,
            );
        }

        self.surface_clock.frame_index += 1;

        // If lower-priority demand was retained behind the queued frame we
        // just consumed, wake winit again so the driver can plan it on a fresh
        // turn. The host does not inspect or rank those demand bits itself.
        let driver_needs_redraw = self.surface_clock.driver.has_pending_demand();

        let mut next_demand = FrameDemand::NONE;
        if self.model.animation_active {
            next_demand.insert(FrameDemand::ANIMATION);
        }

        if next_demand.is_empty() {
            if driver_needs_redraw {
                self.window.request_redraw();
            }
            self.arm_next_wake(started_at, event_loop);
            return;
        }

        // Ask for another redraw so the driver can turn this new demand into a
        // queued plan. If that plan starts in the future, the next redraw will
        // only arm the host timer; it will not do app or render work early.
        self.request_frame(next_demand);
        self.arm_next_wake(started_at, event_loop);
    }
}

fn instant_for(started_at: Instant, deadline: TimerInstant) -> Instant {
    started_at + StdDuration::from_nanos(deadline)
}

impl SurfaceFrameClock {
    fn new(output: OutputId) -> Self {
        let mut config = SchedulerConfig::pacing_only();
        config.initial_depth = 1;

        Self {
            // Winit alone does not give us predicted presentation timestamps, so
            // this example uses pacing-only scheduling. Platform backends can
            // switch to predictive ticks when they have stronger display timing.
            //
            // Start this example at depth 1: without misses or pacing overruns,
            // there is no evidence that the app should add latency. The adaptive
            // policy can still raise depth if repeated overruns show that this is
            // too aggressive.
            driver: FrameDriver::new(config),
            frame_index: 0,
            output,
        }
    }
}

impl DemoModel {
    const fn new() -> Self {
        Self {
            animation_active: true,
            sampled_position: 0,
        }
    }

    fn update_for_frame(&mut self, sample_time: HostTime) {
        let millis = sample_time.ticks() / 1_000_000;
        let phase = millis % 2_000;
        self.sampled_position = if phase <= 1_000 { phase } else { 2_000 - phase };
        self.animation_active = sample_time < HostTime(DEMO_ANIMATION_DURATION.ticks());
    }
}

impl SyntheticRenderer {
    fn render(&mut self, plan: &FramePlan, now: HostTime) -> FrameSubmission {
        // This example has no renderer, so it invents a short submit span.
        // The driver recorded frame build start when it returned ActiveFrame;
        // a real backend would provide the queue submit time and attach
        // platform present feedback when available.
        let budget = plan.commit_deadline.saturating_duration_since(now);
        let build_cost = Duration(SYNTHETIC_BUILD_COST.ticks().min(budget.ticks()));

        FrameSubmission {
            submitted_at: now + build_cost,
            actual_present: None,
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: StartCause) {
        // Winit wakes for many reasons. Drain due timers on every pass so app
        // timers, animation pulses, and frame-start wakes share one deadline
        // queue.
        if let Some(window) = &mut self.window {
            window.drain_timers(host_time(self.started_at));
            window.arm_next_wake(self.started_at, event_loop);
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Frameclock + winit"))
            .expect("failed to create window");
        let mut window = WindowState::new(window, OutputId(0));

        // Kick the first animation frame. After that, animation demand keeps
        // scheduling frames until the demo animation marks itself inactive.
        window.request_frame(FrameDemand::ANIMATION);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_matches = self
            .window
            .as_ref()
            .is_some_and(|window| window.window.id() == window_id);
        if !window_matches {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.window = None;
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(window) = &mut self.window {
                    // Treat resize as continuous input: it is latency-sensitive,
                    // but can still step down if rendering becomes too slow.
                    window.request_frame(FrameDemand::CONTINUOUS_INPUT);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &mut self.window {
                    window.redraw(self.started_at, event_loop);
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop failed");
}
