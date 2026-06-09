// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal winit paced redraw loop using `frameclock`.
//!
//! This example does not render anything. It shows where `frameclock` planning
//! fits in a winit application: convert redraw demand into [`FrameTick`]s, ask
//! the scheduler for a [`FramePlan`], wake at the plan's frame-start time,
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
    DisplayTiming, Duration, FrameDemand, FramePlan, FrameRequest, FrameTick, HostTime, OutputId,
    PresentFeedback, PresentHints, Scheduler, SchedulerConfig, TimingConfidence,
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
    /// Planned frame that should not start until its scheduler-selected
    /// `FramePlan::frame_start`.
    ///
    /// Winit may deliver `RedrawRequested` before the ideal frame-start time.
    /// Keep the plan instead of re-planning so the eventual redraw samples the
    /// time and uses the deadline selected for the same frame opportunity.
    /// If stronger demand arrives while a frame is waiting, this example
    /// discards the queued plan and asks `frameclock` for a new one.
    pending_frame: Option<PlannedFrame>,
    /// Demand that arrived after the last planned frame.
    ///
    /// This is app-visible work: input, resize, animation, or delayed updates.
    /// A scheduled frame-start wake does not insert demand here; it only wakes
    /// the event loop so an existing `pending_frame` can run.
    pending_demand: FrameDemand,
    timers: TimerQueue<Wake>,
    frame_start_wake: Option<TimerId>,
}

struct SurfaceFrameClock {
    scheduler: Scheduler,
    frame_index: u64,
    output: OutputId,
}

/// A frame plan plus the backend hints used to produce it.
///
/// The hints are retained with the plan because feedback must be resolved
/// against the same desired-present/deadline facts that were active when the
/// scheduler made the decision.
#[derive(Clone, Copy)]
struct PlannedFrame {
    /// Scheduler-selected frame timing.
    plan: FramePlan,
    /// Presentation constraints paired with `plan`.
    hints: PresentHints,
}

struct DemoModel {
    animation_active: bool,
    sampled_position: u64,
}

struct SyntheticRenderer;

struct SyntheticSubmission {
    build_start: HostTime,
    submitted_at: HostTime,
    actual_present: Option<HostTime>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Wake {
    // App timers, animation pulses, and delayed UI work would be additional
    // variants in the same queue. The frame-start wake is just one source of
    // demand.
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
            pending_frame: None,
            pending_demand: FrameDemand::NONE,
            timers: TimerQueue::new(),
            frame_start_wake: None,
        }
    }

    /// Adds app-visible demand and asks winit to deliver `RedrawRequested`.
    fn request_frame(&mut self, demand: FrameDemand) {
        self.pending_demand.insert(demand);
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

    /// Stores a planned frame and arms the host timer for its start time.
    ///
    /// This keeps timer ownership at the app/window level. `frameclock`
    /// decides the start time; the host decides how that wake is merged with
    /// input, animation, layout, and other app timers.
    fn queue_frame(&mut self, frame: PlannedFrame) {
        self.schedule_frame_start(frame.plan.frame_start);
        self.pending_frame = Some(frame);
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

    fn plan_frame(&mut self, now: HostTime, demand: FrameDemand) -> PlannedFrame {
        debug_assert!(!demand.is_empty(), "no demand should not plan a frame");

        // Plain winit does not expose the future present time here, so this
        // tick carries refresh-rate information but no predicted present.
        let tick = FrameTick {
            now,
            predicted_present: None,
            refresh_interval: Some(REFRESH_INTERVAL.ticks()),
            confidence: TimingConfidence::PacingOnly,
            frame_index: self.surface_clock.frame_index,
            output: self.surface_clock.output,
            prev_actual_present: None,
        };

        // Hints express app-side intent and constraints. With predictive
        // display timing, desired_present would normally be the display target.
        // In this pacing-only example, latest_commit gives the scheduler a
        // conservative "submit by around the next refresh" boundary.
        let hints = PresentHints {
            desired_present: None,
            latest_commit: now + REFRESH_INTERVAL,
        };

        let plan = self.surface_clock.scheduler.plan(FrameRequest::new(
            tick,
            hints,
            demand,
            DisplayTiming::from_tick(&tick, REFRESH_INTERVAL),
        ));
        PlannedFrame { plan, hints }
    }

    fn redraw(&mut self, started_at: Instant, event_loop: &ActiveEventLoop) {
        // Frameclock uses an explicit monotonic host timeline. A real app would
        // usually centralize this conversion so renderer and platform feedback
        // report times on the same clock.
        let now = host_time(started_at);

        // RedrawRequested is winit's signal that the app may build another
        // frame. A plan can intentionally be earlier than the render work: if
        // frame_start is still in the future, store the plan and let the shared
        // timer queue wake the event loop at the scheduler-selected time.
        let planned_frame = match self.pending_frame.take() {
            Some(frame) if now >= frame.plan.frame_start => frame,
            Some(frame) if demand_preempts_plan(self.pending_demand, frame.plan.demand) => {
                // Input or resize may arrive while an animation frame is parked
                // for a later start time. Re-plan instead of forcing the newer,
                // more urgent work to wait behind the old cadence decision.
                self.cancel_frame_start_wake();
                let demand = self.pending_demand;
                self.pending_demand = FrameDemand::NONE;
                self.plan_frame(now, demand)
            }
            Some(frame) => {
                self.queue_frame(frame);
                self.arm_next_wake(started_at, event_loop);
                return;
            }
            None => {
                // Spurious redraws can happen. With no planned frame and no
                // app-visible demand, stay idle instead of teaching that
                // FrameDemand::NONE is a normal render mode.
                if self.pending_demand.is_empty() {
                    self.arm_next_wake(started_at, event_loop);
                    return;
                }

                let demand = self.pending_demand;
                self.pending_demand = FrameDemand::NONE;
                self.plan_frame(now, demand)
            }
        };
        let plan = planned_frame.plan;
        let hints = planned_frame.hints;

        // A redraw requested by input, resize, or the OS can arrive before the
        // planned frame-start timer. Keep the existing plan and sleep until the
        // scheduler-selected start instead of doing app/render work early.
        if now < plan.frame_start {
            self.queue_frame(planned_frame);
            self.arm_next_wake(started_at, event_loop);
            return;
        }

        // Update application state and models here. Time-varying content samples
        // `plan.sample_time`, not wall-clock "now", so CPU work targets the
        // frame that is expected to be displayed.
        self.model.update_for_frame(plan.sample_time);

        // Render here. A real renderer would acquire the surface texture, encode
        // commands, and submit before `plan.commit_deadline`. If the backend
        // provides a real `plan.target_present`, presentation-aware renderers can
        // use it to pick content or configure platform-specific present timing.
        let submission = self.renderer.render(&plan, now);

        let feedback = PresentFeedback::new(
            &hints,
            submission.build_start,
            submission.submitted_at,
            submission.actual_present,
        );

        // Feeding feedback back into the scheduler lets it adapt safety margins
        // and pipeline depth when frames miss deadlines or submit too late.
        self.surface_clock.scheduler.observe(&feedback);

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
                feedback.pacing_overrun,
            );
        }

        let mut next_demand = self.pending_demand;
        self.pending_demand = FrameDemand::NONE;
        if self.model.animation_active {
            next_demand.insert(FrameDemand::ANIMATION);
        }

        if next_demand.is_empty() {
            self.cancel_frame_start_wake();
            self.arm_next_wake(started_at, event_loop);
            self.surface_clock.frame_index += 1;
            return;
        }

        self.surface_clock.frame_index += 1;

        // Plan the next frame and queue the scheduler-selected frame start
        // alongside the host's other timers. This is the app/window-level
        // queue; the per-surface frameclock only contributes timing context.
        let now = host_time(started_at);
        let next_frame = self.plan_frame(now, next_demand);
        self.queue_frame(next_frame);
        self.arm_next_wake(started_at, event_loop);
    }
}

fn instant_for(started_at: Instant, deadline: TimerInstant) -> Instant {
    started_at + StdDuration::from_nanos(deadline)
}

fn demand_preempts_plan(pending: FrameDemand, planned: FrameDemand) -> bool {
    demand_priority(pending) > demand_priority(planned)
}

fn demand_priority(demand: FrameDemand) -> u8 {
    if demand.contains(FrameDemand::INPUT) {
        4
    } else if demand.contains(FrameDemand::CONTINUOUS_INPUT) {
        3
    } else if demand.contains(FrameDemand::ANIMATION) {
        2
    } else if demand.contains(FrameDemand::BACKGROUND) {
        1
    } else {
        0
    }
}

impl SurfaceFrameClock {
    fn new(output: OutputId) -> Self {
        Self {
            // Winit alone does not give us predicted presentation timestamps, so
            // this example uses pacing-only scheduling. Platform backends can
            // switch to predictive ticks when they have stronger display timing.
            //
            // Start this example at depth 1: without misses or pacing overruns,
            // there is no evidence that the app should add latency. The adaptive
            // policy can still raise depth if repeated overruns show that this is
            // too aggressive.
            scheduler: {
                let mut config = SchedulerConfig::pacing_only();
                config.initial_depth = 1;
                Scheduler::new(config)
            },
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
    fn render(&mut self, plan: &FramePlan, now: HostTime) -> SyntheticSubmission {
        // This example has no renderer, so it invents a short build/submit span.
        // A real app would measure actual CPU build start and queue submit time,
        // then attach platform present feedback when available.
        let budget = plan.commit_deadline.saturating_duration_since(now);
        let build_cost = Duration(SYNTHETIC_BUILD_COST.ticks().min(budget.ticks()));

        SyntheticSubmission {
            build_start: now,
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
