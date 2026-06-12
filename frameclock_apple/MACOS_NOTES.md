# macOS / `frameclock_apple` — work notes for the next agent

Originally written at `76f7bf4`. **Updated at `a631e02`** — most of P0/P1 has
since landed. See the progress section directly below; the detailed items keep
their original text with a `✅ DONE` / `🔶 PARTIAL` / `⬜ TODO` marker.

These are notes for improving the macOS timing path. The retained adapter
(`AppleFrameClock`), deferred actual-present feedback, and the ProMotion cadence
writeback are now wired together and exercised by all three macOS examples. The
remaining work is the power/idle story, the off-device writeback-loop test, and
the CV cross-thread and docs items. Read this whole file before touching code;
several items interact.

## Progress (commits `9d9e554`, `c36b8e5`, `a631e02`)

- ✅ **Examples ported.** `macos_wgpu`, `macos_layers`, and `macos_lotta_layers`
  now drive `AppleFrameClock` (`begin_frame → match result → submit_frame_now`),
  use `AppleFeedbackMode::DeferredActualPresent`, and apply the ProMotion
  writeback via `clock.preferred_frame_rate_range(&frame)` →
  `link.set_preferred_frame_rate_range`. The hand-rolled `PendingFeedback`/
  `Scheduler` boilerplate is gone. (P0 headline)
- ✅ **Deferred-feedback `None` guard** (P0-1 core): `resolve_deferred_feedback`
  now returns early on `actual_present?`, so a tick without a present keeps the
  slot pending; a superseding deferred submission resolves the old one as
  commit-only `Unavailable`. Regression test added.
- ✅ **Feedback capability model** (P0-1/P0-2): `AppleFeedbackMode::{DeferredActualPresent, CommitOnly}`,
  CA defaults to deferred, CV to commit-only. `submit_frame_now` routes through it.
- ✅ **Configurable commit lead** (P1-1): `set_commit_lead` / `use_default_commit_lead`,
  default ≈ refresh/4 subtracted from predicted present, clamped.
- ✅ **Writeback glue on the adapter** (P1-2): `AppleFrameClock::preferred_frame_rate_range(&ActiveFrame)`
  removes the manual interval/timebase plumbing; examples consume it.
- ✅ **Pause controls exist** (P0-3, capability half): `is_paused`/`set_paused`
  on both CA and CV `DisplayLink`.

### Still open / re-scoped
- 🔶 **P0-3 demand-driven idle is still not demonstrated.** The controls exist,
  but all three examples `request(FrameDemand::ANIMATION)` every tick and never
  go idle, so nothing pauses the link to prove the power win. Either add an
  idle/resume cycle to one example, or have `AppleFrameClock` expose a "should
  the link be paused?" signal and wire it. This is the most valuable remaining item.
- 🔶 **P0-2 CV actual-present**: resolved *safely* (CV is now honestly
  `CommitOnly`) but CV is still not a predictive-feedback source. Fine unless the
  human wants CV upgraded — see open questions.
- ⬜ **P1-3 oscillation test**: no `frameclock_apple/tests/` exists yet. Still
  worth a deterministic, off-device writeback-loop test (interval → preferred
  rate → fed back as next `refresh_interval`, borderline build cost, assert no
  2-cycle). This is the one correctness check that can run on CI.
- ⬜ **P1-4** (CV cross-thread `tick.now` staleness), **P2-1** (mutually-exclusive
  features fail silently), **P2-2** (silent late-callback degradation) — unaddressed.
- ℹ️ Examples discard `FrameBegin::resolved_feedback`; the scheduler still
  observes internally, but no resolved summary reaches the HUD/diagnostics.
  Optional polish.

Scope: `frameclock_apple/` and `examples/macos_layers`, `examples/macos_lotta_layers`,
`examples/macos_wgpu`. Do **not** change the core `frameclock` crate's public
API as part of this unless an item below explicitly calls for it — the core
feedback/plan model is settled and other platforms depend on it.

Ground rules:
- Don't reattribute or rewrite existing commits (they're the human's, on `main`).
- Off-device CI can't run Objective-C. Every item below has a part that is
  testable without a device — do that part as a unit/integration test. The
  on-device part needs the human to run it; call that out explicitly in the PR.

---

## The big picture problem

`AppleFrameClock` (frameclock_apple/src/lib.rs:184) owns a `FrameDriver` and
exposes `begin_frame -> FrameBegin`, `submit_frame -> FrameSubmitResult`,
`request(demand)`, `set_display_timing`, and the deferred-feedback lifecycle.
This is the intended public surface.

But all three macOS examples still use the **old low-level path**:

- `macos_wgpu/src/main.rs:18-24,741,758,773-787,863` — `Scheduler::new`,
  `compute_present_hints`, `FrameOpportunity::new`, `scheduler.plan(...)`,
  `scheduler.observe(...)`, manual `PendingFeedback::new`/`resolve`.
- `macos_layers/src/main.rs:18-26,300,335-365,448` — same pattern.
- `macos_lotta_layers/src/main.rs:15-23,245,279-308,399` — same pattern.

Consequences:
1. The new adapter and deferred-feedback API have **zero** real users, so their
   bugs (below) haven't surfaced.
2. Every example unconditionally plans `FrameDemand::ANIMATION` every tick. The
   demand model (idle when nothing is animating, pause the link to save power)
   is never demonstrated and, as it turns out, *can't* be honored today (see P0-3).
3. The ProMotion writeback is never called, so the cadence the scheduler picks
   never reaches Core Animation.

**The single most valuable outcome of this work**: port at least one macOS
example (recommend `macos_wgpu`, since it's the realistic renderer path) to
`AppleFrameClock`, end to end — demand-driven, deferred feedback, and ProMotion
writeback — and make it the proving ground. Porting should *delete* a large
amount of duplicated boilerplate (the three examples currently repeat the
`PendingFeedback` dance almost verbatim); that deletion is itself the signal the
adapter is carrying its weight.

---

## P0 — correctness / must fix

### P0-1. `submit_frame_now` defers feedback that the CV path can never resolve
`AppleFrameClock::submit_frame_now` always uses `FrameSubmission::deferred(...)`
(frameclock_apple/src/lib.rs:276-278). Deferred feedback is resolved on the next
`begin_frame` from `tick.prev_actual_present` (frameclock/src/driver.rs:550-567).

But the **`CVDisplayLink` callback hard-codes `prev_actual_present: None`**
(frameclock_apple/src/cv_display_link.rs:194). So with `AppleFrameClock` on a CV
link:
- The deferred slot is filled on submit.
- The next tick carries `prev_actual_present = None`.
- `resolve_deferred_feedback` resolves **unconditionally** even when the value is
  `None` (driver.rs:554-560), producing commit-timing-only feedback **one frame
  late** instead of immediately — strictly worse than `Unavailable`.

Fix direction:
- The adapter must know whether its link reports previous-frame present. Don't
  blanket-defer. Options (pick one, note the trade-off in the PR):
  - Add a capability flag to `AppleFrameClock` (e.g. constructed as
    `AppleFrameClock::for_ca(...)` / `for_cv(...)`, or a
    `PresentReporting::{DeferredPrevFrame, None}` parameter) and have
    `submit_frame_now` choose `deferred` vs `Unavailable` accordingly.
  - Or drop `submit_frame_now` and make the example pass an explicit
    `FrameSubmission`.
- Separately, **make deferred resolution refuse to resolve against `None`**:
  in `FrameDriver::resolve_deferred_feedback`, only take the pending slot when
  `actual_present.is_some()`; otherwise leave it pending (and decide a bounded
  fallback — e.g. resolve as `Unavailable` if it's still pending after N ticks,
  or on discard). This protects every caller, not just Apple. Add a unit test in
  `frameclock` for "deferred submit, next begin has `prev_actual_present: None`".

### P0-2. `CVDisplayLink` never produces actual-present feedback at all
Even ignoring P0-1, the CV path emits `prev_actual_present: None` forever
(cv_display_link.rs:194), so the *entire* predictive feedback loop degrades to
commit-timing on CV. CV does not hand you the previous frame's *actual* present
directly the way CADisplayLink's `timestamp` does, so there are two honest
choices — do **not** launder the predicted `inOutputTime` as if it were actual
(that violates the crate's core "don't fake presentation truth" principle):
- **Preferred:** retain the previous callback's `inOutputTime.hostTime` (the
  predicted present we last reported) and the previous `inNow.hostTime`, and on
  each callback emit `prev_actual_present = previous inNow.hostTime` — i.e. the
  CV "now" sampled at the vsync after the frame was scheduled, which is a real
  measured host time close to the actual present. Document it as an
  approximation. This at least feeds *measured* timing back, not a prediction.
- **Acceptable:** keep `None` but document loudly that CV is commit-timing-only,
  and make `AppleFrameClock` use `Unavailable` (not `Deferred`) on CV (ties to
  P0-1).
Flag for the human which they prefer; lean toward the second (simpler, honest)
unless they want CV to be a first-class predictive path.

### P0-3. No way to pause the display link → demand-driven idle burns power
`FrameDriver` returns `FrameBeginResult::Idle` when there's no demand, which is
the whole point of the demand model. But the CA `DisplayLink` (current
frameclock_apple/src/ca_display_link.rs) exposes only `start`/`stop` and the
preferred-rate setters — **`is_paused`/`set_paused` are gone** (they existed in
the pre-split `subduction_backend_apple` version). So when the app is idle the
`CADisplayLink` keeps firing every vsync and the example keeps doing wasted
`begin_frame` work that returns `Idle`. CV similarly only has `start`/`stop`
(which return `Result`).

Fix direction:
- Re-add `set_paused(bool)` / `is_paused()` on the CA `DisplayLink`
  (`CADisplayLink.paused`), and a `start`/`stop` story for CV that's cheap to
  toggle.
- Decide who drives it. Cleanest: `AppleFrameClock` exposes the current demand
  state (it already proxies `has_pending_demand`), and the **example** pauses the
  link when `begin_frame` returns `Idle` and there's no pending demand, then
  resumes on `request(...)`. Don't bury link control inside `AppleFrameClock`
  unless it also owns the link (it currently does not — see P1-2 on the split).
- The ported example must demonstrate this: animate for a few seconds, go idle,
  show the link paused (CPU drops), then resume on input.

---

## P1 — correctness/quality

### P1-1. `present_hints` commit deadline is over-optimistic
`present_hints` sets `latest_commit = predicted_present` exactly
(frameclock_apple/src/lib.rs:78). That asserts you can commit right up to the
presentation instant with zero compositor lead time. On Apple the render server
needs the surface committed before its own deadline (typically a few ms before
scan-out). Using `predicted_present` as the commit boundary means the scheduler
only pulls `frame_start` earlier by its *learned build margin*, with no fixed
platform lead.

Fix direction: add a configurable "commit lead" (a `Duration`) subtracted from
`predicted_present` to form `latest_commit`, defaulting to something small and
documented (and clamped to `>= tick.now`). Expose it on `AppleFrameClock`
(constructor or setter). Keep `predicted_present` as `desired_present`. Unit-test
the clamp (`predicted_present - lead` floored at `now`).

### P1-2. ProMotion writeback is split across two objects with no glue
`AppleFrameClock` owns the `FrameDriver` (and thus the `FramePlan.frame_interval`
the scheduler chose), while the `DisplayLink` (which owns
`set_preferred_frame_interval`, ca_display_link.rs:182) is a *separate* object
the host holds. Nothing connects "the plan picked 60 Hz" to "tell the link to
prefer 60 Hz." The host must manually pull `frame.plan().frame_interval`, hold
the `display_timing`, call `link.set_preferred_frame_interval(...)`, and not
forget. That's exactly the kind of wiring that gets done wrong.

Fix direction (pick one, note trade-off):
- Add a convenience on `AppleFrameClock` that, given a ready `ActiveFrame` (or
  its plan), returns the `PreferredFrameRateRange` using the clock's own
  `display_timing` and timebase — so the example does
  `if let Some(r) = clock.preferred_range_for(&frame) { link.set_preferred_frame_rate_range(r); }`.
  Keeps link ownership in the host, removes the duplicated interval/timebase
  plumbing.
- Or let `AppleFrameClock` optionally hold a handle/closure to apply the rate,
  and write it back internally on `Ready`. More magic; only if the human wants
  the clock to own the link.
Either way, **the ported example must actually call the writeback** and the PR
should describe the on-device behavior (ProMotion stepping 120→60→… as demand
class / build cost changes).

### P1-3. Close the ProMotion feedback loop in a deterministic test
The writeback creates a loop the scheduler doesn't model: writeback changes the
next tick's `refresh_interval`/`duration`, which re-enters planning as
`source_interval` and can interact with the animation down-rate threshold
(`frame_interval` from `build_cost`, scheduler.rs). Risk: oscillation between two
rates (e.g. 60↔120) at a borderline build cost.

This is fully testable off-device using the platform-neutral
`preferred_frame_rate_range` (lib.rs:144) + the simulated scheduler: drive a loop
that (a) plans, (b) converts `frame_interval` → preferred rate, (c) feeds the
resulting interval back as the next tick's `refresh_interval`, (d) injects a
borderline/spiky build cost, and assert the rate converges (no 2-cycle). Put this
in `frameclock_apple/tests/` (pure-logic, no objc). If it *does* oscillate, that's
a finding for the core scheduler (it likely needs explicit hysteresis on the
down-rate decision) — report it, don't paper over it in the adapter.

### P1-4. CV cross-thread `tick.now` staleness
The CV callback samples `inNow.hostTime` on the high-priority CV background
thread (cv_display_link.rs:178), then `TickSender::send` dispatches the tick to
the main queue via `DispatchQueue::main().exec_async` (threading.rs:54-57). By
the time the main thread runs `begin_frame`, `tick.now` is already in the past by
the dispatch hop (can be a full frame under load). `predicted_present`
(`inOutputTime`) is a genuine future time so it's fine; only `now` is stale,
which skews `frame_start`/budget math.
- At minimum document this. Better: let the host (or adapter) re-read
  `DisplayLink::now()` on the main thread and use it as the opportunity's `now`
  when the tick arrived via the forwarder, keeping `predicted_present` from the
  CV sample. Decide and note the trade-off (re-reading now loses the exact vsync
  phase but fixes staleness).

---

## P2 — ergonomics / robustness / docs

### P2-1. Mutually-exclusive features fail silently
`default = ["ca-display-link"]`; the lib selects CV only via
`cfg(all(feature = "cv-display-link", not(feature = "ca-display-link")))`
(frameclock_apple/src/lib.rs:29-44). If a host enables *both* (e.g. via feature
unification across a workspace), CV silently disappears and CA wins with no
error. Add a `compile_error!` when both are set, or document the precedence
prominently. Confirm the iOS story (CADisplayLink) builds — the crate advertises
`ios` in keywords but it's unclear anything tests an iOS target.

### P2-2. CADisplayLink late-callback degradation is silent
When the runloop is blocked and the callback arrives after `targetTimestamp`,
`present_hints` drops the stale prediction and the frame silently becomes
pacing-only for that tick (lib.rs:74-79, the `predicted >= now` filter). Correct
behavior, but invisible. Add a lightweight counter/diagnostic (or surface it in
the example's HUD) so "we lost predictive timing for N frames" is observable.
This is the kind of thing that explains mysterious jank on a loaded main thread.

### P2-3. Reduce the triplicated example boilerplate
`macos_layers` and `macos_lotta_layers` are near-identical in their frameclock
wiring (and share with `macos_wgpu`). Once one example is ported to
`AppleFrameClock`, port the others and/or extract the shared host loop into a
small example-support module (there's precedent: `examples/lotta_layers_common`).
The win is concrete: the manual `PendingFeedback`/`compute_present_hints`/
`scheduler.plan`/`observe` block should collapse to `clock.begin_frame(tick)` +
match + `clock.submit_frame(...)`.

### P2-4. CADisplayLink `now` vs `timestamp`
The CA handler reads a fresh `mach_time::now()` for `tick.now`
(ca_display_link.rs:66) rather than the callback's `timestamp`. That's a
defensible choice (now = callback entry) but worth a one-line comment explaining
why, since the CV path uses the callback-provided `inNow` and the asymmetry is
otherwise confusing.

---

## Suggested sequencing

1. P0-1 + the `resolve-against-None` guard in core (small, unblocks everything,
   fully unit-testable in `frameclock`).
2. P0-3 (pause/resume) — needed before a demand-driven example makes sense.
3. Port `macos_wgpu` to `AppleFrameClock`, demand-driven, with P1-1 commit lead
   and P1-2 writeback glue. This is where the design gets validated.
4. P1-3 oscillation test (off-device) and P0-2 CV decision.
5. Port the remaining two examples / extract shared support (P2-3); docs (P2-2,
   P2-4, P2-1).

## Open questions for the human (ask before deciding)
- CV path: make it a real predictive source (P0-2 "preferred"), or accept it as
  commit-timing-only and route `Unavailable`? Affects API shape in P0-1.
- Should `AppleFrameClock` own the `DisplayLink` (so it can drive pause + rate
  writeback itself), or stay a pure planner with the host owning the link? P0-3,
  P1-2 both hinge on this. Current design is "pure planner"; the friction in
  P1-2/P0-3 is the cost of that choice.
- Is iOS actually a supported target to validate, or macOS-only for now?
