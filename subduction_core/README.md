<div align="center">

# Subduction Core

**Retained layer tree and presenter contract for Subduction compositors.**

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/forest-rs/subduction/ci.yml?logo=github&label=CI)](https://github.com/forest-rs/subduction/actions)

</div>

`subduction_core` provides the retained layer tree, dirty evaluation, transform
math, output policy, tracing hooks, and backend presenter contract used by
Subduction compositors.

Display-frame timing, scheduling, present feedback, and affine timeline mapping
live in the sibling `frameclock` crate. `subduction_core` keeps compatibility
re-exports for the old timing module paths while local callers migrate to direct
`frameclock` imports.

For timing callers, the main source changes are:

- import common host types from `frameclock::{FrameTick, SchedulerConfig, ...}`
  and lower-level types such as `Scheduler` and `FramePlan` from
  `frameclock::scheduler` and `frameclock::timing` instead of
  `subduction_core::{timing, scheduler, time, clock}`;
- use `FramePlan::sample_time` instead of `FramePlan::semantic_time`;
- use `FramePlan::target_present` instead of `FramePlan::present_time`;
- choose scheduler presets by timing capability:
  `predictive()`, `estimated()`, or `pacing_only()`.

The crate is `no_std` by default and uses `alloc`. Enable the `std` feature
when integrating with backends or host code that wants standard-library
support in dependencies.

## Feature Flags

- `std`: enables standard-library support in dependencies and in `frameclock`.
- `trace`: enables low-cost trace call sites.
- `trace-rich`: enables detailed per-layer trace events and implies `trace`.

## Minimum supported Rust Version (MSRV)

This crate has been verified to compile with **Rust 1.92** and later.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE] or <http://www.apache.org/licenses/LICENSE-2.0>), or
- MIT license ([LICENSE-MIT] or <http://opensource.org/licenses/MIT>),

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[LICENSE-APACHE]: https://github.com/forest-rs/subduction/blob/main/LICENSE-APACHE
[LICENSE-MIT]: https://github.com/forest-rs/subduction/blob/main/LICENSE-MIT
[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[AUTHORS]: https://github.com/forest-rs/subduction/blob/main/AUTHORS
