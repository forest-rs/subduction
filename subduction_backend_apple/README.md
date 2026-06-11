<div align="center">

# Subduction Backend Apple

**Apple backend for Subduction.**

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/forest-rs/subduction/ci.yml?logo=github&label=CI)](https://github.com/forest-rs/subduction/actions)

</div>

`subduction_backend_apple` provides composable building blocks for driving a
Subduction layer tree on Apple platforms. It includes Core Animation layer
presenters, Metal layer presentation, and AppKit view integration helpers.

Apple frame timing lives in `frameclock_apple`. Use that crate for
`CADisplayLink` / `CVDisplayLink` ticks, Mach host-time conversion, and
retained `frameclock` driver integration.

## Feature Flags

- `appkit`: enables AppKit view integration types.

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
