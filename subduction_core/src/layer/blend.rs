// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Blend mode for compositing layers.

/// Blend mode applied when compositing a layer over its backdrop.
///
/// Defaults to [`SourceOver`](Self::SourceOver), the standard
/// premultiplied alpha-over operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BlendMode {
    /// Standard source-over alpha compositing.
    #[default]
    SourceOver,
    /// Multiply blend.
    Multiply,
    /// Screen blend.
    Screen,
}
