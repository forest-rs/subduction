// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Display output identification plus layer-root backdrop policy.
//!
//! [`OutputId`] is a lightweight handle identifying a specific display or
//! output surface. Backends assign these; core treats them as opaque.
//!
//! This module also defines backend-neutral layer-root semantics such as the
//! backdrop color. These settings belong to the root container the scene is
//! presented into, not to any particular layer in the scene tree.

use core::fmt;

use color::{AlphaColor, Srgb};

/// Identifies a specific display output or surface.
///
/// Backends assign output IDs to distinguish multiple displays. Core code
/// passes them through without interpreting the value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct OutputId(pub u32);

impl fmt::Debug for OutputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OutputId({})", self.0)
    }
}

/// Straight-alpha sRGB color used by layer-root backdrop policy.
///
/// This is the payload type for a solid backdrop. It is not, by itself,
/// enough to express whether an output should preserve transparency or
/// establish a real backdrop; that policy distinction lives in [`Backdrop`].
pub type Color = AlphaColor<Srgb>;

/// The layer-root backdrop policy applied behind all scene layers.
///
/// This is layer-root policy, not layer content: it cannot be transformed,
/// clipped, reordered, or hit-tested.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backdrop {
    /// There is no explicit scene backdrop.
    ///
    /// This is intentionally distinct from [`Backdrop::Color`] with a
    /// transparent color payload: callers are choosing the absence of a
    /// backdrop as policy rather than asking the presenter to establish a
    /// backdrop fill.
    None,
    /// Establish a scene backdrop by filling the output before presenting layers.
    ///
    /// The payload is the backdrop color in straight-alpha sRGB. Even if the
    /// color has partial or zero alpha, this variant still means “this scene
    /// has an explicit backdrop fill,” not “preserve transparency.”
    Color(Color),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backdrop_none_is_distinct_from_color() {
        assert_ne!(
            Backdrop::None,
            Backdrop::Color(Color::from_rgba8(0x00, 0x00, 0x00, 0x00))
        );
    }

    #[test]
    fn backdrop_can_hold_a_color() {
        let color = Color::from_rgba8(0x1e, 0x1e, 0x2e, 0xff);
        assert_eq!(Backdrop::Color(color), Backdrop::Color(color));
    }
}
