// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render plan: an ordered sequence of draw items for one frame.

use alloc::vec::Vec;

use subduction_core::layer::{BlendMode, ClipShape, LayerId, SurfaceId};
use subduction_core::output::OutputId;

/// A single draw command in the render plan.
///
/// Items are produced in back-to-front order, matching the layer tree's
/// traversal order.
#[derive(Clone, Debug)]
pub struct RenderItem {
    /// The layer this item originates from.
    pub layer_id: LayerId,
    /// The surface to draw (if any — grouping nodes have `None`).
    pub surface: Option<SurfaceId>,
    /// World-space transform (column-major 4x4).
    pub world_transform: [f32; 16],
    /// Effective opacity (0.0–1.0, accumulated from ancestors).
    pub effective_opacity: f32,
    /// Clip shape in local coordinates, if any.
    pub clip: Option<ClipShape>,
    /// Blend mode.
    pub blend_mode: BlendMode,
}

/// An ordered list of draw commands for a single frame on a single output.
///
/// Backends translate this into native compositor operations or GPU draw
/// calls depending on their rendering strategy.
#[derive(Clone, Debug, Default)]
pub struct RenderPlan {
    /// Target output for this plan.
    pub output: OutputId,
    /// Draw items in back-to-front order.
    pub items: Vec<RenderItem>,
}

impl RenderPlan {
    /// Creates an empty render plan for the given output.
    #[must_use]
    pub fn new(output: OutputId) -> Self {
        Self {
            output,
            items: Vec::new(),
        }
    }

    /// Clears the plan for reuse.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}
