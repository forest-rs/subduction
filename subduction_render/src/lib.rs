// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render-plan definitions and damage tracking for subduction.
//!
//! This crate provides the intermediate representation between
//! [`subduction_core`]'s layer tree evaluation and backend-specific
//! rendering. It defines:
//!
//! - [`RenderItem`] — a single draw command in the render plan
//! - [`RenderPlan`] — an ordered list of draw commands for one frame
//! - [`DamageRegion`] — spatial damage tracking for partial re-rendering
//! - [`ResourceKey`] — opaque handle for backend-managed resources

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

mod damage;
mod plan;
mod resource;

pub use damage::DamageRegion;
pub use plan::{RenderItem, RenderPlan};
pub use resource::ResourceKey;
pub use subduction_core::layer::BlendMode;
