// Copyright 2026 the Frameclock Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Display output identification.
//!
//! [`OutputId`] is a lightweight handle identifying the display output or
//! presentation surface a frame tick targets. Platform adapters assign these
//! identifiers; `frameclock` treats them as opaque.

use core::fmt;

/// Identifies a display output or presentation surface.
///
/// Platform adapters assign output IDs to distinguish multiple displays or
/// surfaces. The scheduler passes them through without interpreting the value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct OutputId(pub u32);

impl fmt::Debug for OutputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OutputId({})", self.0)
    }
}
