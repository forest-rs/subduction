// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Internal output tracking for `wl_output` globals.

use subduction_core::output::OutputId;
use wayland_client::protocol::wl_output;

/// A tracked `wl_output` global.
pub(crate) struct OutputEntry {
    pub(crate) id: OutputId,
    pub(crate) global_name: u32,
    pub(crate) proxy: wl_output::WlOutput,
}

/// Registry of bound `wl_output` globals with stable monotonic ID allocation.
///
/// Output IDs are never reused: after a display is removed the counter is not
/// decremented, so downstream code can rely on identity stability across
/// hotplug events.
pub(crate) struct OutputRegistry {
    entries: Vec<OutputEntry>, // small vec, typically 1-4 outputs
    next_id: u32,              // monotonic, never decremented
}

impl OutputRegistry {
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 0,
        }
    }

    /// Returns `true` if the given compositor global name is already tracked.
    pub(crate) fn contains_global(&self, global_name: u32) -> bool {
        self.entries.iter().any(|e| e.global_name == global_name)
    }

    /// Binds a new output global and allocates a stable [`OutputId`].
    ///
    /// # Panics
    ///
    /// Panics if `global_name` is already tracked (invariant violation in
    /// dispatch logic) or if the internal counter overflows `u32::MAX`.
    pub(crate) fn add(&mut self, global_name: u32, proxy: wl_output::WlOutput) -> OutputId {
        assert!(
            !self.contains_global(global_name),
            "duplicate wl_output global_name {global_name}"
        );
        let id = OutputId(
            self.next_id
                .checked_add(0)
                .expect("output ID counter exhausted"),
        );
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("output ID counter overflow");
        self.entries.push(OutputEntry {
            id,
            global_name,
            proxy,
        });
        id
    }

    /// Removes the output with the given compositor global name.
    ///
    /// Returns the entry so the caller can release the proxy if needed.
    /// The ID counter is **not** decremented.
    pub(crate) fn remove(&mut self, global_name: u32) -> Option<OutputEntry> {
        let pos = self
            .entries
            .iter()
            .position(|e| e.global_name == global_name)?;
        Some(self.entries.swap_remove(pos))
    }

    /// Looks up the stable [`OutputId`] for a compositor global name.
    #[allow(dead_code, reason = "used by future output routing")]
    pub(crate) fn id_for_global(&self, global_name: u32) -> Option<OutputId> {
        self.entries
            .iter()
            .find(|e| e.global_name == global_name)
            .map(|e| e.id)
    }

    #[allow(dead_code, reason = "used in later integration work")]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code, reason = "used in later integration work")]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for OutputRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for OutputRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputRegistry")
            .field("count", &self.entries.len())
            .field("next_id", &self.next_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::OutputRegistry;
    use subduction_core::output::OutputId;
    use wayland_client::protocol::wl_output;
    use wayland_client::{Connection, Proxy};

    /// Creates an inert `WlOutput` proxy for testing (no real compositor).
    fn inert_output() -> wl_output::WlOutput {
        let (s1, _s2) = std::os::unix::net::UnixStream::pair().unwrap();
        let conn = Connection::from_socket(s1).unwrap();
        wl_output::WlOutput::from_id(&conn, wayland_client::backend::ObjectId::null()).unwrap()
    }

    #[test]
    fn sequential_id_allocation() {
        let mut reg = OutputRegistry::new();
        assert_eq!(reg.add(1, inert_output()), OutputId(0));
        assert_eq!(reg.add(2, inert_output()), OutputId(1));
        assert_eq!(reg.add(3, inert_output()), OutputId(2));
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn non_reuse_after_removal() {
        let mut reg = OutputRegistry::new();
        reg.add(10, inert_output());
        reg.add(11, inert_output());
        reg.remove(10);
        assert_eq!(reg.add(12, inert_output()), OutputId(2));
    }

    #[test]
    fn remove_unknown_global_returns_none() {
        let mut reg = OutputRegistry::new();
        assert!(reg.remove(99).is_none());
    }

    #[test]
    fn empty_registry_state() {
        let reg = OutputRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn id_for_global_lookup() {
        let mut reg = OutputRegistry::new();
        reg.add(5, inert_output());
        reg.add(6, inert_output());

        assert_eq!(reg.id_for_global(5), Some(OutputId(0)));
        assert_eq!(reg.id_for_global(6), Some(OutputId(1)));
        assert_eq!(reg.id_for_global(7), None);
    }

    #[test]
    fn hotplug_sequence_preserves_monotonicity() {
        let mut reg = OutputRegistry::new();
        let id0 = reg.add(1, inert_output());
        let id1 = reg.add(2, inert_output());
        reg.remove(1);
        let id2 = reg.add(3, inert_output());
        reg.remove(2);
        let id3 = reg.add(4, inert_output());

        assert_eq!(id0, OutputId(0));
        assert_eq!(id1, OutputId(1));
        assert_eq!(id2, OutputId(2));
        assert_eq!(id3, OutputId(3));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn contains_global_tracks_presence() {
        let mut reg = OutputRegistry::new();
        assert!(!reg.contains_global(1));
        reg.add(1, inert_output());
        assert!(reg.contains_global(1));
        assert!(!reg.contains_global(2));
        reg.remove(1);
        assert!(!reg.contains_global(1));
    }

    #[test]
    #[should_panic(expected = "duplicate wl_output global_name 1")]
    fn add_duplicate_panics() {
        let mut reg = OutputRegistry::new();
        reg.add(1, inert_output());
        reg.add(1, inert_output());
    }
}
