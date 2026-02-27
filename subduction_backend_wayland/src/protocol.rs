// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland protocol dispatch delegation and capability tracking.
//!
//! This module provides [`WaylandProtocol`], a zero-size delegate type that
//! holds the generic [`Dispatch`] implementations for Wayland protocol objects
//! managed by the backend. Using a separate delegate type avoids the
//! trait-resolution cycle that arises when generic `Dispatch` impls live
//! directly on [`WaylandState`].
//!
//! Both integration modes wire through here via [`delegate_dispatch!`]:
//!
//! ```text
//! WaylandProtocol               (generic Dispatch impls, D: AsMut<WaylandState>)
//!   ^  delegate_dispatch!
//! WaylandState                   (owned mode, concrete impl, no cycle)
//! HostState                      (embedded mode, same delegation)
//! ```

use crate::event_loop::WaylandState;
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::{wl_output, wl_registry};
use wayland_client::{Dispatch, Proxy};

/// Maximum `wl_output` version the backend will bind.
pub(crate) const WL_OUTPUT_MAX_VERSION: u32 = 4;

/// Runtime protocol capability flags.
///
/// Query via [`WaylandState::capabilities`] or
/// [`OwnedQueueMode::capabilities`](crate::OwnedQueueMode::capabilities).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Capabilities {
    /// `true` once `wp_presentation` has been bound.
    pub has_presentation_time: bool,
    /// `true` if the compositor's presentation clock matches the backend clock
    /// domain.
    pub presentation_clock_domain_aligned: bool,
}

impl Capabilities {
    /// All capabilities start as unavailable.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            has_presentation_time: false,
            presentation_clock_domain_aligned: false,
        }
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::new()
    }
}

/// User data attached to each bound `wl_output` proxy.
///
/// Public because embedded-mode hosts need it as a type parameter in
/// [`delegate_dispatch!`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputGlobalData {
    pub(crate) global_name: u32,
}

/// Delegation target for Wayland protocol event dispatch.
///
/// Use with [`delegate_dispatch!`](wayland_client::delegate_dispatch) to wire
/// protocol handling for [`WaylandState`] into an application state type.
/// See the [event-loop module](crate::event_loop) docs for wiring examples.
#[derive(Debug)]
pub struct WaylandProtocol;

// ---------------------------------------------------------------------------
// Dispatch<WlRegistry, (), D>
// ---------------------------------------------------------------------------

impl<D> Dispatch<WlRegistry, (), D> for WaylandProtocol
where
    D: Dispatch<WlRegistry, ()>
        + Dispatch<wl_output::WlOutput, OutputGlobalData>
        + AsMut<WaylandState>
        + 'static,
{
    fn event(
        state: &mut D,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _data: &(),
        _conn: &wayland_client::Connection,
        qh: &wayland_client::QueueHandle<D>,
    ) {
        let ws: &mut WaylandState = state.as_mut();
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } => {
                if interface == wl_output::WlOutput::interface().name {
                    if ws.output_registry.contains_global(name) {
                        return;
                    }
                    let v = version.min(WL_OUTPUT_MAX_VERSION);
                    let proxy: wl_output::WlOutput =
                        registry.bind(name, v, qh, OutputGlobalData { global_name: name });
                    ws.output_registry.add(name, proxy);
                }
                // wp_presentation detection deferred to future commit
            }
            wl_registry::Event::GlobalRemove { name } => {
                if let Some(entry) = ws.output_registry.remove(name)
                    && entry.proxy.version() >= 3
                {
                    entry.proxy.release();
                }
                // Proxy dropped if present; OutputId is never reused.
            }
            _ => {} // Event enum is #[non_exhaustive]
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch<WlOutput, OutputGlobalData, D>
// ---------------------------------------------------------------------------

impl<D> Dispatch<wl_output::WlOutput, OutputGlobalData, D> for WaylandProtocol
where
    D: Dispatch<wl_output::WlOutput, OutputGlobalData> + AsMut<WaylandState> + 'static,
{
    fn event(
        _state: &mut D,
        _proxy: &wl_output::WlOutput,
        _event: wl_output::Event,
        _data: &OutputGlobalData,
        _conn: &wayland_client::Connection,
        _qh: &wayland_client::QueueHandle<D>,
    ) {
        // No-op. Output property events handled in a future commit.
    }
}

#[cfg(test)]
mod tests {
    use super::Capabilities;

    #[test]
    fn capabilities_new_all_false() {
        let caps = Capabilities::new();
        assert!(!caps.has_presentation_time);
        assert!(!caps.presentation_clock_domain_aligned);
    }

    #[test]
    fn capabilities_new_eq_default() {
        assert_eq!(Capabilities::new(), Capabilities::default());
    }
}
