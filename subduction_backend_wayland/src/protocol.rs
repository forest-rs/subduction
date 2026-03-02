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
//! Both integration modes wire through here via `delegate_dispatch!`:
//!
//! ```text
//! WaylandProtocol               (generic Dispatch impls, D: AsMut<WaylandState>)
//!   ^  delegate_dispatch!
//! WaylandState                   (owned mode, concrete impl, no cycle)
//! HostState                      (embedded mode, same delegation)
//! ```

use crate::commit::FeedbackData;
use crate::event_loop::WaylandState;
use crate::presentation::{PresentEvent, presentation_time_to_host_time};
use crate::time::clock_from_presentation_clk_id;
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::{wl_callback, wl_output, wl_registry};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::presentation_time::client::{wp_presentation, wp_presentation_feedback};

/// Maximum `wl_output` version the backend will bind.
pub(crate) const WL_OUTPUT_MAX_VERSION: u32 = 4;

/// Maximum `wp_presentation` version the backend will bind.
const WP_PRESENTATION_VERSION: u32 = 1;

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
/// [`delegate_dispatch!`](wayland_client::delegate_dispatch).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputGlobalData {
    pub(crate) global_name: u32,
}

/// Userdata marker for backend-issued frame callbacks.
///
/// Distinguishes backend callbacks from host/toolkit callbacks on the same
/// queue. Public because embedded-mode hosts need it in
/// [`delegate_dispatch!`](wayland_client::delegate_dispatch).
#[derive(Debug, Clone, Copy)]
pub struct FrameCallbackData;

/// Delegation target for Wayland protocol event dispatch.
///
/// Use with [`delegate_dispatch!`](wayland_client::delegate_dispatch) to wire
/// protocol handling for [`WaylandState`] into an application state type.
/// See the [crate-level docs](crate) for wiring examples.
#[derive(Debug)]
pub struct WaylandProtocol;

// ---------------------------------------------------------------------------
// Dispatch<WlRegistry, (), D>
// ---------------------------------------------------------------------------

impl<D> Dispatch<WlRegistry, (), D> for WaylandProtocol
where
    D: Dispatch<WlRegistry, ()>
        + Dispatch<wl_output::WlOutput, OutputGlobalData>
        + Dispatch<wp_presentation::WpPresentation, ()>
        + AsMut<WaylandState>
        + 'static,
{
    fn event(
        state: &mut D,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<D>,
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
                } else if interface == wp_presentation::WpPresentation::interface().name {
                    if ws.presentation.is_some() {
                        return;
                    }
                    let v = version.min(WP_PRESENTATION_VERSION);
                    let proxy: wp_presentation::WpPresentation = registry.bind(name, v, qh, ());
                    ws.presentation = Some(proxy);
                    ws.capabilities.has_presentation_time = true;
                }
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
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
        // No-op. Output property events handled in a future commit.
    }
}

// ---------------------------------------------------------------------------
// Dispatch<WpPresentation, (), D>
// ---------------------------------------------------------------------------

impl<D> Dispatch<wp_presentation::WpPresentation, (), D> for WaylandProtocol
where
    D: Dispatch<wp_presentation::WpPresentation, ()> + AsMut<WaylandState> + 'static,
{
    fn event(
        state: &mut D,
        _proxy: &wp_presentation::WpPresentation,
        event: wp_presentation::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
        let ws: &mut WaylandState = state.as_mut();
        if let wp_presentation::Event::ClockId { clk_id } = event {
            if let Some(clock) = clock_from_presentation_clk_id(clk_id) {
                ws.clock = clock;
                ws.capabilities.presentation_clock_domain_aligned = true;
            } else {
                ws.capabilities.presentation_clock_domain_aligned = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch<WlCallback, FrameCallbackData, D>
// ---------------------------------------------------------------------------

impl<D> Dispatch<wl_callback::WlCallback, FrameCallbackData, D> for WaylandProtocol
where
    D: Dispatch<wl_callback::WlCallback, FrameCallbackData> + AsMut<WaylandState> + 'static,
{
    fn event(
        state: &mut D,
        _proxy: &wl_callback::WlCallback,
        event: wl_callback::Event,
        _data: &FrameCallbackData,
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            let ws: &mut WaylandState = state.as_mut();
            ws.ticker.on_callback_done(ws.clock, &ws.output_registry);
        }
        // The callback_data field is a millisecond timestamp from an
        // unspecified epoch — not safely comparable to HostTime or
        // presentation feedback timestamps. We use Clock::now() instead.
    }
}

// ---------------------------------------------------------------------------
// Dispatch<WpPresentationFeedback, FeedbackData, D>
// ---------------------------------------------------------------------------

impl<D> Dispatch<wp_presentation_feedback::WpPresentationFeedback, FeedbackData, D>
    for WaylandProtocol
where
    D: Dispatch<wp_presentation_feedback::WpPresentationFeedback, FeedbackData>
        + AsMut<WaylandState>
        + 'static,
{
    fn event(
        state: &mut D,
        _proxy: &wp_presentation_feedback::WpPresentationFeedback,
        event: wp_presentation_feedback::Event,
        data: &FeedbackData,
        _conn: &Connection,
        _qh: &QueueHandle<D>,
    ) {
        let ws: &mut WaylandState = state.as_mut();
        let id = data.submission_id;
        match event {
            wp_presentation_feedback::Event::SyncOutput { output } => {
                let resolved = ws.output_registry.id_for_proxy(&output);
                if let Some(pending) = ws.pending_feedback.get_mut(&id) {
                    // "Known beats unknown": only overwrite if the new lookup
                    // resolves to Some, or if no value was stored yet.
                    if resolved.is_some() || pending.sync_output.is_none() {
                        pending.sync_output = resolved;
                    }
                }
            }
            wp_presentation_feedback::Event::Presented {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
                refresh,
                flags,
                ..
            } => {
                let actual_present = presentation_time_to_host_time(tv_sec_hi, tv_sec_lo, tv_nsec);
                let refresh_interval = if refresh == 0 {
                    None
                } else {
                    Some(u64::from(refresh))
                };
                let raw_flags = match flags {
                    WEnum::Value(k) => k.bits(),
                    WEnum::Unknown(v) => v,
                };
                let output = ws.pending_feedback.remove(&id).and_then(|p| p.sync_output);
                ws.present_events.push(PresentEvent::Presented {
                    id,
                    actual_present,
                    refresh_interval,
                    output,
                    flags: raw_flags,
                });
                ws.ticker.set_last_observed_actual_present(actual_present);
                ws.commit.decrement_pending();
            }
            wp_presentation_feedback::Event::Discarded => {
                let _ = ws.pending_feedback.remove(&id);
                ws.present_events.push(PresentEvent::Discarded { id });
                ws.commit.decrement_pending();
            }
            _ => {} // Event enum is #[non_exhaustive]
        }
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
