// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! xdg-shell window management for Wayland examples.
//!
//! This module provides [`ExampleState`] — a combined state type for the
//! embedded-mode backend integration — along with xdg-shell window creation
//! and all necessary Wayland dispatch wiring.

use wayland_client::protocol::{
    wl_buffer, wl_callback, wl_compositor, wl_output, wl_registry, wl_shm, wl_shm_pool,
    wl_subcompositor, wl_subsurface, wl_surface,
};
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols::wp::presentation_time::client::{wp_presentation, wp_presentation_feedback};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

use subduction_backend_wayland::{
    FeedbackData, FrameCallbackData, LayerSubsurfaceData, LayerSurfaceData, OutputGlobalData,
    WaylandProtocol, WaylandState,
};

use crate::shm::ShmDispatch;

/// xdg-shell state tracked during global binding and configure events.
#[derive(Debug)]
pub struct XdgState {
    /// The `wl_shm` global, bound during registry events.
    pub shm: Option<wl_shm::WlShm>,
    /// The `xdg_wm_base` global.
    pub wm_base: Option<xdg_wm_base::XdgWmBase>,
    /// Set to `true` when the first `xdg_surface.configure` is acked.
    pub configured: bool,
    /// Current window width from configure events (0 = use default).
    pub width: u32,
    /// Current window height from configure events (0 = use default).
    pub height: u32,
    /// Set to `true` on each configure; cleared by the application after
    /// re-creating the root surface buffer.
    pub needs_redraw: bool,
}

impl XdgState {
    /// Creates empty xdg state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shm: None,
            wm_base: None,
            configured: false,
            width: 0,
            height: 0,
            needs_redraw: false,
        }
    }
}

impl Default for XdgState {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience bundle for the xdg-shell window objects.
#[derive(Debug)]
pub struct XdgWindow {
    /// The root `wl_surface`.
    pub surface: wl_surface::WlSurface,
    /// The `xdg_surface` role.
    #[allow(dead_code, reason = "must stay alive for the xdg role")]
    pub xdg_surface: xdg_surface::XdgSurface,
    /// The `xdg_toplevel` role.
    #[allow(dead_code, reason = "must stay alive for the xdg role")]
    pub toplevel: xdg_toplevel::XdgToplevel,
}

/// Combined application state for Wayland examples.
///
/// Contains the subduction backend state and xdg-shell state. Implements
/// all required `Dispatch` traits for both the backend and example needs.
#[derive(Debug)]
pub struct ExampleState {
    /// The subduction Wayland backend state.
    pub wayland: WaylandState,
    /// xdg-shell and `wl_shm` state.
    pub xdg: XdgState,
    /// Whether the application should exit.
    pub running: bool,
}

impl ExampleState {
    /// Creates a new `ExampleState` with empty backend and xdg state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            wayland: WaylandState::new(),
            xdg: XdgState::new(),
            running: true,
        }
    }
}

impl Default for ExampleState {
    fn default() -> Self {
        Self::new()
    }
}

impl AsMut<WaylandState> for ExampleState {
    fn as_mut(&mut self) -> &mut WaylandState {
        &mut self.wayland
    }
}

/// Creates an xdg toplevel window and registers it with the backend.
///
/// Returns the window objects. The caller must perform a roundtrip to
/// receive the initial configure event before the frame loop can start.
pub fn create_window(
    state: &mut ExampleState,
    qh: &QueueHandle<ExampleState>,
    title: &str,
    width: i32,
    height: i32,
) -> XdgWindow {
    let compositor = state
        .wayland
        .compositor()
        .expect("wl_compositor not bound")
        .clone();
    let wm_base = state
        .xdg
        .wm_base
        .as_ref()
        .expect("xdg_wm_base not bound")
        .clone();

    let surface = compositor.create_surface(qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, qh, ());
    let toplevel = xdg_surface.get_toplevel(qh, ());

    toplevel.set_title(title.to_string());
    toplevel.set_min_size(width, height);

    // Register with the backend for frame callbacks.
    state
        .wayland
        .set_surface(surface.clone())
        .expect("surface already set");

    // Initial commit to trigger the configure sequence.
    surface.commit();

    XdgWindow {
        surface,
        xdg_surface,
        toplevel,
    }
}

// ---------------------------------------------------------------------------
// Custom WlRegistry dispatch — binds host globals + forwards to backend
//
// Both the host and the backend need to react to the same registry events:
// the host binds xdg_wm_base and wl_shm for windowing and buffer allocation,
// while the backend binds wl_output, wl_compositor, wl_subcompositor, and
// wp_presentation for output tracking, compositing, and pacing.
//
// Since delegate_dispatch! forwards events to a single delegate with no hook
// for the host to intercept them, we implement Dispatch manually so we can:
//   1. Bind host-specific globals (xdg_wm_base, wl_shm) ourselves.
//   2. Forward every other event to the backend's WaylandProtocol handler,
//      which binds the globals it cares about.
// ---------------------------------------------------------------------------

impl Dispatch<wl_registry::WlRegistry, ()> for ExampleState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _data: &(),
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = &event
        {
            match interface.as_str() {
                "xdg_wm_base" => {
                    let wm_base = registry.bind::<xdg_wm_base::XdgWmBase, _, _>(
                        *name,
                        (*version).min(6),
                        qh,
                        (),
                    );
                    state.xdg.wm_base = Some(wm_base);
                    return;
                }
                "wl_shm" => {
                    let shm =
                        registry.bind::<wl_shm::WlShm, _, _>(*name, (*version).min(1), qh, ());
                    state.xdg.shm = Some(shm);
                    return;
                }
                _ => {}
            }
        }
        // Forward everything else to the backend.
        <WaylandProtocol as Dispatch<wl_registry::WlRegistry, (), Self>>::event(
            state, registry, event, _data, conn, qh,
        );
    }
}

// ---------------------------------------------------------------------------
// Host-only protocol dispatch (xdg-shell windowing)
//
// wayland-client requires a Dispatch impl for every object type created on
// the queue. These are windowing objects the backend doesn't know about:
// xdg_wm_base (compositor ping/pong keepalive), xdg_surface (configure ack),
// and xdg_toplevel (resize dimensions and close events).
// ---------------------------------------------------------------------------

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for ExampleState {
    fn event(
        _state: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for ExampleState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            state.xdg.configured = true;
            state.xdg.needs_redraw = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for ExampleState {
    fn event(
        state: &mut Self,
        _toplevel: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure {
                width,
                height,
                states: _,
            } => {
                if width > 0 && height > 0 {
                    #[expect(
                        clippy::cast_sign_loss,
                        reason = "configure dimensions are always non-negative"
                    )]
                    {
                        state.xdg.width = width as u32;
                        state.xdg.height = height as u32;
                    }
                }
            }
            xdg_toplevel::Event::Close => {
                state.running = false;
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Root wl_surface dispatch (host-created with () user data)
//
// The root wl_surface is created by the host with () user data, while the
// backend's WaylandPresenter creates child surfaces with LayerSurfaceData.
// wayland-client dispatches based on the user data type, so both Dispatch
// impls coexist on the same queue without conflict.
// ---------------------------------------------------------------------------

impl Dispatch<wl_surface::WlSurface, ()> for ExampleState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_surface::WlSurface,
        _event: wl_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// SHM dispatch delegation (host-only buffer allocation)
//
// wl_shm, wl_shm_pool, and wl_buffer are host-only objects used for shared-
// memory buffer allocation. Their events are no-ops (wl_shm.format is handled
// implicitly, wl_buffer.release is unused for static buffers), so they are
// delegated to the ShmDispatch marker type in shm.rs.
// ---------------------------------------------------------------------------

wayland_client::delegate_dispatch!(ExampleState:
    [wl_shm::WlShm: ()] => ShmDispatch);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_shm_pool::WlShmPool: ()] => ShmDispatch);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_buffer::WlBuffer: ()] => ShmDispatch);

// ---------------------------------------------------------------------------
// Backend protocol delegation (embedded-state mode)
//
// These protocols are consumed by the backend — the host never needs to
// intercept output hotplug, presentation clock, frame callback, feedback,
// compositor, or subcompositor events — so plain delegate_dispatch!
// forwarding to WaylandProtocol is sufficient.
//
// WlRegistry is intentionally NOT delegated here — see the manual Dispatch
// impl above, which intercepts host globals before forwarding to the backend.
//
// The WlSurface (LayerSurfaceData) and WlSubsurface (LayerSubsurfaceData)
// delegations are required when using WaylandPresenter, which creates a
// child wl_surface + wl_subsurface pair for each layer in the tree.
// ---------------------------------------------------------------------------

wayland_client::delegate_dispatch!(ExampleState:
    [wl_output::WlOutput: OutputGlobalData] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wp_presentation::WpPresentation: ()] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_callback::WlCallback: FrameCallbackData] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wp_presentation_feedback::WpPresentationFeedback: FeedbackData] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_compositor::WlCompositor: ()] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_subcompositor::WlSubcompositor: ()] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_surface::WlSurface: LayerSurfaceData] => WaylandProtocol);
wayland_client::delegate_dispatch!(ExampleState:
    [wl_subsurface::WlSubsurface: LayerSubsurfaceData] => WaylandProtocol);
