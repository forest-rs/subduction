// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Presenting X11 compositor: owns a top-level window and a wgpu surface, and
//! composites the subduction layer tree onto it by reusing the fallback
//! compositor in `subduction_backend_wgpu`.

#![expect(
    unsafe_code,
    reason = "wgpu surface creation from raw X11 handles requires unsafe"
)]

use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::num::NonZeroU32;
use std::ptr::NonNull;

use frameclock::FrameTick;
use subduction_backend_wgpu::{LayerRoot, WgpuPresenter, WgpuSurfaceTarget};
use subduction_core::backend::Presenter;
use subduction_core::layer::{FrameChanges, LayerStore, SurfaceId};
use x11rb::connection::Connection;
use x11rb::protocol::Event as X11Event;
use x11rb::protocol::present::{self, CompleteKind, ConnectionExt as _};
use x11rb::protocol::xproto::{
    self, AtomEnum, ConnectionExt as _, CreateWindowAux, EventMask, PropMode, WindowClass,
};
use x11rb::wrapper::ConnectionExt as _;
use x11rb::xcb_ffi::XCBConnection;

use crate::present_source::X11PresentSource;

/// Configuration for the compositor's window.
#[derive(Clone, Debug)]
pub struct X11WindowConfig {
    /// The window title.
    pub title: String,
    /// The initial window size in physical pixels.
    pub size: (u32, u32),
}

impl Default for X11WindowConfig {
    fn default() -> Self {
        Self {
            title: String::from("Subduction"),
            size: (1024, 768),
        }
    }
}

/// Errors from bringing up or driving the X11 compositor.
#[derive(Debug)]
pub enum X11BackendError {
    /// A required X extension is absent — this no-legacy backend has no fallback.
    MissingExtension(&'static str),
    /// The X connection or a request failed.
    X11(Box<dyn Error + Send + Sync>),
    /// A wgpu adapter, device, or surface could not be created.
    Wgpu(Box<dyn Error + Send + Sync>),
}

impl fmt::Display for X11BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingExtension(name) => {
                write!(f, "required X extension `{name}` is unavailable")
            }
            Self::X11(err) => write!(f, "X11 error: {err}"),
            Self::Wgpu(err) => write!(f, "wgpu error: {err}"),
        }
    }
}

impl Error for X11BackendError {}

impl From<x11rb::errors::ConnectError> for X11BackendError {
    fn from(err: x11rb::errors::ConnectError) -> Self {
        Self::X11(Box::new(err))
    }
}

impl From<x11rb::errors::ConnectionError> for X11BackendError {
    fn from(err: x11rb::errors::ConnectionError) -> Self {
        Self::X11(Box::new(err))
    }
}

impl From<x11rb::errors::ReplyError> for X11BackendError {
    fn from(err: x11rb::errors::ReplyError) -> Self {
        Self::X11(Box::new(err))
    }
}

impl From<x11rb::errors::ReplyOrIdError> for X11BackendError {
    fn from(err: x11rb::errors::ReplyOrIdError) -> Self {
        Self::X11(Box::new(err))
    }
}

type Result<T> = core::result::Result<T, X11BackendError>;

/// A presenting subduction backend for a top-level X11 window.
///
/// It owns the X11 connection and window, the wgpu surface it presents to, the
/// [`WgpuPresenter`] that composites the layer tree, and the
/// [`X11PresentSource`] that paces frames from the window's vblank clock.
///
/// Implement a frame loop by feeding decoded X11 events to
/// [`handle_event`](Self::handle_event), rendering app content into the surface
/// targets from [`target_for_surface`](Self::target_for_surface), calling
/// [`apply`](Presenter::apply) with the evaluated [`FrameChanges`], and then
/// [`present`](Self::present).
pub struct X11Compositor {
    conn: XCBConnection,
    window: u32,
    wm_delete_window: xproto::Atom,
    present_eid: u32,
    present_serial: u32,
    closed: bool,
    configured_size: (u32, u32),
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    presenter: WgpuPresenter,
    source: X11PresentSource,
}

impl fmt::Debug for X11Compositor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("X11Compositor")
            .field("window", &self.window)
            .field("configured_size", &self.configured_size)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl X11Compositor {
    /// Connects to the X server, opens a window, and builds the wgpu compositor.
    pub fn new(config: &X11WindowConfig) -> Result<Self> {
        let (conn, screen_num) = XCBConnection::connect(None)?;

        let (window, wm_delete_window, size) = create_window(&conn, screen_num, config)?;
        let present_eid = configure_present(&conn, window)?;

        let (width, height) = size;
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let surface = create_surface(&instance, &conn, screen_num, window)?;

        let (device, queue, surface_config) =
            configure_surface(&instance, &surface, width, height)?;
        let root = LayerRoot::new(surface_config.format, (width, height));
        let presenter = WgpuPresenter::new(device, queue, root, (width, height));

        Ok(Self {
            conn,
            window,
            wm_delete_window,
            present_eid,
            present_serial: 0,
            closed: false,
            configured_size: (width, height),
            surface,
            surface_config,
            presenter,
            source: X11PresentSource::new(),
        })
    }

    /// Returns the wgpu device the compositor renders with.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        self.presenter.device()
    }

    /// Returns the wgpu queue the compositor submits with.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        self.presenter.queue()
    }

    /// Returns the render target the app should draw `surface_id`'s content into.
    ///
    /// Targets are allocated when [`apply`](Presenter::apply) first observes the
    /// surface attached as layer content.
    #[must_use]
    pub fn target_for_surface(&self, surface_id: SurfaceId) -> Option<WgpuSurfaceTarget<'_>> {
        self.presenter.target_for_surface(surface_id)
    }

    /// Returns whether the window manager has asked the window to close.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Claims the vblank-notify slot and arms `PresentNotifyMSC` if free.
    ///
    /// Call to keep the vblank stream alive while frames are wanted; it is a
    /// no-op while a notify is already in flight.
    pub fn arm_vblank_notify(&mut self) -> Result<()> {
        if self.source.mark_notify_requested() {
            self.present_serial = self.present_serial.wrapping_add(1);
            self.conn
                .present_notify_msc(self.window, self.present_serial, 0, 1, 0)?;
            self.conn.flush()?;
        }
        Ok(())
    }

    /// Pops the next queued frame tick, if any.
    pub fn poll_tick(&mut self) -> Option<FrameTick> {
        self.source.poll_tick()
    }

    /// Routes one decoded X11 event into the present source and lifecycle state.
    pub fn handle_event(&mut self, event: &X11Event) {
        match event {
            X11Event::PresentCompleteNotify(complete) if complete.event == self.present_eid => {
                if complete.kind == CompleteKind::NOTIFY_MSC {
                    self.source.on_notify_complete(complete.ust, complete.msc);
                } else {
                    // A per-pixmap completion: feed it as submission feedback,
                    // keyed by the serial the presenter stamped.
                    self.source.on_pixmap_complete(
                        complete.serial,
                        complete.ust,
                        complete.msc,
                        complete.mode.into(),
                    );
                }
            }
            X11Event::PresentIdleNotify(idle) if idle.event == self.present_eid => {
                self.source.on_idle(idle.serial);
            }
            X11Event::ConfigureNotify(configure) => {
                self.configured_size = (u32::from(configure.width), u32::from(configure.height));
            }
            X11Event::ClientMessage(message)
                if message.data.as_data32()[0] == self.wm_delete_window =>
            {
                self.closed = true;
            }
            _ => {}
        }
    }

    /// Reconciles the surface with the latest configured window size.
    ///
    /// Returns whether the size changed and the surface was reconfigured.
    pub fn reconcile_size(&mut self) -> bool {
        let (width, height) = self.configured_size;
        if width == 0
            || height == 0
            || (width, height) == (self.surface_config.width, self.surface_config.height)
        {
            return false;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface
            .configure(self.presenter.device(), &self.surface_config);
        self.presenter.root_mut().resize(width, height);
        true
    }

    /// Composites the current layer tree onto the window and presents it.
    ///
    /// Call after rendering app content into the surface targets and after
    /// [`apply`](Presenter::apply). Acquires the swapchain image, composites into
    /// it, submits, and presents; a lost or outdated surface is reconfigured and
    /// the frame skipped.
    pub fn present(&mut self, store: &LayerStore) -> Result<()> {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(output)
            | wgpu::CurrentSurfaceTexture::Suboptimal(output) => output,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(self.presenter.device(), &self.surface_config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(X11BackendError::Wgpu(
                    "surface texture acquisition failed validation".into(),
                ));
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let command_buffer = self.presenter.composite(store, &view);
        self.presenter.queue().submit([command_buffer]);
        output.present();
        Ok(())
    }
}

impl Presenter for X11Compositor {
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        self.presenter.apply(store, changes);
    }
}

/// Creates, titles, and maps the top-level window, wiring `WM_DELETE_WINDOW`.
fn create_window(
    conn: &XCBConnection,
    screen_num: usize,
    config: &X11WindowConfig,
) -> Result<(u32, xproto::Atom, (u32, u32))> {
    let screen = &conn.setup().roots[screen_num];
    let window = conn.generate_id()?;

    let width = u16::try_from(config.size.0).unwrap_or(1024).max(1);
    let height = u16::try_from(config.size.1).unwrap_or(768).max(1);

    let aux = CreateWindowAux::new()
        .background_pixel(screen.black_pixel)
        .event_mask(EventMask::STRUCTURE_NOTIFY);
    conn.create_window(
        screen.root_depth,
        window,
        screen.root,
        0,
        0,
        width,
        height,
        0,
        WindowClass::INPUT_OUTPUT,
        screen.root_visual,
        &aux,
    )?
    .check()?;

    let title = config.title.as_bytes();
    let net_wm_name = intern(conn, b"_NET_WM_NAME")?;
    let utf8_string = intern(conn, b"UTF8_STRING")?;
    let wm_protocols = intern(conn, b"WM_PROTOCOLS")?;
    let wm_delete_window = intern(conn, b"WM_DELETE_WINDOW")?;
    conn.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        title,
    )?;
    conn.change_property8(PropMode::REPLACE, window, net_wm_name, utf8_string, title)?;
    conn.change_property32(
        PropMode::REPLACE,
        window,
        wm_protocols,
        AtomEnum::ATOM,
        &[wm_delete_window],
    )?;
    conn.map_window(window)?;
    conn.flush()?;

    Ok((
        window,
        wm_delete_window,
        (u32::from(width), u32::from(height)),
    ))
}

/// Negotiates the Present extension and selects completion and idle events.
fn configure_present(conn: &XCBConnection, window: u32) -> Result<u32> {
    conn.present_query_version(1, 0)?.reply()?;
    let eid = conn.generate_id()?;
    conn.present_select_input(
        eid,
        window,
        present::EventMask::COMPLETE_NOTIFY | present::EventMask::IDLE_NOTIFY,
    )?;
    conn.flush()?;
    Ok(eid)
}

/// Creates a `wgpu` surface from the raw `xcb_connection_t` and window id.
fn create_surface(
    instance: &wgpu::Instance,
    conn: &XCBConnection,
    screen_num: usize,
    window: u32,
) -> Result<wgpu::Surface<'static>> {
    let raw_connection = conn.get_raw_xcb_connection().cast::<c_void>();
    let window =
        NonZeroU32::new(window).ok_or_else(|| X11BackendError::X11("window id is zero".into()))?;
    let display_handle = wgpu::rwh::XcbDisplayHandle::new(
        NonNull::new(raw_connection),
        i32::try_from(screen_num).unwrap_or(0),
    );
    let window_handle = wgpu::rwh::XcbWindowHandle::new(window);
    let target = wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: Some(wgpu::rwh::RawDisplayHandle::Xcb(display_handle)),
        raw_window_handle: wgpu::rwh::RawWindowHandle::Xcb(window_handle),
    };
    // SAFETY: `raw_connection` and `window` come from the live `XCBConnection`,
    // which this compositor keeps alive for at least as long as the surface.
    let surface = unsafe { instance.create_surface_unsafe(target) }
        .map_err(|err| X11BackendError::Wgpu(Box::new(err)))?;
    Ok(surface)
}

/// Selects an adapter and device, and configures the surface for rendering.
fn configure_surface(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'static>,
    width: u32,
    height: u32,
) -> Result<(wgpu::Device, wgpu::Queue, wgpu::SurfaceConfiguration)> {
    let width = width.max(1);
    let height = height.max(1);
    let (adapter, device, queue) = pollster::block_on(async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(surface),
            })
            .await
            .map_err(|err| X11BackendError::Wgpu(Box::new(err)))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("subduction_backend_x11 device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|err| X11BackendError::Wgpu(Box::new(err)))?;
        Ok::<_, X11BackendError>((adapter, device, queue))
    })?;

    let mut surface_config = surface
        .get_default_config(&adapter, width, height)
        .ok_or_else(|| X11BackendError::Wgpu("the adapter cannot render to this surface".into()))?;
    surface_config.usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    surface.configure(&device, &surface_config);
    Ok((device, queue, surface_config))
}

/// Interns an atom, creating it if necessary.
fn intern(conn: &XCBConnection, name: &[u8]) -> Result<xproto::Atom> {
    Ok(conn.intern_atom(false, name)?.reply()?.atom)
}
