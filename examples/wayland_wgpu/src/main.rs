// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wayland example: frame-callback-paced wgpu rendering.
//!
//! Renders an animated Julia set fractal into a Wayland surface, using
//! `subduction_backend_wayland` for frame callback pacing.
//!
//! This example uses embedded-state mode: the host owns the event queue and
//! embeds [`WaylandState`] in its own state struct, wiring `delegate_dispatch!`
//! for backend protocol objects.
//!
//! Run with: `cargo run -p wayland_wgpu`

#![expect(unsafe_code, reason = "wgpu surface creation from raw Wayland handles requires unsafe")]

use std::ptr::NonNull;

use bytemuck::{Pod, Zeroable};
use subduction_backend_wayland::{
    FeedbackData, FrameCallbackData, OutputGlobalData, WaylandProtocol, WaylandState,
};
use wayland_client::protocol::{wl_callback, wl_compositor, wl_output, wl_registry, wl_surface};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::wp::presentation_time::client::{
    wp_presentation, wp_presentation_feedback,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

const WINDOW_W: u32 = 800;
const WINDOW_H: u32 = 600;
const WINDOW_TITLE: &str = "Subduction · Wayland + wgpu";

// ---------------------------------------------------------------------------
// WGSL shader — Julia set fractal
// ---------------------------------------------------------------------------

const JULIA_SHADER: &str = r"
@group(0) @binding(0) var<uniform> time: f32;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(pos, 0.0, 1.0);
    out.uv = pos * 1.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = time * 0.3;
    let c = vec2(
        -0.7 + 0.2 * cos(t),
        0.27015 + 0.2 * sin(t),
    );

    var z = in.uv;
    var iter = 0u;
    let max_iter = 128u;

    for (var i = 0u; i < max_iter; i++) {
        let zn = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        z = zn;
        if dot(z, z) > 4.0 {
            break;
        }
        iter++;
    }

    if iter == max_iter {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    let t_val = f32(iter) / f32(max_iter);
    let r = 0.5 + 0.5 * cos(6.2832 * t_val + 0.0);
    let g = 0.5 + 0.5 * cos(6.2832 * t_val + 1.0);
    let b = 0.5 + 0.5 * cos(6.2832 * t_val + 2.0);

    return vec4(r, g, b, 1.0);
}
";

/// Fullscreen quad: two triangles covering clip-space (-1..1).
const FULLSCREEN_QUAD: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

/// Uniform data for the time value, padded to 16 bytes for GPU alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimeUniform {
    time: f32,
    _pad: [f32; 3],
}

// ---------------------------------------------------------------------------
// Host state — embeds WaylandState for backend dispatch delegation
// ---------------------------------------------------------------------------

struct HostState {
    wayland: WaylandState,
    // Wayland windowing globals
    compositor: Option<wl_compositor::WlCompositor>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    // Window state
    configured: bool,
    closed: bool,
    width: u32,
    height: u32,
}

impl AsMut<WaylandState> for HostState {
    fn as_mut(&mut self) -> &mut WaylandState {
        &mut self.wayland
    }
}

// ---------------------------------------------------------------------------
// Backend protocol delegation (embedded-state mode)
//
// These four protocols are only consumed by the backend (the host never needs
// to intercept output hotplug, presentation clock, frame callback, or feedback
// events), so plain delegate_dispatch! forwarding is sufficient.
//
// WlRegistry is intentionally NOT delegated here — see the manual Dispatch
// impl below.
// ---------------------------------------------------------------------------

wayland_client::delegate_dispatch!(HostState:
    [wl_output::WlOutput: OutputGlobalData] => WaylandProtocol);
wayland_client::delegate_dispatch!(HostState:
    [wp_presentation::WpPresentation: ()] => WaylandProtocol);
wayland_client::delegate_dispatch!(HostState:
    [wl_callback::WlCallback: FrameCallbackData] => WaylandProtocol);
wayland_client::delegate_dispatch!(HostState:
    [wp_presentation_feedback::WpPresentationFeedback: FeedbackData] => WaylandProtocol);

// ---------------------------------------------------------------------------
// Custom WlRegistry dispatch — binds host globals + forwards to backend
//
// Both the host and the backend need to react to the same registry events:
// the host binds wl_compositor and xdg_wm_base for windowing, while the
// backend binds wl_output and wp_presentation for pacing. Since
// delegate_dispatch! forwards events to a single delegate with no hook for
// the host to intercept them, we implement Dispatch manually so we can:
//   1. Bind host globals ourselves.
//   2. Forward every event to the backend's WaylandProtocol handler.
// ---------------------------------------------------------------------------

impl Dispatch<wl_registry::WlRegistry, ()> for HostState {
    fn event(
        state: &mut Self,
        proxy: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        data: &(),
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        // Bind host-side globals (windowing).
        if let wl_registry::Event::Global {
            name,
            ref interface,
            version,
        } = event
        {
            if interface == wl_compositor::WlCompositor::interface().name
                && state.compositor.is_none()
            {
                let compositor: wl_compositor::WlCompositor = proxy.bind(name, version.min(6), qh, ());
                state.compositor = Some(compositor);
            } else if interface == xdg_wm_base::XdgWmBase::interface().name
                && state.wm_base.is_none()
            {
                let wm_base: xdg_wm_base::XdgWmBase = proxy.bind(name, version.min(6), qh, ());
                state.wm_base = Some(wm_base);
            }
        }

        // Forward to the backend so it can bind its own globals
        // (wl_output, wp_presentation).
        <WaylandProtocol as Dispatch<wl_registry::WlRegistry, (), Self>>::event(
            state, proxy, event, data, conn, qh,
        );
    }
}

// ---------------------------------------------------------------------------
// Host-only protocol dispatch
//
// wayland-client requires a Dispatch impl for every object type created on
// the queue. These are windowing objects the backend doesn't know about.
// ---------------------------------------------------------------------------

impl Dispatch<wl_compositor::WlCompositor, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_compositor::WlCompositor,
        _event: wl_compositor::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // wl_compositor has no events.
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_surface::WlSurface,
        _event: wl_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // wl_surface events (enter/leave) not needed for this example.
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for HostState {
    fn event(
        _state: &mut Self,
        proxy: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for HostState {
    fn event(
        state: &mut Self,
        proxy: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            proxy.ack_configure(serial);
            state.configured = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Close => {
                state.closed = true;
            }
            xdg_toplevel::Event::Configure {
                width,
                height,
                ..
            } => {
                if width > 0 && height > 0 {
                    #[expect(
                        clippy::cast_sign_loss,
                        reason = "width/height are positive after the > 0 check"
                    )]
                    {
                        state.width = width as u32;
                        state.height = height as u32;
                    }
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// wgpu setup helpers
// ---------------------------------------------------------------------------

fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("julia shader"),
        source: wgpu::ShaderSource::Wgsl(JULIA_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("julia layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("julia pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<[f32; 2]>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &wgpu::Surface<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    time_buffer: &wgpu::Buffer,
    vertex_buffer: &wgpu::Buffer,
    time_secs: f32,
) -> wgpu::SurfaceTexture {
    let uniform = TimeUniform {
        time: time_secs,
        _pad: [0.0; 3],
    };
    queue.write_buffer(time_buffer, 0, bytemuck::bytes_of(&uniform));

    let frame = surface
        .get_current_texture()
        .expect("failed to acquire surface texture");
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("frame encoder"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("julia pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
    frame
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    // Step 1: Wayland connection and event queue.
    let connection = Connection::connect_to_env().expect("failed to connect to Wayland display");
    let mut event_queue: EventQueue<HostState> = connection.new_event_queue();
    let qh = event_queue.handle();

    let display = connection.display();
    let registry = display.get_registry(&qh, ());

    let mut state = HostState {
        wayland: WaylandState::new(),
        compositor: None,
        wm_base: None,
        configured: false,
        closed: false,
        width: WINDOW_W,
        height: WINDOW_H,
    };

    // Step 5: Bootstrap — store registry for the backend and roundtrip.
    state.wayland.set_registry(registry);
    event_queue
        .roundtrip(&mut state)
        .expect("initial roundtrip failed");

    let compositor = state
        .compositor
        .as_ref()
        .expect("compositor not found — is a Wayland compositor running?");
    let wm_base = state
        .wm_base
        .as_ref()
        .expect("xdg_wm_base not found — compositor lacks xdg-shell support");

    // Create the wl_surface and register it with the backend.
    let wl_surface = compositor.create_surface(&qh, ());
    state
        .wayland
        .set_surface(wl_surface.clone())
        .expect("failed to set surface");

    // Create xdg_surface and xdg_toplevel.
    let xdg_surface = wm_base.get_xdg_surface(&wl_surface, &qh, ());
    let xdg_toplevel = xdg_surface.get_toplevel(&qh, ());
    xdg_toplevel.set_title(WINDOW_TITLE.to_string());
    xdg_toplevel.set_app_id("subduction.wayland_wgpu".to_string());

    // Empty commit to trigger first configure.
    wl_surface.commit();
    event_queue
        .roundtrip(&mut state)
        .expect("configure roundtrip failed");

    assert!(
        state.configured,
        "xdg_surface.configure not received after roundtrip"
    );

    // Step 2: wgpu setup.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let wgpu_surface = {
        let display_ptr = connection.backend().display_ptr().cast::<std::ffi::c_void>();
        let surface_ptr = Proxy::id(&wl_surface).as_ptr().cast::<std::ffi::c_void>();

        // SAFETY: The Wayland display and surface are valid for the lifetime
        // of this program. The wl_surface is kept alive by the `wl_surface` binding.
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: wgpu::rwh::RawDisplayHandle::Wayland(
                wgpu::rwh::WaylandDisplayHandle::new(
                    NonNull::new(display_ptr).expect("display pointer is null"),
                ),
            ),
            raw_window_handle: wgpu::rwh::RawWindowHandle::Wayland(
                wgpu::rwh::WaylandWindowHandle::new(
                    NonNull::new(surface_ptr).expect("surface pointer is null"),
                ),
            ),
        };
        unsafe { instance.create_surface_unsafe(target) }
            .expect("failed to create wgpu surface")
    };

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&wgpu_surface),
        ..Default::default()
    }))
    .expect("no suitable GPU adapter found");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wayland_wgpu"),
        ..Default::default()
    }))
    .expect("failed to create wgpu device");

    // Configure the surface.
    let surface_config = wgpu_surface
        .get_default_config(&adapter, state.width, state.height)
        .expect("surface not compatible with adapter");
    wgpu_surface.configure(&device, &surface_config);

    // Time uniform bind group layout.
    let time_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("time bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline = create_pipeline(&device, surface_config.format, &time_bgl);

    let time_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("time uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("time bind group"),
        layout: &time_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: time_buffer.as_entire_binding(),
        }],
    });

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fullscreen quad"),
        size: size_of_val(&FULLSCREEN_QUAD) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&FULLSCREEN_QUAD));

    // Track the start time for animation.
    let start_time = subduction_backend_wayland::now();

    // Step 6: First-frame bootstrap — render one frame before entering the
    // tick-driven loop, so the surface is mapped and the first frame callback
    // can be delivered.
    let frame = render_frame(
        &device,
        &queue,
        &wgpu_surface,
        &pipeline,
        &bind_group,
        &time_buffer,
        &vertex_buffer,
        0.0,
    );
    state
        .wayland
        .request_frame(&qh)
        .expect("failed to request initial frame callback");
    frame.present();
    connection.flush().expect("failed to flush connection");

    // Track current surface dimensions for resize handling.
    let mut current_width = state.width;
    let mut current_height = state.height;

    // Step 7: Main frame loop.
    loop {
        event_queue
            .blocking_dispatch(&mut state)
            .expect("dispatch failed");

        if state.closed {
            break;
        }

        // Handle resize.
        if state.width != current_width || state.height != current_height {
            current_width = state.width;
            current_height = state.height;
            let mut config = surface_config.clone();
            config.width = current_width;
            config.height = current_height;
            wgpu_surface.configure(&device, &config);
        }

        while let Some(tick) = state.wayland.poll_tick() {
            // Compute animation time from the tick's timestamp.
            let elapsed_nanos = tick.now.ticks().saturating_sub(start_time.ticks());
            #[expect(
                clippy::cast_precision_loss,
                reason = "Nanosecond counter to f64 seconds — precision loss is acceptable for animation"
            )]
            let elapsed_secs = elapsed_nanos as f64 / 1_000_000_000.0;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Shader uniform intentionally narrows wall-clock seconds from f64 to f32"
            )]
            let time_secs = elapsed_secs as f32;

            let frame = render_frame(
                &device,
                &queue,
                &wgpu_surface,
                &pipeline,
                &bind_group,
                &time_buffer,
                &vertex_buffer,
                time_secs,
            );

            // Request next frame callback BEFORE present, so the request
            // becomes pending state that wgpu's present() commit carries.
            state
                .wayland
                .request_frame(&qh)
                .expect("failed to request frame callback");
            frame.present();
            connection.flush().expect("failed to flush connection");
        }
    }
}
