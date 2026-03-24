// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! macOS hybrid example: `CALayer` tree + wgpu-rendered content.
//!
//! Five orbiting layers managed by [`LayerPresenter`] (Core Animation compositing).
//! Layers 0–1 are solid-color `CALayer`s; layers 2–4 have wgpu content rendered
//! into `CAMetalLayer` sublayers (spinning prism, plasma, Julia set fractal).
//!
//! Run with: `cargo run -p macos-wgpu`

#![expect(unsafe_code, reason = "FFI example requires unsafe code")]

use core::cell::RefCell;
use core::ffi::c_void;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
    NSWindow, NSWindowStyleMask,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_core_graphics::CGColor;
use objc2_foundation::{NSNotification, NSObject, NSObjectProtocol, NSString};
use objc2_quartz_core::{CALayer, CAMetalLayer};
use subduction_backend_apple::{
    DisplayLink, LayerPresenter, Presenter as _, compute_present_hints,
};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::output::OutputId;
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::Duration;
use subduction_core::timing::{FrameTick, PendingFeedback};
use subduction_core::transform::Transform3d;

const WINDOW_W: f64 = 800.0;
const WINDOW_H: f64 = 600.0;
const NUM_LAYERS: usize = 5;
const LAYER_SIZE: f64 = 200.0;
/// Pixel size for GPU surfaces.
const GPU_SIZE: u32 = 200;

/// Colors for the demo layers (RGBA, f64 for `CGColor`).
const COLORS: [[f64; 4]; NUM_LAYERS] = [
    [0.95, 0.26, 0.21, 0.9], // red   (solid)
    [0.30, 0.69, 0.31, 0.9], // green (solid)
    [0.13, 0.59, 0.95, 0.9], // blue  (wgpu: prism)
    [1.00, 0.76, 0.03, 0.9], // amber (wgpu: plasma)
    [0.61, 0.15, 0.69, 0.9], // purple (wgpu: julia)
];

/// Indices of layers that use wgpu rendering.
const GPU_LAYER_INDICES: [usize; 3] = [2, 3, 4];

fn set_layer_bg_color(layer: &CALayer, r: f64, g: f64, b: f64, a: f64) {
    let color = CGColor::new_generic_rgb(r, g, b, a);
    layer.setBackgroundColor(Some(&color));
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrismVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

/// Fullscreen quad: two triangles covering clip-space (−1..1).
const FULLSCREEN_QUAD: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

/// Builds the vertex data for a triangular prism (24 vertices).
fn prism_vertices() -> Vec<PrismVertex> {
    // Equilateral triangle at radius 0.4, centered at origin in XZ plane.
    let r: f32 = 0.4;
    let p = [
        [0.0, r],               // angle 0°
        [r * 0.866, -r * 0.5],  // angle 120°
        [-r * 0.866, -r * 0.5], // angle 240°
    ];
    let half_h: f32 = 0.5;

    // Per-face colors (bluish tones).
    let face_colors: [[f32; 3]; 5] = [
        [0.35, 0.55, 0.90], // side 0→1
        [0.20, 0.35, 0.80], // side 1→2
        [0.50, 0.65, 0.95], // side 2→0
        [0.45, 0.70, 1.00], // top cap
        [0.15, 0.25, 0.65], // bottom cap
    ];

    // Outward normals for each side face (perpendicular to edge, in XZ).
    let side_normals: [[f32; 3]; 3] = [
        // Midpoint of edge 0→1 from center, normalized
        {
            let mx = (p[0][0] + p[1][0]) * 0.5;
            let mz = (p[0][1] + p[1][1]) * 0.5;
            let len = (mx * mx + mz * mz).sqrt();
            [mx / len, 0.0, mz / len]
        },
        {
            let mx = (p[1][0] + p[2][0]) * 0.5;
            let mz = (p[1][1] + p[2][1]) * 0.5;
            let len = (mx * mx + mz * mz).sqrt();
            [mx / len, 0.0, mz / len]
        },
        {
            let mx = (p[2][0] + p[0][0]) * 0.5;
            let mz = (p[2][1] + p[0][1]) * 0.5;
            let len = (mx * mx + mz * mz).sqrt();
            [mx / len, 0.0, mz / len]
        },
    ];

    let mut verts = Vec::with_capacity(24);

    // Helper: push a rectangular face between edge (a, b) at y = ±half_h.
    let mut push_side = |ai: usize, bi: usize, face: usize| {
        let at = [p[ai][0], half_h, p[ai][1]];
        let ab = [p[ai][0], -half_h, p[ai][1]];
        let bt = [p[bi][0], half_h, p[bi][1]];
        let bb = [p[bi][0], -half_h, p[bi][1]];
        let n = side_normals[face];
        let c = face_colors[face];
        // Two triangles: at, ab, bb  and  at, bb, bt
        for &pos in &[at, ab, bb, at, bb, bt] {
            verts.push(PrismVertex {
                position: pos,
                normal: n,
                color: c,
            });
        }
    };

    push_side(0, 1, 0);
    push_side(1, 2, 1);
    push_side(2, 0, 2);

    // Top cap (y = +half_h, normal = +Y).
    let top_n = [0.0_f32, 1.0, 0.0];
    let top_c = face_colors[3];
    for &i in &[0_usize, 1, 2] {
        verts.push(PrismVertex {
            position: [p[i][0], half_h, p[i][1]],
            normal: top_n,
            color: top_c,
        });
    }

    // Bottom cap (y = −half_h, normal = −Y, wound CW from below → CCW).
    let bot_n = [0.0_f32, -1.0, 0.0];
    let bot_c = face_colors[4];
    for &i in &[0_usize, 2, 1] {
        verts.push(PrismVertex {
            position: [p[i][0], -half_h, p[i][1]],
            normal: bot_n,
            color: bot_c,
        });
    }

    verts
}

const PRISM_SHADER: &str = r"
@group(0) @binding(0) var<uniform> time: f32;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let c = cos(time);
    let s = sin(time);

    // Y-axis rotation.
    let rotated = vec3(
        in.position.x * c + in.position.z * s,
        in.position.y,
        -in.position.x * s + in.position.z * c,
    );
    let rot_n = vec3(
        in.normal.x * c + in.normal.z * s,
        in.normal.y,
        -in.normal.x * s + in.normal.z * c,
    );

    // Simple perspective.
    let z_off = 2.5;
    let fov = 1.8;
    let depth = rotated.z + z_off;

    var out: VertexOutput;
    out.clip_position = vec4(
        rotated.x * fov / depth,
        rotated.y * fov / depth,
        0.5,
        1.0,
    );
    out.color = in.color;
    out.normal = rot_n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3(0.4, 0.7, 0.6));
    let n = normalize(in.normal);
    let diffuse = max(dot(n, light_dir), 0.2);
    return vec4(in.color * diffuse, 1.0);
}
";

const PLASMA_SHADER: &str = r"
@group(0) @binding(0) var<uniform> time: f32;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = time;
    let x = in.uv.x * 10.0;
    let y = in.uv.y * 10.0;

    let v1 = sin(x + t);
    let v2 = sin(y + t * 1.3);
    let v3 = sin((x + y) * 0.7 + t * 0.7);
    let v4 = sin(sqrt(x * x + y * y + 1.0) + t);
    let val = (v1 + v2 + v3 + v4) * 0.25 + 0.5;

    let r = sin(val * 6.2832) * 0.5 + 0.5;
    let g = sin(val * 6.2832 + 2.094) * 0.5 + 0.5;
    let b = sin(val * 6.2832 + 4.189) * 0.5 + 0.5;

    return vec4(r, g, b, 1.0);
}
";

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

struct GpuLayerState {
    #[expect(
        dead_code,
        reason = "kept alive so the CAMetalLayer is not deallocated"
    )]
    metal_layer: Retained<CAMetalLayer>,
    surface: wgpu::Surface<'static>,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    time_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

define_class! {
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AppDelegate"]
    #[ivars = ()]
    struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn app_did_finish_launching(&self, _notification: &NSNotification) {
            setup_window(MainThreadMarker::from(self));
        }

        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn should_terminate_after_last_window_closed(
            &self,
            _app: &NSApplication,
        ) -> bool {
            true
        }
    }
}

impl AppDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = mtm.alloc::<Self>().set_ivars(());
        unsafe { msg_send![super(this), init] }
    }
}

/// All mutable state lives in a main-thread-only `thread_local!`.
struct AnimState {
    store: LayerStore,
    presenter: LayerPresenter,
    scheduler: Scheduler,
    sub_ids: Vec<LayerId>,
    start_ticks: u64,
    timebase: subduction_core::time::Timebase,
    pending_feedback: Option<PendingFeedback>,

    // wgpu shared state
    device: wgpu::Device,
    queue: wgpu::Queue,
    gpu_layers: Vec<GpuLayerState>,
}

thread_local! {
    static ANIM_STATE: RefCell<Option<AnimState>> = const { RefCell::new(None) };
    static KEEP_ALIVE: RefCell<Option<DisplayLink>> = const { RefCell::new(None) };
}

/// Creates the render pipeline for the prism shader (3-attribute vertex input).
fn create_prism_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("prism shader"),
        source: wgpu::ShaderSource::Wgsl(PRISM_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prism layout"),
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("prism pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<PrismVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 24,
                        shader_location: 2,
                    },
                ],
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
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Creates a render pipeline for a fullscreen-quad shader (2D position input).
fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader_source: &str,
    label: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
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
        multiview_mask: None,
        cache: None,
    })
}

/// Creates a `CAMetalLayer`, sets its bounds, contentsScale to 1, and adds it
/// as a sublayer of `parent`. Returns the retained layer.
fn create_metal_sublayer(parent: &CALayer) -> Retained<CAMetalLayer> {
    let ml = CAMetalLayer::new();
    ml.setBounds(CGRect::new(
        CGPoint::new(0.0, 0.0),
        CGSize::new(LAYER_SIZE, LAYER_SIZE),
    ));
    ml.setContentsScale(1.0);
    // Fill the parent layer's bounds.
    ml.setFrame(CGRect::new(
        CGPoint::new(0.0, 0.0),
        CGSize::new(LAYER_SIZE, LAYER_SIZE),
    ));
    parent.addSublayer(&ml);
    ml
}

fn setup_window(mtm: MainThreadMarker) {
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Resizable
        | NSWindowStyleMask::Miniaturizable;

    let frame = CGRect::new(CGPoint::new(200.0, 200.0), CGSize::new(WINDOW_W, WINDOW_H));

    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc::<NSWindow>(),
            frame,
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    unsafe { window.setReleasedWhenClosed(false) };
    window.setTitle(&NSString::from_str("Subduction · Hybrid CALayer + wgpu"));

    // Dark background via CGColor on the content view's layer.
    let content_view = window.contentView().expect("window has content view");
    content_view.setWantsLayer(true);

    let root_layer = content_view.layer().expect("content view has no layer");
    set_layer_bg_color(&root_layer, 0.12, 0.12, 0.15, 1.0);

    // --- Build the subduction layer tree ---
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut sub_ids: Vec<LayerId> = Vec::new();
    let mut presenter = LayerPresenter::new(root_layer);

    for _ in 0..NUM_LAYERS {
        let layer_id = store.create_layer();
        store.add_child(root_id, layer_id);
        sub_ids.push(layer_id);
    }

    // Initial evaluate — produces `added` entries that the presenter needs.
    let changes = store.evaluate();
    presenter.apply(&store, &changes);

    // --- Set visual properties on the presenter-managed CALayers ---

    // All layers get bounds and corner radius.
    for (i, &layer_id) in sub_ids.iter().enumerate() {
        if let Some(ca) = presenter.get_layer(layer_id.index()) {
            ca.setCornerRadius(12.0);
            ca.setBounds(CGRect::new(
                CGPoint::new(0.0, 0.0),
                CGSize::new(LAYER_SIZE, LAYER_SIZE),
            ));
            // Solid-color layers (0, 1) get a background color.
            if i < 2 {
                let [r, g, b, a] = COLORS[i];
                set_layer_bg_color(ca, r, g, b, a);
            }
        }
    }

    // --- Create CAMetalLayer sublayers for GPU layers (2, 3, 4) ---
    let mut metal_layers: Vec<Retained<CAMetalLayer>> = Vec::with_capacity(3);
    for &idx in &GPU_LAYER_INDICES {
        let parent = presenter
            .get_layer(sub_ids[idx].index())
            .expect("presenter should have this layer");
        metal_layers.push(create_metal_sublayer(parent));
    }

    // --- Initialize wgpu from the first CAMetalLayer ---
    let instance = wgpu::Instance::default();

    // SAFETY: the CAMetalLayer is kept alive via `GpuLayerState::metal_layer`.
    let first_surface = unsafe {
        let ptr: *const CAMetalLayer = &*metal_layers[0];
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(
                ptr as *mut c_void,
            ))
            .expect("failed to create surface")
    };

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&first_surface),
        ..Default::default()
    }))
    .expect("no suitable GPU adapter found");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("subduction_wgpu"),
        ..Default::default()
    }))
    .expect("failed to create device");

    // Create surfaces for the remaining two layers.
    let mut surfaces: Vec<wgpu::Surface<'static>> = vec![first_surface];
    for ml in &metal_layers[1..] {
        // SAFETY: same as above — each CAMetalLayer is kept alive.
        let surface = unsafe {
            let ptr: *const CAMetalLayer = &**ml;
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(
                    ptr as *mut c_void,
                ))
                .expect("failed to create surface")
        };
        surfaces.push(surface);
    }

    // Configure all surfaces.
    for surface in &surfaces {
        let config = surface
            .get_default_config(&adapter, GPU_SIZE, GPU_SIZE)
            .expect("surface not compatible with adapter");
        surface.configure(&device, &config);
    }

    let tex_format = surfaces[0]
        .get_default_config(&adapter, GPU_SIZE, GPU_SIZE)
        .expect("surface not compatible")
        .format;

    // --- Time-uniform bind group layout (shared by all 3 shaders) ---
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

    // --- Build per-GPU-layer state ---
    let prism_pipeline = create_prism_pipeline(&device, tex_format, &time_bgl);
    let plasma_pipeline =
        create_fullscreen_pipeline(&device, tex_format, &time_bgl, PLASMA_SHADER, "plasma");
    let julia_pipeline =
        create_fullscreen_pipeline(&device, tex_format, &time_bgl, JULIA_SHADER, "julia");

    let pipelines = [prism_pipeline, plasma_pipeline, julia_pipeline];

    // Prism vertex buffer.
    let prism_verts = prism_vertices();
    let prism_vb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prism vertices"),
        size: (prism_verts.len() * size_of::<PrismVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&prism_vb, 0, bytemuck::cast_slice(&prism_verts));

    // Fullscreen quad vertex buffer (shared by plasma + julia).
    let quad_vb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fullscreen quad"),
        size: size_of_val(&FULLSCREEN_QUAD) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&quad_vb, 0, bytemuck::cast_slice(&FULLSCREEN_QUAD));

    let vertex_buffers = [prism_vb, quad_vb.clone(), quad_vb];
    let prism_vertex_count = u32::try_from(prism_verts.len())
        .expect("prism vertex count must fit in u32 draw-call vertex count");
    let vertex_counts: [u32; 3] = [prism_vertex_count, 6, 6];

    let mut gpu_layers: Vec<GpuLayerState> = Vec::with_capacity(3);
    // We must consume `surfaces` and `metal_layers` together.
    let surface_iter = surfaces.into_iter();
    let ml_iter = metal_layers.into_iter();

    for (i, ((surface, ml), pipeline)) in surface_iter.zip(ml_iter).zip(pipelines).enumerate() {
        let time_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("time uniform"),
            size: 16, // align to 16 bytes for uniform
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

        // Clone the appropriate vertex buffer (we already wrote data into them).
        let vb = if i == 0 {
            // Prism — need its own buffer.
            vertex_buffers[0].clone()
        } else {
            // Plasma / Julia share the quad buffer.
            vertex_buffers[1].clone()
        };

        gpu_layers.push(GpuLayerState {
            metal_layer: ml,
            surface,
            pipeline,
            bind_group,
            time_buffer,
            vertex_buffer: vb,
            vertex_count: vertex_counts[i],
        });
    }

    window.center();
    window.makeKeyAndOrderFront(None);

    // --- CADisplayLink-driven animation ---
    let timebase = DisplayLink::timebase();
    let start_ticks = DisplayLink::now().ticks();
    let scheduler = Scheduler::new(SchedulerConfig::macos());

    ANIM_STATE.with(|cell| {
        *cell.borrow_mut() = Some(AnimState {
            store,
            presenter,
            scheduler,
            sub_ids,
            start_ticks,
            timebase,
            pending_feedback: None,
            device,
            queue,
            gpu_layers,
        });
    });

    let link = DisplayLink::new(on_tick, OutputId(0), mtm);
    link.start();

    KEEP_ALIVE.with(|cell| {
        *cell.borrow_mut() = Some(link);
    });
}

fn on_tick(tick: FrameTick) {
    ANIM_STATE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let Some(s) = borrow.as_mut() else { return };

        // Resolve previous frame's feedback with actual_present from this tick.
        if let Some(pending) = s.pending_feedback.take() {
            let feedback = pending.resolve(tick.prev_actual_present);
            s.scheduler.observe(&feedback);
        }

        let build_start = DisplayLink::now();

        // Compute hints and plan the frame.
        let safety = Duration(s.scheduler.safety_margin_ticks());
        let hints = compute_present_hints(&tick, safety);
        let plan = s.scheduler.plan(&tick, &hints);

        // Convert semantic_time to elapsed seconds for the animation.
        let elapsed_ticks = plan.semantic_time.ticks().saturating_sub(s.start_ticks);
        let elapsed_nanos = s.timebase.ticks_to_nanos(elapsed_ticks);
        let t = elapsed_nanos as f64 / 1_000_000_000.0;

        // Animate transforms and opacities in the layer store.
        animate_transforms(&mut s.store, &s.sub_ids, t);

        // Evaluate dirty state and apply to the CALayer tree.
        let changes = s.store.evaluate();
        s.presenter.apply(&s.store, &changes);

        // --- Render wgpu content into GPU layers ---
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Shader uniform intentionally narrows wall-clock seconds from f64 to f32"
        )]
        let time_f32 = t as f32;

        for gpu in &s.gpu_layers {
            // Upload time uniform.
            s.queue
                .write_buffer(&gpu.time_buffer, 0, bytemuck::bytes_of(&time_f32));

            let frame = match gpu.surface.get_current_texture() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("failed to acquire GPU frame: {e}");
                    continue;
                }
            };
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = s
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu layer encoder"),
                });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gpu layer pass"),
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
                    multiview_mask: None,
                });
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                pass.draw(0..gpu.vertex_count, 0..1);
            }

            s.queue.submit(Some(encoder.finish()));
            frame.present();
        }

        // Store pending feedback for resolution on next tick.
        s.pending_feedback = Some(PendingFeedback {
            hints,
            build_start,
            submitted_at: DisplayLink::now(),
        });
    });
}

fn animate_transforms(store: &mut LayerStore, sub_ids: &[LayerId], t: f64) {
    let cx = WINDOW_W / 2.0;
    let cy = WINDOW_H / 2.0;

    for (i, &layer_id) in sub_ids.iter().enumerate() {
        let phase = i as f64 * core::f64::consts::TAU / NUM_LAYERS as f64;
        let radius = 150.0 + 50.0 * (t * 0.5 + phase).sin();
        let angle = t * (0.6 + i as f64 * 0.1) + phase;

        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();

        // Also rotate each layer around its own center.
        let rotation = t * 2.0 + phase;

        let transform =
            Transform3d::from_translation(x, y, 0.0) * Transform3d::from_rotation_z(rotation);
        store.set_transform(layer_id, transform);

        // Pulse opacity.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Opacity is intentionally narrowed from bounded [0, 1] f64 to f32 for layer API"
        )]
        let opacity = (0.5 + 0.5 * (t * 1.5 + phase).sin()) as f32;
        store.set_opacity(layer_id, opacity);
    }
}

fn main() {
    let mtm = MainThreadMarker::new().expect("must be on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    let delegate = AppDelegate::new(mtm);
    let delegate_proto = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(delegate_proto));

    app.run();
}
