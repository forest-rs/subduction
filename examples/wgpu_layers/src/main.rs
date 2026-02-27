// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal wgpu fallback compositor example.
//!
//! Creates a window with several animated colored layers composited via
//! [`WgpuPresenter`]. Each layer gets a solid-color fill rendered into its
//! texture, then the presenter composites them with orbiting transforms and
//! pulsing opacity onto the window surface.
//!
//! Run with: `cargo run -p wgpu_layers`

#![expect(
    clippy::cast_possible_truncation,
    reason = "example code with small known values"
)]

use std::sync::Arc;

use subduction_backend_wgpu::WgpuPresenter;
use subduction_core::backend::Presenter as _;
use subduction_core::layer::{LayerId, LayerStore, SurfaceId};
use subduction_core::transform::Transform3d;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_W: u32 = 800;
const WINDOW_H: u32 = 600;
const NUM_LAYERS: usize = 5;
const LAYER_SIZE: u32 = 200;

/// RGBA colors for the demo layers.
const COLORS: [[f32; 4]; NUM_LAYERS] = [
    [0.95, 0.26, 0.21, 0.9], // red
    [0.30, 0.69, 0.31, 0.9], // green
    [0.13, 0.59, 0.95, 0.9], // blue
    [1.00, 0.76, 0.03, 0.9], // amber
    [0.61, 0.15, 0.69, 0.9], // purple
];

/// WGSL shader that fills the entire layer with a solid color from a uniform.
const FILL_SHADER: &str = r"
struct FillColor {
    color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> fill: FillColor;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle trick: 3 vertices that cover clip space.
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Output premultiplied alpha (rgb * a) for correct compositor blending.
    let c = fill.color;
    return vec4(c.rgb * c.a, c.a);
}
";

struct App {
    state: Option<RunState>,
}

struct RunState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    presenter: WgpuPresenter,
    store: LayerStore,
    layer_ids: Vec<LayerId>,
    fill_pipeline: wgpu::RenderPipeline,
    fill_bind_groups: Vec<wgpu::BindGroup>,
    frame_count: u64,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Subduction · wgpu Compositor")
                        .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_W, WINDOW_H)),
                )
                .expect("failed to create window"),
        );

        let instance = wgpu::Instance::default();

        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_layers"),
            ..Default::default()
        }))
        .expect("failed to create device");

        let surface_config = surface
            .get_default_config(&adapter, WINDOW_W, WINDOW_H)
            .expect("surface not compatible with adapter");
        surface.configure(&device, &surface_config);

        let output_format = surface_config.format;

        // --- Build a fill pipeline for rendering solid colors into layer textures ---
        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fill shader"),
            source: wgpu::ShaderSource::Wgsl(FILL_SHADER.into()),
        });

        let fill_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fill bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let fill_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fill layout"),
            bind_group_layouts: &[&fill_bgl],
            push_constant_ranges: &[],
        });

        let fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fill pipeline"),
            layout: Some(&fill_layout),
            vertex: wgpu::VertexState {
                module: &fill_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fill_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
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
        });

        // Create per-layer color uniform buffers and bind groups.
        let mut fill_bind_groups = Vec::with_capacity(NUM_LAYERS);
        for color in &COLORS {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fill color"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(color));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fill bg"),
                layout: &fill_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            fill_bind_groups.push(bg);
        }

        // --- Build the subduction layer tree ---
        let mut store = LayerStore::new();
        let root_id = store.create_layer();

        let mut layer_ids = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let id = store.create_layer();
            store.add_child(root_id, id);
            store.set_content(id, Some(SurfaceId(i as u32)));
            layer_ids.push(id);
        }

        // Create the presenter and do the initial apply.
        let presenter = WgpuPresenter::new(
            device,
            queue,
            output_format,
            (WINDOW_W, WINDOW_H),
            (LAYER_SIZE, LAYER_SIZE),
        );

        let changes = store.evaluate();
        let mut presenter = presenter;
        presenter.apply(&store, &changes);

        let state = RunState {
            window,
            surface,
            surface_config,
            presenter,
            store,
            layer_ids,
            fill_pipeline,
            fill_bind_groups,
            frame_count: 0,
        };
        state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(s) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                s.surface_config.width = new_size.width.max(1);
                s.surface_config.height = new_size.height.max(1);
                s.surface.configure(s.presenter.device(), &s.surface_config);
                s.presenter
                    .resize_output(s.surface_config.width, s.surface_config.height);
            }
            WindowEvent::RedrawRequested => {
                s.frame_count += 1;
                let t = s.frame_count as f64 / 60.0;

                // Animate transforms and opacities.
                let cx = f64::from(WINDOW_W) / 2.0;
                let cy = f64::from(WINDOW_H) / 2.0;

                for (i, &layer_id) in s.layer_ids.iter().enumerate() {
                    let phase = i as f64 * core::f64::consts::TAU / NUM_LAYERS as f64;
                    let radius = 150.0 + 50.0 * (t * 0.5 + phase).sin();
                    let angle = t * (0.6 + i as f64 * 0.1) + phase;

                    let x = cx + radius * angle.cos();
                    let y = cy + radius * angle.sin();
                    let rotation = t * 2.0 + phase;

                    let transform = Transform3d::from_translation(x, y, 0.0)
                        * Transform3d::from_rotation_z(rotation);
                    s.store.set_transform(layer_id, transform);

                    let opacity = (0.5 + 0.5 * (t * 1.5 + phase).sin()) as f32;
                    s.store.set_opacity(layer_id, opacity);
                }

                // Evaluate and apply.
                let changes = s.store.evaluate();
                s.presenter.apply(&s.store, &changes);

                // Render solid colors into each layer's texture.
                let mut encoder =
                    s.presenter
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("fill encoder"),
                        });

                for (i, &layer_id) in s.layer_ids.iter().enumerate() {
                    let surface_id = SurfaceId(i as u32);
                    let Some(view) = s.presenter.texture_for_surface(surface_id) else {
                        continue;
                    };

                    let _ = layer_id; // used only for iteration index

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("fill pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    pass.set_pipeline(&s.fill_pipeline);
                    pass.set_bind_group(0, &s.fill_bind_groups[i], &[]);
                    pass.draw(0..3, 0..1);
                }

                s.presenter.queue().submit([encoder.finish()]);

                // Composite and present.
                let output_frame = match s.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Outdated) => return,
                    Err(e) => {
                        eprintln!("surface error: {e}");
                        return;
                    }
                };
                let output_view = output_frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let cmd = s.presenter.composite(&s.store, &output_view);
                // Queue reference must be obtained before submitting since
                // composite() borrows &mut self.
                s.presenter.queue().submit([cmd]);
                output_frame.present();

                s.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop failed");
}
