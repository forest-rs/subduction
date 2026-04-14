// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! wgpu stress-test: 1000+ animated layers composited by [`WgpuPresenter`].
//!
//! 10 groups of 100 children each, animated using the shared
//! [`lotta_layers_common`] logic. Each child is a tiny colored square
//! rendered into its own wgpu texture, then composited with transforms,
//! opacity, and back-to-front ordering.
//!
//! Run with: `cargo run -p wgpu_lotta_layers --release`

#![expect(
    clippy::cast_possible_truncation,
    reason = "example code with small known values"
)]

use std::sync::Arc;

use lotta_layers_common::LAYER_SIZE;
use subduction_backend_wgpu::{LayerRoot, Presenter as _, WgpuPresenter};
use subduction_core::layer::{LayerId, LayerStore, SurfaceId};
use subduction_core::output::Color;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_W: u32 = 1024;
const WINDOW_H: u32 = 768;
const NUM_GROUPS: usize = 10;
const LAYERS_PER_GROUP: usize = 100;
const NUM_CHILDREN: usize = NUM_GROUPS * LAYERS_PER_GROUP;

/// WGSL shader that fills the render target with a solid color from a uniform.
const FILL_SHADER: &str = r"
struct FillColor {
    color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> fill: FillColor;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
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

/// Returns an `[r, g, b, a]` color for a given index using golden-angle hue spacing.
fn layer_color(index: usize) -> [f32; 4] {
    let hue = (index as f64 * 137.508) % 360.0;
    let [r, g, b] = lotta_layers_common::hsl_to_rgb(hue, 0.7, 0.6);
    [r as f32, g as f32, b as f32, 0.9]
}

struct App {
    state: Option<RunState>,
}

struct RunState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    presenter: WgpuPresenter,
    store: LayerStore,
    group_ids: Vec<LayerId>,
    child_ids: Vec<LayerId>,
    fill_pipeline: wgpu::RenderPipeline,
    fill_bind_groups: Vec<wgpu::BindGroup>,
    frame_count: u64,
    /// Whether the initial fill has been rendered into each child texture.
    filled: bool,
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
                        .with_title("Subduction · wgpu Lotta Layers (1000+)")
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
            label: Some("wgpu_lotta_layers"),
            ..Default::default()
        }))
        .expect("failed to create device");

        let surface_config = surface
            .get_default_config(&adapter, WINDOW_W, WINDOW_H)
            .expect("surface not compatible with adapter");
        surface.configure(&device, &surface_config);

        let output_format = surface_config.format;

        // --- Fill pipeline for rendering solid colors into layer textures ---
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
            immediate_size: 0,
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
            multiview_mask: None,
            cache: None,
        });

        // Per-child color bind groups.
        let mut fill_bind_groups = Vec::with_capacity(NUM_CHILDREN);
        for i in 0..NUM_CHILDREN {
            let color = layer_color(i);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fill color"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&color));

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

        let mut group_ids = Vec::with_capacity(NUM_GROUPS);
        let mut child_ids = Vec::with_capacity(NUM_CHILDREN);

        for g in 0..NUM_GROUPS {
            let group = store.create_layer();
            store.add_child(root_id, group);
            group_ids.push(group);

            for c in 0..LAYERS_PER_GROUP {
                let child = store.create_layer();
                store.add_child(group, child);
                let surface_idx = g * LAYERS_PER_GROUP + c;
                store.set_content(child, Some(SurfaceId(surface_idx as u32)));
                child_ids.push(child);
            }
        }

        // Layer textures are LAYER_SIZE × LAYER_SIZE pixels.
        let layer_px = LAYER_SIZE as u32;
        let backdrop_color = Color::from_rgba8(0x1a, 0x1a, 0x24, 0xff);
        let root =
            LayerRoot::new(output_format, (WINDOW_W, WINDOW_H)).with_backdrop_color(backdrop_color);
        let mut presenter = WgpuPresenter::new(device, queue, root, (layer_px, layer_px));

        let changes = store.evaluate();
        presenter.apply(&store, &changes);

        let state = RunState {
            window,
            surface,
            surface_config,
            presenter,
            store,
            group_ids,
            child_ids,
            fill_pipeline,
            fill_bind_groups,
            frame_count: 0,
            filled: false,
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
                    .root_mut()
                    .resize(s.surface_config.width, s.surface_config.height);
            }
            WindowEvent::RedrawRequested => {
                s.frame_count += 1;
                let t = s.frame_count as f64 / 60.0;

                // Animate using the shared lotta_layers_common logic.
                lotta_layers_common::animate_groups(
                    &mut s.store,
                    &s.group_ids,
                    &s.child_ids,
                    NUM_GROUPS,
                    LAYERS_PER_GROUP,
                    f64::from(WINDOW_W) / 2.0,
                    f64::from(WINDOW_H) / 2.0,
                    t,
                );

                let changes = s.store.evaluate();
                s.presenter.apply(&s.store, &changes);

                // Fill each child's texture once (content is static).
                if !s.filled {
                    let mut encoder = s.presenter.device().create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("fill encoder"),
                        },
                    );

                    for (i, &child_id) in s.child_ids.iter().enumerate() {
                        let surface_id = SurfaceId(i as u32);
                        let Some(target) = s.presenter.target_for_surface(surface_id) else {
                            continue;
                        };

                        let _ = child_id;

                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("fill pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target.view(),
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
                            multiview_mask: None,
                        });

                        pass.set_pipeline(&s.fill_pipeline);
                        pass.set_bind_group(0, &s.fill_bind_groups[i], &[]);
                        pass.draw(0..3, 0..1);
                    }

                    s.presenter.queue().submit([encoder.finish()]);
                    s.filled = true;
                }

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
