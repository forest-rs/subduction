// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Web example: animated DOM elements driven by `subduction_backend_web`.
//!
//! Creates a dark container with six animated elements (a WebGL canvas, a WebGPU
//! canvas, and four colored divs) that orbit and pulse opacity, demonstrating the
//! web backend's building blocks: [`RafLoop`] for timing, [`DomPresenter`] for
//! presentation, and [`Scheduler`] for frame planning.
//!
//! Build with: `wasm-pack build --target web examples/web_layers`
//!
//! Then serve `examples/web_layers/` and open `index.html` in a browser.
//!
//! [`RafLoop`]: subduction_backend_web::RafLoop
//! [`DomPresenter`]: subduction_backend_web::DomPresenter
//! [`Scheduler`]: subduction_core::scheduler::Scheduler

// This crate only runs in the browser; suppress dead-code warnings when
// cargo-checking on a native host target.
#![no_std]
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "this crate only runs in the browser")
)]
#![expect(unsafe_code, reason = "js_sys::Float32Array::view requires unsafe")]
#![expect(
    clippy::cast_possible_truncation,
    reason = "demo code intentionally narrows canvas dimensions and shader uniforms"
)]

extern crate alloc;

use alloc::format;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::f64::consts::TAU;

use wasm_bindgen::prelude::*;
use web_sys::{
    Document, HtmlCanvasElement, HtmlElement, WebGl2RenderingContext, WebGlProgram, WebGlShader,
    WebGlUniformLocation,
};

use kurbo::Size;
use subduction_backend_web::RafLoop;
use subduction_backend_web::{DomPresenter, Presenter as _};
use subduction_core::layer::{LayerId, LayerStore};
use subduction_core::output::OutputId;
use subduction_core::scheduler::{Scheduler, SchedulerConfig};
use subduction_core::time::Duration;
use subduction_core::timing::{FrameTick, PresentFeedback};
use subduction_core::transform::Transform3d;

const CONTAINER_W: f64 = 800.0;
const CONTAINER_H: f64 = 600.0;
const NUM_LAYERS: usize = 6;

/// Element specs: (width, height).
const ELEMENT_SPECS: [(f64, f64); NUM_LAYERS] = [
    (120.0, 120.0), // WebGL — spinning RGB triangle
    (120.0, 120.0), // WebGPU — plasma effect
    (80.0, 80.0),   // red div
    (80.0, 80.0),   // green div
    (80.0, 80.0),   // blue div
    (80.0, 80.0),   // amber div
];

const DIV_COLORS: [&str; 4] = [
    "rgba(242, 67, 54, 0.9)",  // red
    "rgba(77, 176, 80, 0.9)",  // green
    "rgba(33, 150, 243, 0.9)", // blue
    "rgba(255, 194, 8, 0.9)",  // amber
];

const GL_VERTEX_SHADER: &str = r"#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_color;
uniform float u_time;
out vec3 v_color;
void main() {
    float c = cos(u_time);
    float s = sin(u_time);
    vec2 r = vec2(a_position.x * c - a_position.y * s,
                  a_position.x * s + a_position.y * c);
    gl_Position = vec4(r, 0.0, 1.0);
    v_color = a_color;
}
";

const GL_FRAGMENT_SHADER: &str = r"#version 300 es
precision mediump float;
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
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

/// Fullscreen quad: two triangles covering clip-space (−1..1).
const FULLSCREEN_QUAD: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

struct WebGlState {
    context: WebGl2RenderingContext,
    program: WebGlProgram,
    time_location: WebGlUniformLocation,
}

struct WgpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    time_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
}

struct AnimState {
    store: LayerStore,
    scheduler: Scheduler,
    presenter: DomPresenter,
    /// Element sizes indexed by raw layer slot index.
    sizes: Vec<(f64, f64)>,
    layer_ids: Vec<LayerId>,
    start_us: u64,
    timebase: subduction_core::time::Timebase,
    webgl: Option<WebGlState>,
    wgpu: Option<WgpuState>,
}

/// Entry point — called automatically by `wasm_bindgen(start)`.
#[cfg_attr(all(target_arch = "wasm32", not(test)), wasm_bindgen(start))]
pub fn main() -> Result<(), JsValue> {
    let window = web_sys::window().expect("no global window");
    let document = window.document().expect("no document");

    let container = create_container(&document)?;
    document.body().expect("no body").append_child(&container)?;

    // Build layer tree.
    let mut store = LayerStore::new();
    let root_id = store.create_layer();

    let mut layer_ids = Vec::with_capacity(NUM_LAYERS);
    let mut sizes = vec![(0.0, 0.0)]; // root placeholder at slot 0

    for &spec in &ELEMENT_SPECS {
        let layer_id = store.create_layer();
        store.add_child(root_id, layer_id);
        let slot = layer_id.index() as usize;
        if sizes.len() <= slot {
            sizes.resize(slot + 1, (0.0, 0.0));
        }
        sizes[slot] = spec;
        layer_ids.push(layer_id);
    }

    // Initial evaluate — presenter.apply() creates a <div> for each added layer.
    let initial = store.evaluate();
    let mut presenter = DomPresenter::new(container);
    presenter.apply(&store, &initial);

    // Customize the presenter's divs: set sizes, colors, and append canvas children.
    let mut webgl_state: Option<WebGlState> = None;
    let mut wgpu_canvas: Option<HtmlCanvasElement> = None;

    for (i, &layer_id) in layer_ids.iter().enumerate() {
        let idx = layer_id.index();
        let el = presenter.get_element(idx).expect("element just created");
        let (w, h) = ELEMENT_SPECS[i];
        let s = el.style();
        s.set_property("width", &format!("{w}px"))?;
        s.set_property("height", &format!("{h}px"))?;

        match i {
            0 => {
                // WebGL canvas — create and append as child of the presenter's div.
                let (canvas, gl) = create_webgl_canvas(&document, w, h)?;
                el.append_child(&canvas)?;
                webgl_state = Some(gl);
            }
            1 => {
                // WebGPU canvas — create and append as child.
                let canvas: HtmlCanvasElement = document.create_element("canvas")?.unchecked_into();
                canvas.set_width(w as u32);
                canvas.set_height(h as u32);
                el.append_child(&canvas)?;
                wgpu_canvas = Some(canvas);
            }
            _ => {
                // Colored div.
                s.set_property("background", DIV_COLORS[i - 2])?;
                s.set_property("border-radius", "12px")?;
            }
        }
    }

    let timebase = subduction_backend_web::timebase();
    let start_us = subduction_backend_web::now().ticks();

    let state = Rc::new(RefCell::new(AnimState {
        store,
        scheduler: Scheduler::new(SchedulerConfig::web()),
        presenter,
        layer_ids,
        sizes,
        start_us,
        timebase,
        webgl: webgl_state,
        wgpu: None,
    }));

    // Spawn async wgpu initialization — populates `state.wgpu` when ready.
    init_wgpu_async(
        Rc::clone(&state),
        wgpu_canvas.expect("WebGPU canvas not created"),
    );

    // Start animation immediately: WebGL renders right away, wgpu starts
    // once the async initialization completes.
    let state_cb = Rc::clone(&state);
    let raf = RafLoop::new(move |tick| on_tick(&state_cb, tick), OutputId(0));
    raf.start();

    // Keep the RafLoop alive — there is no graceful shutdown on the web.
    core::mem::forget(raf);

    Ok(())
}

/// Spawns the asynchronous wgpu adapter/device negotiation.
///
/// On non-wasm targets this is a no-op since the example only runs in a browser.
#[cfg(target_arch = "wasm32")]
fn init_wgpu_async(state: Rc<RefCell<AnimState>>, canvas: HtmlCanvasElement) {
    wasm_bindgen_futures::spawn_local(async move {
        let wgpu_state = init_wgpu(&canvas).await;
        state.borrow_mut().wgpu = Some(wgpu_state);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn init_wgpu_async(_state: Rc<RefCell<AnimState>>, _canvas: HtmlCanvasElement) {}

#[cfg(target_arch = "wasm32")]
async fn init_wgpu(canvas: &HtmlCanvasElement) -> WgpuState {
    let instance = wgpu::Instance::default();

    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .expect("create wgpu surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .expect("no suitable GPU adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("web-layers-wgpu"),
            ..Default::default()
        })
        .await
        .expect("device creation failed");

    let width = canvas.width();
    let height = canvas.height();
    let config = surface
        .get_default_config(&adapter, width, height)
        .expect("surface not compatible with adapter");
    surface.configure(&device, &config);

    let format = config.format;

    // Time-uniform bind group layout.
    let time_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("time bgl"),
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

    // Plasma render pipeline.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("plasma"),
        source: wgpu::ShaderSource::Wgsl(PLASMA_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("plasma"),
        bind_group_layouts: &[&time_bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("plasma"),
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
    });

    // Buffers.
    let time_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("time"),
        size: 16, // align to 16 bytes for uniform
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quad"),
        size: size_of_val(&FULLSCREEN_QUAD) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&FULLSCREEN_QUAD));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("time"),
        layout: &time_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: time_buffer.as_entire_binding(),
        }],
    });

    WgpuState {
        device,
        queue,
        surface,
        pipeline,
        bind_group,
        time_buffer,
        vertex_buffer,
    }
}

fn create_container(doc: &Document) -> Result<HtmlElement, JsValue> {
    let el: HtmlElement = doc.create_element("div")?.unchecked_into();
    let s = el.style();
    s.set_property("width", &format!("{CONTAINER_W}px"))?;
    s.set_property("height", &format!("{CONTAINER_H}px"))?;
    s.set_property("position", "relative")?;
    s.set_property("overflow", "hidden")?;
    s.set_property("background", "#1e1e2e")?;
    s.set_property("border-radius", "16px")?;
    s.set_property("box-shadow", "0 8px 32px rgba(0,0,0,0.5)")?;
    Ok(el)
}

fn create_webgl_canvas(
    doc: &Document,
    w: f64,
    h: f64,
) -> Result<(HtmlCanvasElement, WebGlState), JsValue> {
    let canvas: HtmlCanvasElement = doc.create_element("canvas")?.unchecked_into();
    canvas.set_width(w as u32);
    canvas.set_height(h as u32);

    let gl: WebGl2RenderingContext = canvas
        .get_context("webgl2")?
        .expect("browser does not support WebGL2")
        .unchecked_into();

    let vs = compile_gl_shader(&gl, WebGl2RenderingContext::VERTEX_SHADER, GL_VERTEX_SHADER)?;
    let fs = compile_gl_shader(
        &gl,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        GL_FRAGMENT_SHADER,
    )?;

    let program = gl.create_program().expect("create GL program");
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);

    if !gl
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let log = gl.get_program_info_log(&program).unwrap_or_default();
        return Err(JsValue::from_str(&format!("GL program link failed: {log}")));
    }

    gl.use_program(Some(&program));

    // Triangle: position (x, y) + color (r, g, b) per vertex.
    #[rustfmt::skip]
    let vertices: [f32; 15] = [
         0.0,  0.7,    1.0, 0.0, 0.0, // top — red
        -0.6, -0.4,    0.0, 1.0, 0.0, // bottom-left — green
         0.6, -0.4,    0.0, 0.0, 1.0, // bottom-right — blue
    ];

    let buffer = gl.create_buffer().expect("create GL buffer");
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));

    // SAFETY: `vertices` is a local array that outlives the `Float32Array` view
    // and is not moved or modified while the view exists.
    unsafe {
        let array = js_sys::Float32Array::view(&vertices);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &array,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let stride = 5 * 4; // 5 floats × 4 bytes
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_with_i32(0, 2, WebGl2RenderingContext::FLOAT, false, stride, 0);
    gl.enable_vertex_attrib_array(1);
    gl.vertex_attrib_pointer_with_i32(1, 3, WebGl2RenderingContext::FLOAT, false, stride, 8);

    let time_location = gl
        .get_uniform_location(&program, "u_time")
        .expect("u_time uniform not found");

    gl.clear_color(0.0, 0.0, 0.0, 1.0);

    let state = WebGlState {
        context: gl,
        program,
        time_location,
    };

    Ok((canvas, state))
}

fn compile_gl_shader(
    gl: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, JsValue> {
    let shader = gl.create_shader(shader_type).expect("create GL shader");
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if !gl
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let log = gl.get_shader_info_log(&shader).unwrap_or_default();
        return Err(JsValue::from_str(&format!(
            "GL shader compile failed: {log}"
        )));
    }

    Ok(shader)
}

fn render_webgl(gl: &WebGlState, t: f32) {
    let ctx = &gl.context;
    ctx.use_program(Some(&gl.program));
    ctx.uniform1f(Some(&gl.time_location), t);
    ctx.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    ctx.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, 3);
}

fn render_wgpu(gpu: &WgpuState, t: f32) {
    gpu.queue
        .write_buffer(&gpu.time_buffer, 0, bytemuck::bytes_of(&t));

    let frame = match gpu.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return,
    };
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("plasma encoder"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("plasma pass"),
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
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }

    gpu.queue.submit(Some(encoder.finish()));
    frame.present();
}

fn on_tick(state: &Rc<RefCell<AnimState>>, tick: FrameTick) {
    let mut s = state.borrow_mut();

    let build_start = subduction_backend_web::now();

    let safety = Duration(s.scheduler.safety_margin_ticks());
    let hints = subduction_backend_web::compute_present_hints(&tick, safety);
    let plan = s.scheduler.plan(&tick, &hints);

    // Elapsed seconds for animation.
    let elapsed_us = plan.semantic_time.ticks().saturating_sub(s.start_us);
    let elapsed_nanos = s.timebase.ticks_to_nanos(elapsed_us);
    let t = elapsed_nanos as f64 / 1_000_000_000.0;

    // Destructure to satisfy the borrow checker: mutable store + presenter,
    // immutable ids/sizes.
    let AnimState {
        ref mut store,
        ref mut presenter,
        ref layer_ids,
        ref mut sizes,
        ..
    } = *s;
    animate_transforms(store, layer_ids, sizes, t);
    let changes = store.evaluate();
    presenter.apply(store, &changes);

    // Render GPU content.
    let time_f32 = t as f32;
    if let Some(ref gl) = s.webgl {
        render_webgl(gl, time_f32);
    }
    if let Some(ref gpu) = s.wgpu {
        render_wgpu(gpu, time_f32);
    }

    let submitted_at = subduction_backend_web::now();

    let feedback = PresentFeedback::new(&hints, build_start, submitted_at, None);
    s.scheduler.observe(&feedback);
}

fn animate_transforms(
    store: &mut LayerStore,
    layer_ids: &[LayerId],
    sizes: &mut [(f64, f64)],
    t: f64,
) {
    let cx = CONTAINER_W / 2.0;
    let cy = CONTAINER_H / 2.0;

    for (i, &layer_id) in layer_ids.iter().enumerate() {
        let phase = i as f64 * TAU / NUM_LAYERS as f64;
        let radius = 150.0 + 50.0 * (t * 0.5 + phase).sin();
        let angle = t * (0.6 + i as f64 * 0.1) + phase;

        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();

        let rotation = t * 2.0 + phase;

        // Animate bounds on the last layer (amber div) — pulsing size.
        if i == NUM_LAYERS - 1 {
            let (base_w, base_h) = ELEMENT_SPECS[i];
            let scale = 0.75 + 0.5 * (t * 1.2 + phase).sin();
            let w = base_w * scale;
            let h = base_h * scale;
            store.set_bounds(layer_id, Size::new(w, h));
            sizes[layer_id.index() as usize] = (w, h);
        }

        // Center of the element for the centering offset (sizes indexed by slot).
        let (w, h) = sizes[layer_id.index() as usize];
        let half_w = w / 2.0;
        let half_h = h / 2.0;

        // T(x,y) * Rz(angle) * T(-half_w, -half_h)
        // Places the element's visual center at (x,y) and rotates around it.
        let transform = Transform3d::from_translation(x, y, 0.0)
            * Transform3d::from_rotation_z(rotation)
            * Transform3d::from_translation(-half_w, -half_h, 0.0);

        store.set_transform(layer_id, transform);

        let opacity = (0.5 + 0.5 * (t * 1.5 + phase).sin()) as f32;
        store.set_opacity(layer_id, opacity);
    }
}
