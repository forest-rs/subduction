// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WGSL shader source for the compositor pipeline.

pub(crate) const COMPOSITOR_SHADER: &str = r"
struct Uniforms {
    transform: mat4x4<f32>,
    opacity: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

// Unit quad: two triangles covering [0,1]×[0,1].
const QUAD_POS = array<vec2<f32>, 6>(
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let pos = QUAD_POS[vi];
    var out: VertexOutput;
    out.clip_position = uniforms.transform * vec4(pos, 0.0, 1.0);
    out.uv = pos;
    return out;
}

@group(0) @binding(0) var layer_texture: texture_2d<f32>;
@group(0) @binding(1) var layer_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(layer_texture, layer_sampler, in.uv);
    // Scale all channels by opacity — output is premultiplied alpha.
    return color * uniforms.opacity;
}
";
