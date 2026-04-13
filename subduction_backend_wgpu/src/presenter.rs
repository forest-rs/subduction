// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! [`WgpuPresenter`]: a fallback compositor that composites layer textures via wgpu.

use std::collections::HashMap;

use subduction_core::backend::Presenter;
use subduction_core::layer::{ClipShape, FrameChanges, LayerStore, SurfaceId};
use subduction_core::transform::Transform3d;

use crate::pipeline::CompositorPipeline;

/// Minimum uniform buffer offset alignment required by wgpu.
const UNIFORM_ALIGN: u64 = 256;
/// Texture usages required by the presenter compositor itself.
const REQUIRED_LAYER_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING;

/// Per-layer uniform data uploaded to the GPU.
///
/// `transform` is the pre-multiplied `ortho * world_transform * scale_to_layer_size` matrix.
/// `opacity` is the effective opacity. Padding fills to 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniforms {
    transform: [[f32; 4]; 4],
    opacity: f32,
    _pad: [f32; 3],
}

/// GPU state for a single layer: its texture, view, and texture bind group.
struct LayerEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

impl std::fmt::Debug for LayerEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerEntry")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Cached GPU resources for the dynamic uniform buffer, reused across frames.
#[derive(Debug)]
struct UniformCache {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Current buffer capacity in bytes.
    capacity: u64,
}

/// Borrowed render target for a presenter-owned layer texture.
///
/// This exposes both the underlying texture and the default texture view so
/// host renderers can choose the API they need. Backends such as Vello render
/// through the view, while Skia/Ganesh needs the texture handle itself.
#[derive(Clone, Copy, Debug)]
pub struct WgpuLayerTarget<'a> {
    texture: &'a wgpu::Texture,
    view: &'a wgpu::TextureView,
}

impl<'a> WgpuLayerTarget<'a> {
    /// Returns the presenter-owned wgpu texture for this layer.
    pub fn texture(self) -> &'a wgpu::Texture {
        self.texture
    }

    /// Returns the default texture view for this layer.
    pub fn view(self) -> &'a wgpu::TextureView {
        self.view
    }
}

/// A wgpu-based fallback compositor.
///
/// Allocates one texture per layer and composites them in traversal order
/// (back-to-front) with world transforms, opacity, and scissor clipping.
///
/// Layer textures use **premultiplied alpha**. Apps should render premultiplied
/// content into each layer's texture for correct blending.
///
/// Presenter-owned layer textures are host render targets. By default,
/// [`WgpuPresenter::new`] allocates them as `Rgba8Unorm` textures with
/// [`wgpu::TextureUsages::TEXTURE_BINDING`],
/// [`wgpu::TextureUsages::RENDER_ATTACHMENT`], and
/// [`wgpu::TextureUsages::STORAGE_BINDING`] so renderers such as Vello can
/// render into them directly even when the output surface uses a different
/// format such as `Bgra8Unorm`.
///
/// # Usage
///
/// ```rust,ignore
/// let changes = store.evaluate();
/// presenter.apply(&store, &changes);
///
/// // App renders content into each layer's texture.
/// for (surface_id, draw_fn) in &my_surfaces {
///     if let Some(target) = presenter.target_for_surface(*surface_id) {
///         draw_fn(&device, &queue, target.view());
///     }
/// }
///
/// // Composite and present.
/// let output = surface.get_current_texture().unwrap();
/// let output_view = output.texture.create_view(&Default::default());
/// let cmd = presenter.composite(&store, &output_view);
/// queue.submit([cmd]);
/// output.present();
/// ```
#[derive(Debug)]
pub struct WgpuPresenter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: CompositorPipeline,

    /// Per-layer GPU state, keyed by slot index.
    layer_entries: HashMap<u32, LayerEntry>,
    /// Maps `SurfaceId.0` → slot index for content lookup.
    surface_to_slot: HashMap<u32, u32>,
    /// Reverse: slot index → `SurfaceId.0` for O(1) cleanup.
    slot_to_surface: HashMap<u32, u32>,

    /// Default size for new layer textures.
    default_layer_size: (u32, u32),
    /// Size (width, height) of the output surface in pixels.
    output_size: (u32, u32),
    /// Texture format of the output surface.
    output_format: wgpu::TextureFormat,
    /// Texture format of presenter-owned layer textures.
    layer_format: wgpu::TextureFormat,
    /// Texture usages of presenter-owned layer textures.
    layer_usage: wgpu::TextureUsages,

    /// Persistent uniform buffer + bind group, grown as needed.
    uniform_cache: Option<UniformCache>,
}

/// Texture allocation policy for [`WgpuPresenter`].
///
/// This config controls the contract for presenter-owned layer textures:
/// their size, format, and usage flags. The compositor output remains in
/// `output_format`, which is typically the surface format.
///
/// # Migration note
///
/// Prior to this config, `WgpuPresenter` allocated layer textures with the
/// same format as the output surface and only
/// [`wgpu::TextureUsages::TEXTURE_BINDING`] plus
/// [`wgpu::TextureUsages::RENDER_ATTACHMENT`]. The default is now
/// `Rgba8Unorm` plus storage binding so compute-based renderers can write
/// directly into presenter-owned layer textures.
#[derive(Clone, Copy, Debug)]
pub struct WgpuPresenterConfig {
    /// Texture format of the composited output surface.
    pub output_format: wgpu::TextureFormat,
    /// `(width, height)` of the output surface in pixels.
    pub output_size: (u32, u32),
    /// Default `(width, height)` for newly allocated layer textures.
    pub default_layer_size: (u32, u32),
    /// Texture format of presenter-owned layer textures.
    pub layer_format: wgpu::TextureFormat,
    /// Texture usages of presenter-owned layer textures.
    ///
    /// The presenter always requires [`wgpu::TextureUsages::TEXTURE_BINDING`]
    /// so it can sample these textures during composition.
    pub layer_usage: wgpu::TextureUsages,
}

impl WgpuPresenterConfig {
    /// Creates a presenter config with defaults suitable for host-rendered
    /// layer textures.
    pub fn new(
        output_format: wgpu::TextureFormat,
        output_size: (u32, u32),
        default_layer_size: (u32, u32),
    ) -> Self {
        Self {
            output_format,
            output_size,
            default_layer_size,
            layer_format: wgpu::TextureFormat::Rgba8Unorm,
            layer_usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::STORAGE_BINDING,
        }
    }

    /// Overrides the layer texture format.
    #[must_use]
    pub fn with_layer_format(mut self, layer_format: wgpu::TextureFormat) -> Self {
        self.layer_format = layer_format;
        self
    }

    /// Overrides the layer texture usage flags.
    ///
    /// [`wgpu::TextureUsages::TEXTURE_BINDING`] is always added because the
    /// presenter compositor samples each layer texture.
    #[must_use]
    pub fn with_layer_usage(mut self, layer_usage: wgpu::TextureUsages) -> Self {
        self.layer_usage = layer_usage | REQUIRED_LAYER_USAGE;
        self
    }
}

impl WgpuPresenter {
    /// Creates a new wgpu presenter.
    ///
    /// - `device` / `queue`: the wgpu device and queue to use.
    /// - `output_format`: the texture format of the composited output.
    /// - `output_size`: `(width, height)` of the output surface in pixels.
    /// - `default_layer_size`: default `(width, height)` for new layer textures.
    ///
    /// Layer textures default to `Rgba8Unorm` with texture binding, render
    /// attachment, and storage binding. Use [`WgpuPresenter::new_with_config`]
    /// or the builder methods on `Self` to override that contract.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        output_format: wgpu::TextureFormat,
        output_size: (u32, u32),
        default_layer_size: (u32, u32),
    ) -> Self {
        Self::new_with_config(
            device,
            queue,
            WgpuPresenterConfig::new(output_format, output_size, default_layer_size),
        )
    }

    /// Creates a new wgpu presenter with explicit layer texture policy.
    pub fn new_with_config(
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: WgpuPresenterConfig,
    ) -> Self {
        let pipeline = CompositorPipeline::new(&device, config.output_format);

        Self {
            device,
            queue,
            pipeline,
            layer_entries: HashMap::new(),
            surface_to_slot: HashMap::new(),
            slot_to_surface: HashMap::new(),
            default_layer_size: config.default_layer_size,
            output_size: config.output_size,
            output_format: config.output_format,
            layer_format: config.layer_format,
            layer_usage: config.layer_usage | REQUIRED_LAYER_USAGE,
            uniform_cache: None,
        }
    }

    /// Overrides the layer texture format for future allocations.
    ///
    /// This is intended for setup-time configuration before the first
    /// [`Presenter::apply`] call.
    #[must_use]
    pub fn with_layer_format(mut self, layer_format: wgpu::TextureFormat) -> Self {
        self.layer_format = layer_format;
        self
    }

    /// Overrides the layer texture usage flags for future allocations.
    ///
    /// [`wgpu::TextureUsages::TEXTURE_BINDING`] is always preserved because
    /// the presenter compositor samples each layer texture.
    #[must_use]
    pub fn with_layer_usage(mut self, layer_usage: wgpu::TextureUsages) -> Self {
        self.layer_usage = layer_usage | REQUIRED_LAYER_USAGE;
        self
    }

    /// Returns the layer target for a [`SurfaceId`] so the app can render into it.
    ///
    /// The returned target uses [`WgpuPresenter::layer_format`] and the
    /// corresponding texture was allocated with [`WgpuPresenter::layer_usage`].
    pub fn target_for_surface(&self, surface_id: SurfaceId) -> Option<WgpuLayerTarget<'_>> {
        let slot = self.surface_to_slot.get(&surface_id.0)?;
        self.target_for_slot(*slot)
    }

    /// Returns the layer target for a raw slot index.
    ///
    /// The returned target uses [`WgpuPresenter::layer_format`] and the
    /// corresponding texture was allocated with [`WgpuPresenter::layer_usage`].
    pub fn target_for_slot(&self, idx: u32) -> Option<WgpuLayerTarget<'_>> {
        self.layer_entries.get(&idx).map(|entry| WgpuLayerTarget {
            texture: &entry.texture,
            view: &entry.view,
        })
    }

    /// Returns the texture format used for presenter-owned layer textures.
    pub fn layer_format(&self) -> wgpu::TextureFormat {
        self.layer_format
    }

    /// Returns the texture format used for the composited output surface.
    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    /// Returns the texture usage flags used for presenter-owned layer textures.
    pub fn layer_usage(&self) -> wgpu::TextureUsages {
        self.layer_usage
    }

    /// Updates the output surface size (call after reconfiguring the surface).
    pub fn resize_output(&mut self, width: u32, height: u32) {
        self.output_size = (width, height);
    }

    /// Returns a reference to the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns a reference to the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Composites all visible layers into the given output view.
    ///
    /// Call after [`Presenter::apply`] and after the app has rendered content
    /// into each layer's texture.
    pub fn composite(
        &mut self,
        store: &LayerStore,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let traversal = store.traversal_order();

        // Collect visible layers that have a texture allocated.
        let visible: Vec<u32> = traversal
            .iter()
            .copied()
            .filter(|&idx| !store.effective_hidden_at(idx) && self.layer_entries.contains_key(&idx))
            .collect();

        // Build uniform data with 256-byte aligned stride.
        let stride = uniform_stride();
        let required_size = stride * visible.len().max(1) as u64;

        // CPU-side staging buffer.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "uniform buffer size fits in usize on all supported platforms"
        )]
        let mut uniform_data = vec![0_u8; required_size as usize];

        let ortho = ortho_projection(self.output_size.0, self.output_size.1);

        for (i, &idx) in visible.iter().enumerate() {
            let entry = &self.layer_entries[&idx];
            let world = store.world_transform_at(idx);
            let bounds = store.bounds_at(idx);
            let (sw, sh) = if bounds.width > 0.0 && bounds.height > 0.0 {
                (bounds.width, bounds.height)
            } else {
                (f64::from(entry.size.0), f64::from(entry.size.1))
            };
            let scale = Transform3d::from_scale(sw, sh, 1.0);
            let combined = ortho * world * scale;
            let opacity = store.effective_opacity_at(idx);

            let uniforms = LayerUniforms {
                transform: transform_to_f32(&combined),
                opacity,
                _pad: [0.0; 3],
            };

            #[expect(
                clippy::cast_possible_truncation,
                reason = "uniform offset fits in usize"
            )]
            let offset = (stride * i as u64) as usize;
            let bytes = bytemuck::bytes_of(&uniforms);
            uniform_data[offset..offset + bytes.len()].copy_from_slice(bytes);
        }

        // Grow the persistent uniform buffer if needed, or create it.
        let needs_new_buffer = match &self.uniform_cache {
            Some(cache) => cache.capacity < required_size,
            None => true,
        };

        if needs_new_buffer {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compositor uniforms"),
                size: required_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compositor uniform bg"),
                layout: &self.pipeline.uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: Some(
                            wgpu::BufferSize::new(size_of::<LayerUniforms>() as u64)
                                .expect("uniform size is non-zero"),
                        ),
                    }),
                }],
            });
            self.uniform_cache = Some(UniformCache {
                buffer,
                bind_group,
                capacity: required_size,
            });
        }

        let cache = self.uniform_cache.as_ref().expect("just created");
        self.queue.write_buffer(&cache.buffer, 0, &uniform_data);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compositor encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compositor pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
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

            pass.set_pipeline(&self.pipeline.render_pipeline);

            for (i, &idx) in visible.iter().enumerate() {
                let entry = &self.layer_entries[&idx];
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "dynamic offset fits in u32 for any reasonable layer count"
                )]
                let dynamic_offset = (stride * i as u64) as u32;

                pass.set_bind_group(0, &entry.bind_group, &[]);
                pass.set_bind_group(1, &cache.bind_group, &[dynamic_offset]);

                // Apply scissor clip if present.
                if let Some(clip) = store.clip_at(idx) {
                    let rect =
                        clip_to_scissor(clip, &store.world_transform_at(idx), self.output_size);
                    pass.set_scissor_rect(rect.0, rect.1, rect.2, rect.3);
                } else {
                    pass.set_scissor_rect(0, 0, self.output_size.0, self.output_size.1);
                }

                pass.draw(0..6, 0..1);
            }
        }

        encoder.finish()
    }

    /// Creates a layer texture, view, and bind group with the given size.
    fn create_layer_entry(&self, size: (u32, u32)) -> LayerEntry {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer texture"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.layer_format,
            usage: self.layer_usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("layer texture view"),
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer texture bg"),
            layout: &self.pipeline.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.pipeline.sampler),
                },
            ],
        });

        LayerEntry {
            texture,
            view,
            bind_group,
            size,
        }
    }

    /// Removes the surface mapping for a given slot, if any.
    fn remove_surface_for_slot(&mut self, slot: u32) {
        if let Some(surface) = self.slot_to_surface.remove(&slot) {
            self.surface_to_slot.remove(&surface);
        }
    }
}

impl Presenter for WgpuPresenter {
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        // Removals: drop GPU resources for removed layers.
        for &idx in &changes.removed {
            self.layer_entries.remove(&idx);
            self.remove_surface_for_slot(idx);
        }

        // Additions: allocate textures for new layers.
        for &idx in &changes.added {
            let entry = self.create_layer_entry(self.default_layer_size);
            self.layer_entries.insert(idx, entry);
        }

        // Content changes: update surface ↔ slot mapping.
        for &idx in &changes.content {
            self.remove_surface_for_slot(idx);
            if let Some(surface_id) = store.content_at(idx) {
                self.surface_to_slot.insert(surface_id.0, idx);
                self.slot_to_surface.insert(idx, surface_id.0);
            }
        }

        // Bounds changes: resize layer textures.
        for &idx in &changes.bounds {
            let bounds = store.bounds_at(idx);
            if bounds.width > 0.0 && bounds.height > 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "bounds are non-negative pixel dimensions that fit in u32"
                )]
                let size = (bounds.width as u32, bounds.height as u32);
                if self.layer_entries.contains_key(&idx) {
                    let entry = self.create_layer_entry(size);
                    self.layer_entries.insert(idx, entry);
                }
            }
        }

        // Hidden, unhidden, transforms, opacities, clips: no cached state to update.
        // These are read directly from `LayerStore` during `composite()`.
    }
}

/// Returns the dynamic uniform buffer stride (aligned to 256 bytes).
fn uniform_stride() -> u64 {
    let raw = size_of::<LayerUniforms>() as u64;
    // Round up to UNIFORM_ALIGN.
    (raw + UNIFORM_ALIGN - 1) & !(UNIFORM_ALIGN - 1)
}

/// Builds an orthographic projection matrix mapping pixel coords to clip space.
///
/// Maps `(0,0)` at top-left to `(-1, 1)` and `(w, h)` to `(1, -1)` in clip space.
fn ortho_projection(width: u32, height: u32) -> Transform3d {
    let w = f64::from(width);
    let h = f64::from(height);
    Transform3d::from_cols(
        [2.0 / w, 0.0, 0.0, 0.0],
        [0.0, -2.0 / h, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 1.0],
    )
}

/// Converts a `Transform3d` (f64 columns) to f32 column-major array.
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional f64→f32 narrowing for GPU upload"
)]
fn transform_to_f32(t: &Transform3d) -> [[f32; 4]; 4] {
    let c = t.to_cols_array_2d();
    [
        [
            c[0][0] as f32,
            c[0][1] as f32,
            c[0][2] as f32,
            c[0][3] as f32,
        ],
        [
            c[1][0] as f32,
            c[1][1] as f32,
            c[1][2] as f32,
            c[1][3] as f32,
        ],
        [
            c[2][0] as f32,
            c[2][1] as f32,
            c[2][2] as f32,
            c[2][3] as f32,
        ],
        [
            c[3][0] as f32,
            c[3][1] as f32,
            c[3][2] as f32,
            c[3][3] as f32,
        ],
    ]
}

/// Converts a clip shape to a scissor rect `(x, y, width, height)` in output pixels.
///
/// Transforms the clip rectangle's corners by the 2D affine part of the world
/// transform, then takes the axis-aligned bounding box and clamps to the output
/// dimensions. For `RoundedRect`, falls back to the bounding rect.
///
/// # Panics
///
/// Debug-asserts that the transform has no perspective component (i.e. the
/// bottom row is `[0, 0, 0, 1]`). Perspective transforms produce incorrect
/// scissor rects; full perspective clipping is not yet implemented.
fn clip_to_scissor(
    clip: ClipShape,
    world: &Transform3d,
    output_size: (u32, u32),
) -> (u32, u32, u32, u32) {
    let c = world.to_cols_array_2d();
    debug_assert!(
        (c[0][3]).abs() < 1e-6 && (c[1][3]).abs() < 1e-6 && (c[3][3] - 1.0).abs() < 1e-6,
        "clip_to_scissor does not support perspective transforms"
    );

    let rect = match clip {
        ClipShape::Rect(r) => r,
        ClipShape::RoundedRect(rr) => rr.rect(),
    };

    // Transform the four corners and take the AABB.
    let corners = [
        (rect.x0, rect.y0),
        (rect.x1, rect.y0),
        (rect.x1, rect.y1),
        (rect.x0, rect.y1),
    ];

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for (px, py) in corners {
        let tx = c[0][0] * px + c[1][0] * py + c[3][0];
        let ty = c[0][1] * px + c[1][1] * py + c[3][1];
        min_x = min_x.min(tx);
        min_y = min_y.min(ty);
        max_x = max_x.max(tx);
        max_y = max_y.max(ty);
    }

    // Clamp to output bounds.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped non-negative f64 pixel coords fit in u32"
    )]
    let (x0, y0, x1, y1) = (
        (min_x.max(0.0) as u32).min(output_size.0),
        (min_y.max(0.0) as u32).min(output_size.1),
        (max_x.ceil().max(0.0) as u32).min(output_size.0),
        (max_y.ceil().max(0.0) as u32).min(output_size.1),
    );

    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presenter_config_defaults_to_vello_compatible_layers() {
        let config =
            WgpuPresenterConfig::new(wgpu::TextureFormat::Bgra8Unorm, (800, 600), (256, 256));

        assert_eq!(config.output_format, wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(config.layer_format, wgpu::TextureFormat::Rgba8Unorm);
        assert!(
            config
                .layer_usage
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        assert!(
            config
                .layer_usage
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        );
        assert!(
            config
                .layer_usage
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        );
    }

    #[test]
    fn presenter_config_preserves_required_sampling_usage() {
        let config =
            WgpuPresenterConfig::new(wgpu::TextureFormat::Bgra8Unorm, (800, 600), (256, 256))
                .with_layer_format(wgpu::TextureFormat::Bgra8Unorm)
                .with_layer_usage(wgpu::TextureUsages::STORAGE_BINDING);

        assert_eq!(config.layer_format, wgpu::TextureFormat::Bgra8Unorm);
        assert!(
            config
                .layer_usage
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        assert!(
            config
                .layer_usage
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        );
    }

    #[test]
    fn ortho_maps_origin_to_top_left() {
        let proj = ortho_projection(800, 600);
        let c = proj.to_cols_array_2d();
        // (0, 0) should map to (-1, 1) in clip space.
        let x = c[3][0]; // translation x
        let y = c[3][1]; // translation y
        assert!((x - (-1.0)).abs() < 1e-6);
        assert!((y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ortho_maps_bottom_right() {
        let proj = ortho_projection(800, 600);
        let c = proj.to_cols_array_2d();
        // (800, 600) should map to (1, -1).
        let x = c[0][0] * 800.0 + c[3][0];
        let y = c[1][1] * 600.0 + c[3][1];
        assert!((x - 1.0).abs() < 1e-6);
        assert!((y - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn uniform_stride_is_aligned() {
        let stride = uniform_stride();
        assert_eq!(stride % UNIFORM_ALIGN, 0);
        assert!(stride >= size_of::<LayerUniforms>() as u64);
    }

    #[test]
    fn transform_to_f32_identity() {
        let f = transform_to_f32(&Transform3d::IDENTITY);
        assert_eq!(f[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(f[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(f[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(f[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn clip_identity_transform() {
        let clip = ClipShape::Rect(kurbo::Rect::new(10.0, 20.0, 100.0, 80.0));
        let (x, y, w, h) = clip_to_scissor(clip, &Transform3d::IDENTITY, (800, 600));
        assert_eq!(x, 10);
        assert_eq!(y, 20);
        assert_eq!(w, 90);
        assert_eq!(h, 60);
    }

    #[test]
    fn clip_clamped_to_output() {
        let clip = ClipShape::Rect(kurbo::Rect::new(-50.0, -50.0, 900.0, 700.0));
        let (x, y, w, h) = clip_to_scissor(clip, &Transform3d::IDENTITY, (800, 600));
        assert_eq!(x, 0);
        assert_eq!(y, 0);
        assert_eq!(w, 800);
        assert_eq!(h, 600);
    }

    #[test]
    #[should_panic(expected = "perspective")]
    fn clip_rejects_perspective() {
        let mut t = Transform3d::IDENTITY;
        t.cols[0][3] = 0.5; // perspective component
        let clip = ClipShape::Rect(kurbo::Rect::new(0.0, 0.0, 100.0, 100.0));
        clip_to_scissor(clip, &t, (800, 600));
    }
}
