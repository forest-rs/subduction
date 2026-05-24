// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! [`WgpuPresenter`]: a fallback compositor that composites surface textures via wgpu.

use std::collections::HashMap;

use color::LinearSrgb;
use subduction_core::backend::Presenter;
use subduction_core::layer::{ClipShape, FrameChanges, LayerStore, SurfaceId};
use subduction_core::output::{Backdrop, Color};
use subduction_core::transform::Transform3d;

use crate::pipeline::CompositorPipeline;

/// Minimum uniform buffer offset alignment required by wgpu.
const UNIFORM_ALIGN: u64 = 256;
/// Texture usages required by the presenter compositor itself.
const REQUIRED_SURFACE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING;

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

/// GPU state for one [`SurfaceId`].
struct SurfaceEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

impl std::fmt::Debug for SurfaceEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceEntry")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Current frame attachment state between surface content and layer slots.
///
/// Layer slots are raw `LayerStore` storage rows from `FrameChanges`; they are
/// useful presenter plumbing, but they are not content identity. This table
/// keeps the one-current-slot invariant for each surface without making slots
/// part of the public render-target API.
#[derive(Debug, Default)]
struct SurfaceAttachments {
    attached_surface_to_slot: HashMap<SurfaceId, u32>,
    slot_to_attached_surface: HashMap<u32, SurfaceId>,
}

impl SurfaceAttachments {
    fn surface_for_slot(&self, slot: u32) -> Option<SurfaceId> {
        self.slot_to_attached_surface.get(&slot).copied()
    }

    fn is_attached_to_slot(&self, surface_id: SurfaceId, slot: u32) -> bool {
        self.attached_surface_to_slot.get(&surface_id).copied() == Some(slot)
    }

    fn attach(&mut self, surface_id: SurfaceId, slot: u32) {
        self.detach_slot(slot);
        self.detach_surface(surface_id);
        self.attached_surface_to_slot.insert(surface_id, slot);
        self.slot_to_attached_surface.insert(slot, surface_id);
    }

    fn detach_slot(&mut self, slot: u32) -> Option<SurfaceId> {
        let surface_id = self.slot_to_attached_surface.remove(&slot)?;
        if self.attached_surface_to_slot.get(&surface_id).copied() == Some(slot) {
            self.attached_surface_to_slot.remove(&surface_id);
        }
        Some(surface_id)
    }

    fn detach_surface(&mut self, surface_id: SurfaceId) -> Option<u32> {
        let slot = self.attached_surface_to_slot.remove(&surface_id)?;
        if self.slot_to_attached_surface.get(&slot).copied() == Some(surface_id) {
            self.slot_to_attached_surface.remove(&slot);
        }
        Some(slot)
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

/// Borrowed render target for a presenter-owned surface texture.
///
/// This exposes both the underlying texture and the default texture view so
/// host renderers can choose the API they need. Backends such as Vello render
/// through the view, while Skia/Ganesh needs the texture handle itself.
#[derive(Clone, Copy, Debug)]
pub struct WgpuSurfaceTarget<'a> {
    texture: &'a wgpu::Texture,
    view: &'a wgpu::TextureView,
}

impl<'a> WgpuSurfaceTarget<'a> {
    /// Returns the presenter-owned wgpu texture for this surface.
    pub fn texture(self) -> &'a wgpu::Texture {
        self.texture
    }

    /// Returns the default texture view for this surface.
    pub fn view(self) -> &'a wgpu::TextureView {
        self.view
    }
}

/// Final compositing root for a wgpu-presented scene.
#[derive(Debug, PartialEq)]
pub struct LayerRoot {
    output_format: wgpu::TextureFormat,
    size: (u32, u32),
    backdrop: Backdrop,
}

impl LayerRoot {
    /// Creates a new layer root for a final compositing target.
    #[must_use]
    pub fn new(output_format: wgpu::TextureFormat, size: (u32, u32)) -> Self {
        Self {
            output_format,
            size,
            backdrop: Backdrop::None,
        }
    }

    /// Returns this root configured with the given backdrop.
    #[must_use]
    pub fn with_backdrop(mut self, backdrop: Backdrop) -> Self {
        self.set_backdrop(backdrop);
        self
    }

    /// Returns this root configured with a solid backdrop color.
    #[must_use]
    pub fn with_backdrop_color(mut self, color: Color) -> Self {
        self.set_backdrop_color(color);
        self
    }

    /// Returns the composited output format.
    #[must_use]
    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    /// Returns the output size in pixels.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Updates the output size in pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.size = (width, height);
    }

    /// Returns the configured backdrop policy.
    #[must_use]
    pub fn backdrop(&self) -> Backdrop {
        self.backdrop
    }

    /// Updates the backdrop policy.
    pub fn set_backdrop(&mut self, backdrop: Backdrop) {
        self.backdrop = backdrop;
    }

    /// Updates the backdrop to a solid color.
    pub fn set_backdrop_color(&mut self, color: Color) {
        self.set_backdrop(Backdrop::Color(color));
    }

    /// Removes any explicit backdrop.
    pub fn remove_backdrop(&mut self) {
        self.set_backdrop(Backdrop::None);
    }
}

/// A wgpu-based fallback compositor.
///
/// Allocates one texture per [`SurfaceId`] and composites attached surfaces in
/// layer traversal order (back-to-front) with world transforms, opacity, and
/// scissor clipping.
///
/// Layer slots remain private presenter plumbing: [`FrameChanges`] reports raw
/// slot indices so the presenter can read `LayerStore::*_at(idx)` efficiently,
/// but render targets are keyed by surface identity. A contentless grouping
/// layer does not allocate a texture.
///
/// Presenter-owned surface textures are host render targets. By default,
/// [`WgpuPresenter::new`] allocates them as `Rgba8Unorm` textures with
/// [`wgpu::TextureUsages::TEXTURE_BINDING`],
/// [`wgpu::TextureUsages::RENDER_ATTACHMENT`], and
/// [`wgpu::TextureUsages::STORAGE_BINDING`] so renderers such as Vello can
/// render into them directly even when the output surface uses a different
/// format such as `Bgra8Unorm`.
///
/// Surface textures live until [`release_surface`](Self::release_surface) is
/// called. Detaching a surface from a layer or destroying that layer only
/// removes the current attachment; it does not destroy the surface texture.
///
/// Surface textures use **premultiplied alpha**. Apps should render
/// premultiplied content into each surface texture for correct blending.
///
/// # Usage
///
/// ```rust,ignore
/// let root = LayerRoot::new(surface_format, (width, height))
///     .with_backdrop_color(scene_backdrop);
/// let mut presenter = WgpuPresenter::new(device, queue, root, (256, 256));
///
/// let changes = store.evaluate();
/// presenter.apply(&store, &changes);
///
/// // App renders content into each surface texture.
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
    root: LayerRoot,

    /// Per-surface GPU state, keyed by content identity.
    surface_entries: HashMap<SurfaceId, SurfaceEntry>,
    /// Current frame mapping between surface IDs and layer slots.
    attachments: SurfaceAttachments,

    /// Default size for new surface textures.
    default_surface_size: (u32, u32),
    /// Texture format of presenter-owned surface textures.
    surface_format: wgpu::TextureFormat,
    /// Texture usages of presenter-owned surface textures.
    surface_usage: wgpu::TextureUsages,

    /// Persistent uniform buffer + bind group, grown as needed.
    uniform_cache: Option<UniformCache>,
}

/// Texture allocation policy for [`WgpuPresenter`].
///
/// This config controls the contract for presenter-owned surface textures:
/// their size, format, and usage flags. The final compositing target is owned
/// separately by [`LayerRoot`].
///
/// # Migration note
///
/// Prior to this config, `WgpuPresenter` allocated surface textures with the
/// same format as the output surface and only
/// [`wgpu::TextureUsages::TEXTURE_BINDING`] plus
/// [`wgpu::TextureUsages::RENDER_ATTACHMENT`]. The default is now
/// `Rgba8Unorm` plus storage binding so compute-based renderers can write
/// directly into presenter-owned surface textures.
#[derive(Clone, Copy, Debug)]
pub struct WgpuPresenterConfig {
    /// Default `(width, height)` for newly allocated surface textures.
    pub default_surface_size: (u32, u32),
    /// Texture format of presenter-owned surface textures.
    pub surface_format: wgpu::TextureFormat,
    /// Texture usages of presenter-owned surface textures.
    ///
    /// The presenter always requires [`wgpu::TextureUsages::TEXTURE_BINDING`]
    /// so it can sample these textures during composition.
    pub surface_usage: wgpu::TextureUsages,
}

impl WgpuPresenterConfig {
    /// Creates a presenter config with defaults suitable for host-rendered
    /// surface textures.
    pub fn new(default_surface_size: (u32, u32)) -> Self {
        Self {
            default_surface_size,
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
            surface_usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::STORAGE_BINDING,
        }
    }

    /// Overrides the surface texture format.
    #[must_use]
    pub fn with_surface_format(mut self, surface_format: wgpu::TextureFormat) -> Self {
        self.surface_format = surface_format;
        self
    }

    /// Overrides the surface texture usage flags.
    ///
    /// [`wgpu::TextureUsages::TEXTURE_BINDING`] is always added because the
    /// presenter compositor samples each surface texture.
    #[must_use]
    pub fn with_surface_usage(mut self, surface_usage: wgpu::TextureUsages) -> Self {
        self.surface_usage = surface_usage | REQUIRED_SURFACE_USAGE;
        self
    }
}

impl WgpuPresenter {
    /// Creates a new wgpu presenter.
    ///
    /// - `device` / `queue`: the wgpu device and queue to use.
    /// - `root`: the final compositing root.
    /// - `default_surface_size`: default `(width, height)` for new surface textures.
    ///
    /// Surface textures default to `Rgba8Unorm` with texture binding, render
    /// attachment, and storage binding. Use [`WgpuPresenter::new_with_config`]
    /// or the builder methods on `Self` to override that contract.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        root: LayerRoot,
        default_surface_size: (u32, u32),
    ) -> Self {
        Self::new_with_config(
            device,
            queue,
            root,
            WgpuPresenterConfig::new(default_surface_size),
        )
    }

    /// Creates a new wgpu presenter with explicit surface texture policy.
    pub fn new_with_config(
        device: wgpu::Device,
        queue: wgpu::Queue,
        root: LayerRoot,
        config: WgpuPresenterConfig,
    ) -> Self {
        let pipeline = CompositorPipeline::new(&device, root.output_format());

        Self {
            device,
            queue,
            pipeline,
            root,
            surface_entries: HashMap::new(),
            attachments: SurfaceAttachments::default(),
            default_surface_size: config.default_surface_size,
            surface_format: config.surface_format,
            surface_usage: config.surface_usage | REQUIRED_SURFACE_USAGE,
            uniform_cache: None,
        }
    }

    /// Overrides the surface texture format for future allocations.
    ///
    /// This is intended for setup-time configuration before the first
    /// [`Presenter::apply`] call.
    #[must_use]
    pub fn with_surface_format(mut self, surface_format: wgpu::TextureFormat) -> Self {
        self.surface_format = surface_format;
        self
    }

    /// Overrides the surface texture usage flags for future allocations.
    ///
    /// [`wgpu::TextureUsages::TEXTURE_BINDING`] is always preserved because
    /// the presenter compositor samples each surface texture.
    #[must_use]
    pub fn with_surface_usage(mut self, surface_usage: wgpu::TextureUsages) -> Self {
        self.surface_usage = surface_usage | REQUIRED_SURFACE_USAGE;
        self
    }

    /// Returns the render target for a [`SurfaceId`].
    ///
    /// Targets are allocated when [`apply`](Presenter::apply) first observes
    /// `surface_id` attached as layer content. A target may remain available
    /// after the surface is detached from a layer; call
    /// [`release_surface`](Self::release_surface) to drop it.
    pub fn target_for_surface(&self, surface_id: SurfaceId) -> Option<WgpuSurfaceTarget<'_>> {
        self.surface_entries
            .get(&surface_id)
            .map(|entry| WgpuSurfaceTarget {
                texture: &entry.texture,
                view: &entry.view,
            })
    }

    /// Releases the presenter-owned render target for a surface.
    ///
    /// This removes any current attachment mapping and drops the cached wgpu
    /// texture. It does not mutate [`LayerStore`]; callers should detach the
    /// surface from any layer before releasing it, or intentionally accept that
    /// the store still names a surface the presenter no longer has cached.
    ///
    /// Returns `true` if either a target or attachment mapping was removed.
    pub fn release_surface(&mut self, surface_id: SurfaceId) -> bool {
        let had_entry = self.surface_entries.remove(&surface_id).is_some();
        let had_attachment = self.attachments.detach_surface(surface_id).is_some();
        had_entry || had_attachment
    }

    /// Returns the texture format used for presenter-owned surface textures.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Returns the texture usage flags used for presenter-owned surface textures.
    pub fn surface_usage(&self) -> wgpu::TextureUsages {
        self.surface_usage
    }

    /// Returns the scene root.
    pub fn root(&self) -> &LayerRoot {
        &self.root
    }

    /// Returns a mutable reference to the scene root.
    pub fn root_mut(&mut self) -> &mut LayerRoot {
        &mut self.root
    }

    /// Returns a reference to the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns a reference to the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Composites all visible attached surfaces into the given output view.
    ///
    /// Call after [`Presenter::apply`] and after the app has rendered content
    /// into each attached surface texture.
    pub fn composite(
        &mut self,
        store: &LayerStore,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let traversal = store.traversal_order();

        // Collect visible attached surfaces in layer traversal order.
        let visible: Vec<(u32, SurfaceId)> = traversal
            .iter()
            .copied()
            .filter_map(|idx| {
                if store.effective_hidden_at(idx) {
                    return None;
                }
                let surface_id = store.content_at(idx)?;
                if self.attachments.is_attached_to_slot(surface_id, idx)
                    && self.surface_entries.contains_key(&surface_id)
                {
                    Some((idx, surface_id))
                } else {
                    None
                }
            })
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

        let output_size = self.root.size();
        let ortho = ortho_projection(output_size.0, output_size.1);

        for (i, &(idx, surface_id)) in visible.iter().enumerate() {
            let entry = &self.surface_entries[&surface_id];
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
                        load: wgpu::LoadOp::Clear(backdrop_clear_color(self.root.backdrop())),
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

            for (i, &(idx, surface_id)) in visible.iter().enumerate() {
                let entry = &self.surface_entries[&surface_id];
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "dynamic offset fits in u32 for any reasonable layer count"
                )]
                let dynamic_offset = (stride * i as u64) as u32;

                pass.set_bind_group(0, &entry.bind_group, &[]);
                pass.set_bind_group(1, &cache.bind_group, &[dynamic_offset]);

                // Apply scissor clip if present.
                if let Some(clip) = store.clip_at(idx) {
                    let rect = clip_to_scissor(clip, &store.world_transform_at(idx), output_size);
                    pass.set_scissor_rect(rect.0, rect.1, rect.2, rect.3);
                } else {
                    pass.set_scissor_rect(0, 0, output_size.0, output_size.1);
                }

                pass.draw(0..6, 0..1);
            }
        }

        encoder.finish()
    }

    /// Creates a surface texture, view, and bind group with the given size.
    fn create_surface_entry(&self, size: (u32, u32)) -> SurfaceEntry {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("surface texture"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: self.surface_usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("surface texture view"),
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface texture bg"),
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

        SurfaceEntry {
            texture,
            view,
            bind_group,
            size,
        }
    }

    fn surface_size_for_slot(&self, store: &LayerStore, slot: u32) -> (u32, u32) {
        let bounds = store.bounds_at(slot);
        if bounds.width > 0.0 && bounds.height > 0.0 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "bounds are non-negative pixel dimensions that fit in u32"
            )]
            {
                (bounds.width as u32, bounds.height as u32)
            }
        } else {
            self.default_surface_size
        }
    }

    fn ensure_surface_entry(&mut self, surface_id: SurfaceId, size: (u32, u32)) {
        if self.surface_entries.contains_key(&surface_id) {
            return;
        }

        let entry = self.create_surface_entry(size);
        self.surface_entries.insert(surface_id, entry);
    }

    fn resize_surface_entry(&mut self, surface_id: SurfaceId, size: (u32, u32)) {
        if self
            .surface_entries
            .get(&surface_id)
            .map(|entry| entry.size)
            == Some(size)
        {
            return;
        }

        let entry = self.create_surface_entry(size);
        self.surface_entries.insert(surface_id, entry);
    }
}

impl Presenter for WgpuPresenter {
    fn apply(&mut self, store: &LayerStore, changes: &FrameChanges) {
        // Removals detach slots. Surface resources are released explicitly.
        for &idx in &changes.removed {
            self.attachments.detach_slot(idx);
        }

        // Content changes update the current surface ↔ slot attachment.
        for &idx in &changes.content {
            self.attachments.detach_slot(idx);
            if let Some(surface_id) = store.content_at(idx) {
                self.attachments.attach(surface_id, idx);
                let size = self.surface_size_for_slot(store, idx);
                self.ensure_surface_entry(surface_id, size);
            }
        }

        // Bounds changes resize the attached surface texture, if any.
        for &idx in &changes.bounds {
            if let Some(surface_id) = self.attachments.surface_for_slot(idx) {
                let size = self.surface_size_for_slot(store, idx);
                self.resize_surface_entry(surface_id, size);
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

/// Converts output backdrop policy into a premultiplied clear color.
fn backdrop_clear_color(backdrop: Backdrop) -> wgpu::Color {
    match backdrop {
        Backdrop::None => wgpu::Color::TRANSPARENT,
        Backdrop::Color(color) => {
            let premultiplied = color.convert::<LinearSrgb>().premultiply();
            let [r, g, b, a] = premultiplied.components;
            wgpu::Color {
                r: f64::from(r),
                g: f64::from(g),
                b: f64::from(b),
                a: f64::from(a),
            }
        }
    }
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
    use subduction_core::output::Color;

    fn surface_id(index: u32) -> SurfaceId {
        SurfaceId::from_raw_parts(index, 0)
    }

    #[test]
    fn surface_attachments_move_surface_to_new_slot() {
        let mut attachments = SurfaceAttachments::default();
        let surface = surface_id(1);

        attachments.attach(surface, 10);
        attachments.attach(surface, 20);

        assert_eq!(attachments.surface_for_slot(10), None);
        assert_eq!(attachments.surface_for_slot(20), Some(surface));
        assert!(attachments.is_attached_to_slot(surface, 20));
        assert!(!attachments.is_attached_to_slot(surface, 10));
    }

    #[test]
    fn surface_attachments_replace_slot_surface() {
        let mut attachments = SurfaceAttachments::default();
        let old_surface = surface_id(1);
        let new_surface = surface_id(2);

        attachments.attach(old_surface, 10);
        attachments.attach(new_surface, 10);

        assert_eq!(attachments.surface_for_slot(10), Some(new_surface));
        assert!(attachments.is_attached_to_slot(new_surface, 10));
        assert!(!attachments.is_attached_to_slot(old_surface, 10));
    }

    #[test]
    fn surface_attachments_detach_surface_clears_reverse_mapping() {
        let mut attachments = SurfaceAttachments::default();
        let surface = surface_id(1);

        attachments.attach(surface, 10);
        assert_eq!(attachments.detach_surface(surface), Some(10));

        assert_eq!(attachments.surface_for_slot(10), None);
        assert!(!attachments.is_attached_to_slot(surface, 10));
    }

    #[test]
    fn surface_attachments_detach_old_slot_after_move_preserves_new_slot() {
        let mut attachments = SurfaceAttachments::default();
        let surface = surface_id(1);

        attachments.attach(surface, 10);
        attachments.attach(surface, 20);

        assert_eq!(attachments.detach_slot(10), None);
        assert_eq!(attachments.surface_for_slot(20), Some(surface));
        assert!(attachments.is_attached_to_slot(surface, 20));
    }

    #[test]
    fn layer_root_mutates_in_place() {
        let mut root = LayerRoot::new(wgpu::TextureFormat::Bgra8Unorm, (640, 480));
        let backdrop_color = Color::from_rgba8(0x1e, 0x1e, 0x2e, 0xff);

        root.resize(800, 600);
        root.set_backdrop_color(backdrop_color);

        assert_eq!(root.size(), (800, 600));
        assert_eq!(root.backdrop(), Backdrop::Color(backdrop_color));

        root.remove_backdrop();

        assert_eq!(root.backdrop(), Backdrop::None);
    }

    #[test]
    fn presenter_config_defaults_to_vello_compatible_surfaces() {
        let config = WgpuPresenterConfig::new((256, 256));

        assert_eq!(config.surface_format, wgpu::TextureFormat::Rgba8Unorm);
        assert!(
            config
                .surface_usage
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        assert!(
            config
                .surface_usage
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        );
        assert!(
            config
                .surface_usage
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        );
    }

    #[test]
    fn presenter_config_preserves_required_sampling_usage() {
        let config = WgpuPresenterConfig::new((256, 256))
            .with_surface_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_surface_usage(wgpu::TextureUsages::STORAGE_BINDING);

        assert_eq!(config.surface_format, wgpu::TextureFormat::Bgra8Unorm);
        assert!(
            config
                .surface_usage
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        assert!(
            config
                .surface_usage
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        );
    }

    #[test]
    fn solid_backdrop_clears_with_premultiplied_linear_color() {
        let clear = backdrop_clear_color(Backdrop::Color(Color::new([0.5, 0.25, 0.75, 0.5])));

        assert!((clear.r - 0.107_020_57).abs() < 1e-6);
        assert!((clear.g - 0.025_438_04).abs() < 1e-6);
        assert!((clear.b - 0.261_260_78).abs() < 1e-6);
        assert!((clear.a - 0.5).abs() < 1e-6);
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
