//! The `Renderer` facade: owns the entire Vulkan session and exposes a tiny
//! API the engine drives (`new`, `upload_chunks`, `remove_chunk`, `draw_frame`,
//! `capture_frame`, `resize`, `Drop`).
//!
//! Internal structure (all Vulkan handles live on `Renderer`):
//! - instance + optional debug messenger
//! - surface (from a raw window handle via `ash-window`)
//! - physical device + queue families (graphics + present, possibly one)
//! - logical device + graphics/present queues
//! - memory allocator (`gpu-allocator`)
//! - swapchain + image views + depth image + framebuffers
//! - render pass + chunk graphics pipeline + pipeline layout + descriptor sets
//! - atlas texture + sampler
//! - per-frame (Ã—2 in flight): command buffer, fences/semaphores, camera UBO
//! - per-chunk: vertex + index buffers in a `HashMap`

mod device;
mod indirect;
mod init;
mod pipeline;
mod swapchain;

use device::QueueFamilies;
use pipeline::{CameraUbo, FogUbo, ShadowUbo, SkyUbo};
pub(crate) use pipeline::spirv_to_u32;
use swapchain::create_framebuffer_with;

use crate::MaterialTable;

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use ash::vk;
use ash::{Entry, Instance as AshInstance};
use glam::{Mat4, Vec3};
use parking_lot::RwLock;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use std::path::{Path};

use voxel_core::{
    math::{chunk_origin, ChunkPos},
    Camera, Frustum,
};

use crate::buffer::{GpuBuffer, GpuImage};
use crate::alloc::Alloc;
use crate::atlas::{build_atlas_with_textures, build_atlas_with_packs};
use crate::texture::{begin_one_time, end_and_submit, transition_image_layout, AtlasTexture};
use crate::ui::UiDrawData;

use gpu_allocator::MemoryLocation;

const FRAMES_IN_FLIGHT: usize = 2;
const GPU_TIMESTAMP_COUNT: u32 = 8; // frame_start, shadow_end, sky_end, opaque_end, transparent_end, ui_end, main_pass_end, post_end

/// GPU timing results for a single frame (in milliseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimings {
    pub frame_ms: f32,
    pub shadow_ms: f32,
    pub sky_ms: f32,
    pub opaque_ms: f32,
    pub transparent_ms: f32,
    pub ui_ms: f32,
    pub post_ms: f32,
}

/// Which render pass a chunk mesh belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshPass {
    Opaque,
    Transparent,
}

/// Vertex + index data for one chunk pass, as raw bytes ready to upload.
pub struct ChunkUpload {
    pub pos: ChunkPos,
    pub pass: MeshPass,
    /// Vertex bytes (24 bytes each, layout = [`crate::Vertex`]).
    pub vertices: Vec<u8>,
    /// Index bytes (4 bytes each, `u32`).
    pub indices: Vec<u8>,
    pub index_count: u32,
}

/// Renderer configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct RendererConfig {
    /// Enable Vulkan validation layers + debug messenger (debug builds).
    pub validation: bool,
    /// Clear colour (sky) in linear RGB 0..1.
    pub clear_color: [f32; 4],
    /// Use FIFO (vsync) present mode. If false, prefer MAILBOX.
    pub vsync: bool,
    /// Fog colour (linear RGB) and density (unused placeholder).
    pub fog_color: [f32; 3],
    /// Distance at which fog fully obscures chunks.
    pub fog_distance: f32,
    /// Directory containing PNG texture overrides (filenames `<tile_index>.png`).
    /// If `None` or the directory doesn't exist, the procedural atlas is used.
    pub textures_dir: Option<std::path::PathBuf>,
    /// Directory containing texture pack `.zip` files.
    /// Texture packs override base textures on a per-tile basis.
    pub texture_packs_dir: Option<std::path::PathBuf>,
    /// Directory containing the GLSL shader sources (`*.vert` / `*.frag`).
    /// When `None`, no shader hot-reload is performed and a fresh pipeline is
    /// compiled at startup via `build.rs` (the legacy baked-in path).
    pub shader_dir: Option<std::path::PathBuf>,
    /// Soft-particle fade radius in world units (e.g. 0.3 m). Particles
    /// within this distance of scene geometry fade their alpha to zero
    /// along the intersection, eliminating hard edges.
    pub particle_softness: f32,
    /// MSAA sample count (1, 2, 4, or 8). Set to 1 to disable MSAA.
    pub msaa_samples: u32,
    /// Enable hardware occlusion culling for chunk rendering.
    pub occlusion_culling: bool,
    /// Enable screen-space ambient occlusion (SSAO). Requires MSAA disabled.
    pub ssao_enabled: bool,
    /// Enable the Phase-1 GPU-driven chunk rendering pipeline: indirect
    /// multi-draw + compute-shader frustum culling + bindless origins SSBO.
    /// When `true`, the per-chunk `vkCmdDrawIndexed` loop is replaced by one
    /// `vkCmdDrawIndexedIndirect` per pass. When `false` (default), the legacy
    /// per-chunk path runs. Requires `multi_draw_indirect` +
    /// `draw_indirect_first_instance` device features (auto-enabled).
    pub gpu_driven: bool,
    /// Enable the Phase-2 GPU compute chunk mesher. Requires `gpu_driven`.
    /// When `true`, distant chunks are meshed on the GPU via a compute shader
    /// (naive face extraction, no greedy merge) instead of the CPU mesher.
    pub gpu_meshing: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            validation: false,
            clear_color: [0.52, 0.72, 0.95, 1.0],
            vsync: true,
            fog_color: [0.62, 0.80, 0.96],
            fog_distance: 320.0,
            textures_dir: None,
            texture_packs_dir: None,
            shader_dir: None,
            particle_softness: 0.3,
            msaa_samples: 4,
            occlusion_culling: true,
            ssao_enabled: true,
            gpu_driven: false,
            gpu_meshing: false,
        }
    }
}

/// Per-chunk GPU buffers for a single render pass.
struct PassBuffers {
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    index_count: u32,
}

/// Per-chunk GPU buffers, split by render pass. A chunk at a water-land border
/// has both; a purely inland chunk has only opaque.
struct ChunkBuffers {
    opaque: Option<PassBuffers>,
    transparent: Option<PassBuffers>,
}

impl ChunkBuffers {
    fn new() -> Self {
        Self {
            opaque: None,
            transparent: None,
        }
    }

    fn destroy(self, device: &ash::Device, alloc: &Alloc) {
        if let Some(b) = self.opaque {
            b.vbo.destroy(device, alloc);
            b.ibo.destroy(device, alloc);
        }
        if let Some(b) = self.transparent {
            b.vbo.destroy(device, alloc);
            b.ibo.destroy(device, alloc);
        }
    }
}

/// Items that cannot be destroyed the moment we'd like to because the GPU
/// may still be referencing them (in flight on the graphics queue). We
/// tag each with the `frame_counter` value at submission time and drain
/// them in `drain_pending_destructions` once the counter has advanced by
/// at least `FRAMES_IN_FLIGHT` — by then the corresponding per-frame
/// in-flight fence has signalled, so the GPU is past every resource that
/// batch submitted.
///
/// This is the heart of the "two freezes + crash" fix:
///  * No more `device.wait_for_fences(..., u64::MAX)` in `upload_chunks`
///    (which previously stalled the main thread for many seconds during
///    initial world streaming — perceived as freeze #2).
///  * No more eager `vbo.destroy()` / `ibo.destroy()` /
///    `staging.destroy()` immediately after submit (which previously
///    released GPU-side memory mid-draw — Vulkan validation dropped
///    the device and produced the crash).
enum PendingDestroy {
    ChunkValue(PassBuffers),
    Staging(GpuBuffer),
    /// Per-batch upload command buffer + its associated fence.
    /// Allocated from `command_pool` + a fresh `vk::Fence`, submitted
    /// once with `fence`, freed in `drain_pending_destructions` after
    /// `wait_for_fences(fence)` returns signaled. The cmd buffer rule
    /// (`VUID-vkFreeCommandBuffers-pCommandBuffers-00047` — "must not
    /// be in pending state") is stricter than the buffer rule
    /// (`vkDestroyBuffer` tolerates FIFO/FRAMES_IN_FLIGHT geographic
    /// heuristics), so we cannot rely on the heap-ordering hint that
    /// the ChunkValue/Staging defers use — we must observe the fence.
    CommandBuffer {
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
    },
}

struct Frame {
    cmd: vk::CommandBuffer,
    in_flight_fence: vk::Fence,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    camera_ubo: GpuBuffer,
    shadow_ubo: GpuBuffer,
    descriptor_set: vk::DescriptorSet,
    /// Tile remap UBO + descriptor set (set 1, binding 0 of the chunk
    /// pipeline layout). Currently holds an identity mapping (`map[i] = i`)
    /// — animated tile rotation hasn't shipped — but it must exist because
    /// `shaders/chunk.frag` declares `layout(set = 1, binding = 0) uniform
    /// TileRemap { uint map[256]; } tile_remap;` and the validation layer
    /// blocks pipeline creation without it.
    tile_remap_ubo: GpuBuffer,
    tile_remap_descriptor_set: vk::DescriptorSet,
}

/// Per-chunk occlusion query tracking.
#[derive(Clone, Debug)]
struct OcclusionState {
    /// Index into the per-frame query pool.
    query_index: u32,
    /// Was this chunk visible in the last completed occlusion query?
    was_visible: bool,
    /// How many consecutive frames this chunk has been invisible.
    consecutive_invisible: u32,
}

/// Per-frame-in-flight occlusion query data.
struct OcclusionFrameData {
    /// Query pool for this frame.
    query_pool: vk::QueryPool,
    /// Indices of queries written this frame (for reset + readback).
    used_queries: Vec<u32>,
}

/// How many frames a chunk must be invisible before we switch to AABB proxy.
const OCCLUSION_INVISIBLE_THRESHOLD: u32 = 2;
/// Maximum occlusion queries per frame-in-flight.
const MAX_OCCLUSION_QUERIES: u32 = 16384;

/// Tiny 1×1×6 cubemap used as a placeholder when no
/// `assets/textures/panorama/*.png` files are present.
///
/// Layout mirrors `Panorama` but is single-pass: no staging, no upload,
/// just an image + view + sampler so the panorama descriptor set always
/// has a valid binding. Owned by [`Renderer`] (not constructed inline in
/// `Renderer::new`) so [`Renderer::drop`] can free the underlying
/// `gpu_allocator` allocation — pre-refactor these four resources lived in
/// `let` locals inside the `else` branch and silently leaked at shutdown
/// (the `panorama_placeholder` allocation accumulated each launch).
struct PanoramaPlaceholder {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
}

impl PanoramaPlaceholder {
    /// Free Vulkan handles + the underlying allocation. Call from the
    /// renderer [`Drop`] once `device_wait_idle` has confirmed the GPU is
    /// past any frame that might have sampled the placeholder.
    fn destroy(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        self.image = vk::Image::null();
        self.view = vk::ImageView::null();
        self.sampler = vk::Sampler::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }
}

#[allow(dead_code)]
pub struct Renderer {
    config: RendererConfig,
    _entry: Entry,
    instance: AshInstance,
    #[allow(dead_code)]
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    #[allow(dead_code)]
    queues: QueueFamilies,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    surface_instance: ash::khr::surface::Instance,
    swapchain_device: ash::khr::swapchain::Device,
    alloc: ManuallyDrop<Arc<Alloc>>,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    depth: Option<GpuImage>,
    // ── MSAA resolve targets ──
    msaa_color: Option<GpuImage>,
    msaa_depth: Option<GpuImage>,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    wireframe_pipeline: vk::Pipeline,
    transparent_pipeline: vk::Pipeline,
    wireframe_enabled: bool,
    #[allow(dead_code)]
    descriptor_pool: vk::DescriptorPool,
    #[allow(dead_code)]
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Set-1 descriptor set layout for the chunk material pipeline
    /// (`tile_remap` UBO consumed by `shaders/chunk.frag`). Held on the
    /// renderer for `recreate_chunk_pipelines` + tile_remap descriptor set
    /// allocation. See `pipeline::create_tile_remap_descriptor_set_layout`.
    #[allow(dead_code)]
    tile_remap_set_layout: vk::DescriptorSetLayout,

    command_pool: vk::CommandPool,
    // Cached shader SPIR-V blobs. Populated at startup from build.rs-compiled
    // defaults; the reload path calls `compile_shader` to refresh a single
    // file. Used so we can hot-reload pipelines without always shelling
    // out to glslangValidator for the unchanged partner shader.
    chunk_vert_spirv: Vec<u8>,
    chunk_frag_spirv: Vec<u8>,
    ui_vert_spirv: Vec<u8>,
    ui_frag_spirv: Vec<u8>,
    sky_vert_spirv: Vec<u8>,
    sky_frag_spirv: Vec<u8>,
    shadow_vert_spirv: Vec<u8>,
    shadow_frag_spirv: Vec<u8>,
    post_vert_spirv: Vec<u8>,
    post_frag_spirv: Vec<u8>,
    #[allow(dead_code)]
    entity_vert_spirv: Vec<u8>,
    entity_frag_spirv: Vec<u8>,
    atlas: AtlasTexture,
    #[allow(dead_code)]
    fog_ubo: GpuBuffer,
    /// Tile material lookup table UBO (chunk descriptor binding 5). Updated
    /// each frame by the engine via [`Renderer::set_tile_material_table`]
    /// from the world's registry + the engine's water-level / strength
    /// scalars. Total size = `MaterialTable::SIZE_BYTES` (4112 bytes).
    #[allow(dead_code)]
    tile_material_ubo: GpuBuffer,
    /// Per-frame scratch we copy the engine's `MaterialTable` into before
    /// uploading it to the host-visible UBO. Avoids aliasing engine-owned
    /// memory across the GPU write.
    pending_material_table: MaterialTable,
    /// Flushed once per frame in `flush_pending_ubos()`. Lets the engine
    /// decide WHEN to push (it owns the registry) without taking a borrow
    /// on the renderer at hot-path time.
    material_table_dirty: bool,

    // â”€â”€ UI pipeline â”€â”€
    ui_pipeline: vk::Pipeline,
    ui_pipeline_layout: vk::PipelineLayout,
    #[allow(dead_code)]
    ui_descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    ui_descriptor_pool: vk::DescriptorPool,
    ui_descriptor_set: vk::DescriptorSet,
    #[allow(dead_code)]
    font_texture: AtlasTexture,
    minimap_texture: crate::dynamic_texture::DynamicAtlasTexture,
    ui_vbo: GpuBuffer,
    ui_ibo: GpuBuffer,

    // â”€â”€ Sky pipeline â”€â”€
    sky_pipeline: vk::Pipeline,
    sky_pipeline_layout: vk::PipelineLayout,
    #[allow(dead_code)]
    sky_descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    sky_descriptor_pool: vk::DescriptorPool,
    sky_descriptor_set: vk::DescriptorSet,
    sky_ubo: GpuBuffer,

    // Panorama pipeline (title screen cubemap)
    panorama_pipeline: vk::Pipeline,
    panorama_pipeline_layout: vk::PipelineLayout,
    panorama_descriptor_set_layout: vk::DescriptorSetLayout,
    panorama_descriptor_pool: vk::DescriptorPool,
    panorama_descriptor_set: vk::DescriptorSet,
    panorama: crate::panorama::Panorama,
    /// `Some` when the user has no `assets/textures/panorama/*.png`
    /// files: holds the placeholder cubemap so Drop can release the
    /// underlying image + view + sampler + allocation. `None` when a
    /// real panorama was loaded this launch (everything lives on
    /// `panorama` instead).
    panorama_placeholder: Option<PanoramaPlaceholder>,
    panorama_vert_spirv: Vec<u8>,
    panorama_frag_spirv: Vec<u8>,

    // Entity pipeline
    entity_pipeline: vk::Pipeline,
    entity_pipeline_layout: vk::PipelineLayout,
    entity_vbo: GpuBuffer,
    entity_ibo: GpuBuffer,

    // Held item pipeline (ALWAYS depth compare, no write).
    entity_held_pipeline: vk::Pipeline,

    // ── Particle pipeline (subpass 1 of the main render pass) ──
    // Reads back the depth-written by chunks + entity + held item passes via
    // subpass 1's depth attachment (READ_ONLY_OPTIMAL) so chunks/held items
    // occlude particles naturally. Phase 1 uses the depth test; Phase 2 will
    // swap it for an input-attachment read to compute the soft fade.
    // Shares the chunk `pipeline_layout` (descriptor set is compatible) and
    // re-uses `entity_vbo` for the quad per-vertex data.
    particle_pipeline: vk::Pipeline,
    particle_pipeline_layout: vk::PipelineLayout,
    particle_depth_set_layout: vk::DescriptorSetLayout,
    particle_depth_descriptor_pool: vk::DescriptorPool,
    particle_depth_descriptor_sets: Vec<vk::DescriptorSet>,
    particle_vert_spirv: Vec<u8>,
    particle_frag_spirv: Vec<u8>,
    particle_instance_vbo: GpuBuffer,
    particle_manager: crate::particle::ParticleManager,
    particle_softness: f32,

    // Overlay pipeline (wireframe line rendering).
    overlay_pipeline: vk::Pipeline,
    overlay_pipeline_layout: vk::PipelineLayout,
    overlay_vbo: GpuBuffer,
    overlay_vert_spirv: Vec<u8>,
    overlay_frag_spirv: Vec<u8>,
    overlay_data: crate::overlay::OverlayData,

    // ── Occlusion culling (hardware occlusion queries) ──
    occlusion_pipeline: vk::Pipeline,
    aabb_index_buffer: GpuBuffer,
    aabb_vert_spirv: Vec<u8>,
    aabb_frag_spirv: Vec<u8>,
    /// Per-chunk visibility state. Written by readback (draw_frame, &mut self)
    /// and read during recording (record_chunk_passes, &self via interior mut).
    occlusion_state: RwLock<std::collections::HashMap<ChunkPos, OcclusionState>>,
    occlusion_frames: RwLock<Vec<OcclusionFrameData>>,
    /// Config toggle.
    occlusion_culling_enabled: bool,

    // Model registry: loaded glTF models indexed by model_id.
    models: Vec<crate::model::Model>,

    // â”€â”€ Shadow pass â”€â”€
    shadow_render_pass: vk::RenderPass,
    shadow_pipeline: vk::Pipeline,
    shadow_pipeline_layout: vk::PipelineLayout,
    shadow_image: GpuImage,
    shadow_layer_views: Vec<vk::ImageView>,
    shadow_sampler: vk::Sampler,
    shadow_framebuffers: Vec<vk::Framebuffer>,
    shadow_ubo_data: ShadowUbo,

    // â”€â”€ Offscreen color (for post-processing) â”€â”€
    offscreen_images: Vec<GpuImage>,
    offscreen_framebuffers: Vec<vk::Framebuffer>,

    // â”€â”€ Post pass â”€â”€
    post_render_pass: vk::RenderPass,
    post_pipeline: vk::Pipeline,
    post_pipeline_layout: vk::PipelineLayout,
    post_descriptor_set_layout: vk::DescriptorSetLayout,
    post_descriptor_pool: vk::DescriptorPool,
    post_descriptor_sets: Vec<vk::DescriptorSet>,
    post_sampler: vk::Sampler,
    /// NEAREST sampler for depth buffer (SSAO).
    depth_sampler: vk::Sampler,
    post_framebuffers: Vec<vk::Framebuffer>,
    post_params: [f32; 4],
    /// SSAO push constants: [radius, bias, strength, enabled].
    ssao_params: [f32; 4],
    /// Projection params for SSAO depth linearization: [near, far, screen_w, screen_h].
    proj_params: [f32; 4],

    // --- Slice 2: scene-opaque-color copy + transparent render pass ---
    /// Slice 2's `scene_opaque_color` image: a single-sample colour image
    /// (TRANSFER_DST + SAMPLED) sized to the offscreen colour extent. After
    /// the main render pass ends we `vkCmdCopyImage` from the offscreen
    /// resolve target into THIS image and transition to
    /// `SHADER_READ_ONLY_OPTIMAL`. The transparent chunk pipeline then
    /// samples it via chunk descriptor binding 6 for `TRANSLUCENT_ABSORB`
    /// (glass tinted absorption) and refracted-depth lookup (water).
    ///
    /// MVP simplification: a single shared image across frames. Per-frame
    /// ring-buffer swap will be added once we validate the COPY is sound.
    scene_opaque_color: GpuImage,
    /// Linear clamp-to-edge sampler shared by every frame's descriptor set
    /// for binding 6 (`scene_opaque_color`). `LINEAR` so refraction can
    /// sample neighbouring pixels without hard texel-edges producing moire.
    scene_opaque_sampler: vk::Sampler,
    /// NEAREST clamp-to-edge sampler for binding 7 (`scene_opaque_depth`).
    /// Exact-texel depth reads for the SSR ray-march (see slice-3 fields).
    scene_depth_sampler: vk::Sampler,
    /// Second render pass for slice 2. Created with `LOAD_OP` so it picks up
    /// the colour/depth images left behind by the main pass, draws the
    /// transparent chunk meshes (which sample `scene_opaque_color` via
    /// binding 6), and writes the result back. Sequenced between the main
    /// pass and the post pass in `draw_frame`.
    transparent_render_pass: vk::RenderPass,
    /// One framebuffer per swapchain image. Targets the same `offscreen_images`
    /// attachment pair the main pass uses so transparent draws composite on
    /// top of the opaque scene.
    transparent_framebuffers: Vec<vk::Framebuffer>,
    // --- Slice 3: reflections (scene_opaque_depth + reflection UBO) ---
    /// Single-sample depth companion to `scene_opaque_color`: after the main
    /// render pass the resolved scene depth is copied into this image
    /// (`record_scene_opaque_copy`) and the transparent chunk shader samples
    /// it via binding 7 for the SSR ray-march + water-column absorption.
    /// With MSAA, the main render pass resolves its multisampled depth into
    /// `depth` (Vulkan 1.2 `VkSubpassDescriptionDepthStencilResolve`) first.
    scene_opaque_depth: GpuImage,
    /// Reflection/environment UBO (chunk descriptor binding 8): sky colours,
    /// sun direction, near/far, underwater + SSR-valid flags, and the master
    /// reflection strength. Updated once per frame in `flush_pending_ubos`.
    reflection_ubo: GpuBuffer,
    /// Master reflection strength in [0, 1], pushed by the engine each frame
    /// via [`Renderer::set_reflection_strength`]. 0 disables reflections.
    reflection_strength: f32,
    /// True when a valid single-sample scene depth exists for the SSR
    /// ray-march: always when MSAA is off; with MSAA only when the device
    /// supports depth-stencil resolve (Vulkan 1.2+). When false the shader
    /// skips the ray-march and falls back to the analytic sky reflection.
    ssr_depth_valid: bool,
    /// True when the MAIN render pass was created with the 4th single-sample
    /// depth-resolve attachment (MSAA on + depth-stencil-resolve supported).
    /// Framebuffer creation (constructor + `recreate_swapchain`) must attach
    /// the matching `depth.view` then.
    depth_resolve_active: bool,

    frames: Vec<Frame>,
    chunks: RwLock<HashMap<ChunkPos, ChunkBuffers>>,
    /// Phase-1 GPU-driven chunk pipeline. `Some` when `config.gpu_driven`;
    /// `None` otherwise (legacy per-chunk path). When present, `upload_chunks`,
    /// `remove_chunk`, `chunk_count`, and `record_chunk_passes` route here.
    gpu_driven: Option<indirect::GpuDriven>,
    /// Per-frame deletion queue — buffers that `upload_chunks` could not
    /// destroy immediately because the GPU was still mid-draw with them.
    /// Drained at the start of `draw_frame` once they are at least
    /// `FRAMES_IN_FLIGHT` frames old.
    pending_destruction: Vec<(usize, PendingDestroy)>,

    // â”€â”€ GPU timers â”€â”€
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    timings: GpuTimings,

    /// Set when the window was resized; swapchain is recreated next draw.
    needs_resize: bool,
    frame_counter: usize,
    /// Dynamic sky params set by the engine each frame for day/night.
    sky_horizon: [f32; 3],
    sky_zenith: [f32; 3],
    sky_fog: [f32; 3],
    sky_ambient: f32,
    sky_underwater: bool,
    sun_dir: [f32; 3],
}

/// Load texture pack tile mappings from `texture_packs_dir` (if configured)
/// and merge them with the base `textures_dir` mapping. Returns `None`
/// when there is no packs dir or no packs were found.
fn load_pack_mapping(
    textures_dir: &Path,
    texture_packs_dir: Option<&Path>,
) -> Option<std::collections::HashMap<u32, String>> {
    let packs_dir = match texture_packs_dir {
        Some(d) if d.is_dir() => d,
        Some(d) => {
            log::warn!(
                "texture_packs_dir configured but not found: {}",
                d.display()
            );
            return None;
        }
        None => return None,
    };
    match voxel_asset_pipeline::texture_pack::load_all_texture_packs(packs_dir, textures_dir) {
        Ok((merged, packs)) => {
            if !packs.is_empty() {
                log::info!(
                    "loaded {} texture pack(s) with {} overridden tile(s)",
                    packs.len(),
                    merged.len()
                );
            }
            if merged.is_empty() { None } else { Some(merged) }
        }
        Err(e) => {
            log::warn!("failed to load texture packs: {e}");
            None
        }
    }
}

impl Renderer {
    /// Create a complete renderer for `window`.
    pub fn new(
        window_handle: RawWindowHandle,
        display_handle: RawDisplayHandle,
        config: RendererConfig,
    ) -> Result<Self> {
        let entry = unsafe { Entry::load() }.map_err(|e| anyhow!("Vulkan loader: {e}"))?;

        // --- instance ---
        let instance = device::create_instance(&entry, display_handle, config.validation)?;

        let debug_messenger = if config.validation {
            device::create_debug_messenger(&entry, &instance).ok()
        } else {
            None
        };

        // --- surface ---
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .map_err(|e| anyhow!("create_surface: {e:?}"))?;
        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        // --- physical device + queues ---
        let (physical_device, queues) =
            device::pick_physical_device(&instance, &surface_instance, surface)?;

        // --- logical device ---
        let (device, graphics_queue, present_queue) = device::create_logical_device(
            &instance,
            physical_device,
            queues,
            &surface_instance,
            surface,
        )?;
        let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);

        let alloc = ManuallyDrop::new(Arc::new(Alloc::new(&instance, physical_device, &device)?));

        // --- command pool ---
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queues.graphics);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .context("create_command_pool")?;

        // --- atlas texture ---
        // The textures directory must exist and contain a textures.toml
        // config. If it doesn't, all tiles show the blue+black error pattern.
        let atlas_pixels = match config.textures_dir.as_deref() {
            Some(dir) if dir.is_dir() => build_atlas_with_textures(dir),
            _ => {
                log::warn!(
                    "no textures_dir configured or directory not found â€” all tiles will show error pattern"
                );
                build_atlas_with_textures(Path::new(""))
            }
        };
        // Cache shader SPIR-V at startup. `build.rs` has already compiled
        // them; we just borrow the bytes into owned Vec<u8> so the reload
        // path can update a single file via glslangValidator.
        let spirv = init::load_shader_blobs();
        let chunk_vert_spirv = spirv.chunk_vert;
        let chunk_frag_spirv = spirv.chunk_frag;
        let ui_vert_spirv = spirv.ui_vert;
        let ui_frag_spirv = spirv.ui_frag;
        let sky_vert_spirv = spirv.sky_vert;
        let sky_frag_spirv = spirv.sky_frag;
        let shadow_vert_spirv = spirv.shadow_vert;
        let shadow_frag_spirv = spirv.shadow_frag;
        let post_vert_spirv = spirv.post_vert;
        let post_frag_spirv = spirv.post_frag;
        let entity_vert_spirv = spirv.entity_vert;
        let entity_frag_spirv = spirv.entity_frag;
        let particle_vert_spirv = spirv.particle_vert;
        let particle_frag_spirv = spirv.particle_frag;
        let overlay_vert_spirv = spirv.overlay_vert;
        let overlay_frag_spirv = spirv.overlay_frag;
        let aabb_vert_spirv = spirv.aabb_vert;
        let aabb_frag_spirv = spirv.aabb_frag;
        let atlas =
            AtlasTexture::new(&device, &alloc, command_pool, graphics_queue, &atlas_pixels)?;

        // --- descriptor set layout + pool + fog UBO ---
        let descriptor_set_layout = pipeline::create_descriptor_set_layout(&device)?;
        let tile_remap_set_layout =
            pipeline::create_tile_remap_descriptor_set_layout(&device)?;
        // `max_sets = FRAMES_IN_FLIGHT * 2`: each frame in flight now has
        // TWO descriptor sets — the chunk material set (binding 0..8) and
        // the new tile_remap set (set 1 binding 0). Both are allocated from
        // the same pool.
        let descriptor_pool =
            pipeline::create_descriptor_pool(&device, FRAMES_IN_FLIGHT * 2)?;
        let tile_remap_descriptor_sets = pipeline::allocate_descriptor_sets(
            &device,
            descriptor_pool,
            tile_remap_set_layout,
            FRAMES_IN_FLIGHT,
        )?;
        let mut fog_ubo = GpuBuffer::host_visible(
            &device,
            &alloc,
            std::mem::size_of::<FogUbo>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "fog_ubo",
        )?;
        {
            let fog = FogUbo {
                color_and_density: [
                    config.fog_color[0],
                    config.fog_color[1],
                    config.fog_color[2],
                    1.0,
                ],
                ambient_and_sun: [1.0, 0.0, 1.0, 0.0], // full daylight, sun straight up
            };
            let slice = fog_ubo.mapped_slice_mut()?;
            let bytes: &[u8] = bytemuck::bytes_of(&fog);
            slice[..bytes.len()].copy_from_slice(bytes);
        }
        // Flush the init write so the first frame sees valid fog data.
        if let Err(e) = fog_ubo.flush_whole(&device) {
            log::warn!("fog_ubo init flush failed: {e}");
        }            // Tile material lookup table: the descriptor binding 5 the chunk
            // shader uses for leaves SSS, wet-edge tint, sun caustics, and (later)
            // glass tinted absorption. Host-visible; the engine pushes a fresh
            // table each frame via `set_tile_material_table`.
            let tile_material_ubo = GpuBuffer::host_visible(
                &device,
                &alloc,
                MaterialTable::SIZE_BYTES as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "tile_material_ubo",
            )?;
            // Deliberately NOT seeding the UBO here. The first frame's flush
            // will write whatever the engine pushed (defaulting to
            // MaterialTable::empty() if nothing was pushed yet). This avoids
            // the "init seeds zeros + first flush overwrites" double-upload
            // on frame 0.

        // --- render pass ---
        // We need the swapchain before framebuffers, and the render pass before
        // the pipeline. Build swapchain first.
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) = swapchain::create_swapchain(
            &device,
            &swapchain_device,
            &surface_instance,
            physical_device,
            surface,
            config.vsync,
        )?;
        let swapchain_image_views =
            swapchain::create_image_views(&device, &swapchain_images, swapchain_format)?;

        let depth_format = device::find_depth_format(&instance, physical_device);

        // Resolve the requested MSAA sample count against what the device supports.
        let msaa_samples = init::resolve_msaa_samples(&instance, physical_device, config.msaa_samples);
        if msaa_samples != vk::SampleCountFlags::TYPE_1 {
            log::info!("MSAA {:?} enabled", msaa_samples);
        }

        // ── Depth-stencil-resolve support (for the SSR scene-depth copy) ──
        let depth_resolve_mode = init::probe_depth_resolve(&instance, physical_device);
        // The SSR scene-depth copy is valid without MSAA (the single-sample
        // depth attachment is copied directly) and with MSAA only when depth
        // resolve works. Recorded once here; surfaced to the shader via the
        // reflection UBO's `proj_misc.w` flag.
        let ssr_depth_valid =
            msaa_samples == vk::SampleCountFlags::TYPE_1 || depth_resolve_mode.is_some();

        let render_pass = pipeline::create_render_pass(
            &device,
            swapchain_format,
            depth_format,
            msaa_samples,
            depth_resolve_mode,
        )?;
        // The slice-2 transparent render pass is created BEFORE the chunk
        // pipelines so `transparent_pipeline` can be created directly against
        // it (instead of relying on render-pass compatibility with the main
        // pass, which silently broke once the main pass grew a 4th
        // attachment).
        let transparent_render_pass = pipeline::create_transparent_render_pass(
            &device,
            swapchain_format,
            depth_format,
            vk::SampleCountFlags::TYPE_1,
        )?;
        let pipeline_layout = pipeline::create_pipeline_layout(
            &device,
            descriptor_set_layout,
            tile_remap_set_layout,
        )?;
        let pipeline = pipeline::create_graphics_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            vk::PolygonMode::FILL,
            vk::CullModeFlags::BACK,
            &chunk_vert_spirv,
            &chunk_frag_spirv,
            msaa_samples,
            true,
        )?;
        let wireframe_pipeline = pipeline::create_graphics_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            vk::PolygonMode::LINE,
            vk::CullModeFlags::BACK,
            &chunk_vert_spirv,
            &chunk_frag_spirv,
            msaa_samples,
            true,
        )?;
        // The transparent pipeline is created against the TRANSPARENT render
        // pass (not the main pass): it is only ever bound inside
        // `transparent_render_pass` (slice 2), and after the main pass grew a
        // 4th (depth-resolve) attachment the two passes are no longer
        // trivially compatible.
        // Depth write is DISABLED for transparent geometry so water/glass
        // don't occlude each other or corrupt the depth buffer for SSAO.
        // MSAA is forced to 1x for the transparent pass: the water shader's
        // SSR ray-march is extremely expensive at 4x (24 texture fetches per
        // step × 4 samples), and transparent geometry doesn't benefit much
        // from MSAA anyway.  This eliminates the GPU timeout (device lost)
        // that occurs when large water surfaces are in view.
        let transparent_pipeline = pipeline::create_graphics_pipeline(
            &device,
            transparent_render_pass,
            pipeline_layout,
            vk::PolygonMode::FILL,
            vk::CullModeFlags::NONE,
            &chunk_vert_spirv,
            &chunk_frag_spirv,
            vk::SampleCountFlags::TYPE_1,
            false,
        )?;

        // ── Particle descriptor set layout + pipeline layout (Phase 2) ──
        // Created BEFORE the depth GpuImage so the depth-input descriptor
        // sets (allocated just below) can reference the set layout, and so
        // `create_particle_pipeline` later in this function can reference the
        // pipeline layout.
        let particle_depth_set_layout =
            pipeline::create_particle_descriptor_set_layout(&device)?;
        let particle_pipeline_layout = pipeline::create_particle_pipeline_layout(
            &device,
            descriptor_set_layout,
            particle_depth_set_layout,
        )?;
        let depth = GpuImage::depth(&device, &alloc, swapchain_extent, depth_format)?;

        // ── MSAA resolve images (multisampled color + depth) ──
        let (msaa_color_img, msaa_depth_img) = if msaa_samples != vk::SampleCountFlags::TYPE_1 {
            let color = GpuImage::color_attachment_msaa(
                &device,
                &alloc,
                swapchain_extent,
                swapchain_format,
                msaa_samples,
                "msaa_color",
            )?;
            let depth_msaa = GpuImage::depth_msaa(
                &device,
                &alloc,
                swapchain_extent,
                depth_format,
                msaa_samples,
            )?;
            (Some(color), Some(depth_msaa))
        } else {
            (None, None)
        };

        // Particle depth-input descriptor pool: FRAMES_IN_FLIGHT sets, each
        // with one INPUT_ATTACHMENT at binding 0 pointing at `depth.view`.
        let particle_depth_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::INPUT_ATTACHMENT,
            descriptor_count: FRAMES_IN_FLIGHT as u32,
        }];
        let particle_depth_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(FRAMES_IN_FLIGHT as u32)
            .pool_sizes(&particle_depth_pool_sizes);
        let particle_depth_descriptor_pool = unsafe {
            device.create_descriptor_pool(&particle_depth_pool_info, None)
        }
        .map_err(|e| anyhow!("create_particle_depth_descriptor_pool: {e:?}"))?;
        let particle_depth_set_layouts_ref = [particle_depth_set_layout];
        let particle_depth_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(particle_depth_descriptor_pool)
            .set_layouts(&particle_depth_set_layouts_ref);
        let particle_depth_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&particle_depth_alloc_info)
        }
        .map_err(|e| anyhow!("allocate_particle_depth_descriptor_sets: {e:?}"))?;
        // Use MSAA depth view for particle input when MSAA is active.
        let particle_depth_view = if msaa_samples != vk::SampleCountFlags::TYPE_1 {
            msaa_depth_img.as_ref().unwrap().view
        } else {
            depth.view
        };
        for &set in &particle_depth_descriptor_sets {
            pipeline::update_particle_descriptor_set(&device, set, particle_depth_view);
        }

        // â”€â”€ Offscreen color images (post-processing source) â”€â”€
        // The main render pass writes to these instead of the swapchain images.
        let mut offscreen_images = Vec::with_capacity(swapchain_images.len());
        for i in 0..swapchain_images.len() {
            let img = GpuImage::color_attachment(
                &device,
                &alloc,
                swapchain_extent,
                swapchain_format,
                "offscreen",
            )?;
            let cmd_init = begin_one_time(&device, command_pool)?;
            transition_image_layout(
                &device,
                cmd_init,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            end_and_submit(&device, command_pool, graphics_queue, cmd_init)?;
            offscreen_images.push(img);
            let _ = i;
        }
        let offscreen_framebuffers = if msaa_samples != vk::SampleCountFlags::TYPE_1 {
            // With MSAA: 3 attachments – msaa_color (MSAA), msaa_depth (MSAA),
            // offscreen (resolve) — plus a 4th single-sample depth-resolve
            // attachment when depth-stencil-resolve is active.
            let msaa_c = msaa_color_img.as_ref().unwrap();
            let msaa_d = msaa_depth_img.as_ref().unwrap();
            let depth_resolve_active = depth_resolve_mode.is_some();
            offscreen_images
                .iter()
                .map(|img| {
                    if depth_resolve_active {
                        create_framebuffer_with(
                            &device,
                            render_pass,
                            &[msaa_c.view, msaa_d.view, img.view, depth.view],
                            swapchain_extent,
                        )
                    } else {
                        create_framebuffer_with(
                            &device,
                            render_pass,
                            &[msaa_c.view, msaa_d.view, img.view],
                            swapchain_extent,
                        )
                    }
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Without MSAA: 2 attachments – offscreen color + depth
            offscreen_images
                .iter()
                .map(|img| {
                    create_framebuffer_with(
                        &device,
                        render_pass,
                        &[img.view, depth.view],
                        swapchain_extent,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };

        // â”€â”€ Shadow pass init â”€â”€
        let shadow_extent = vk::Extent2D {
            width: 2048,
            height: 2048,
        };
        let shadow_render_pass = pipeline::create_shadow_render_pass(&device, depth_format)?;
        let shadow_pipeline_layout = pipeline::create_shadow_pipeline_layout(&device)?;
        let shadow_pipeline =
            pipeline::create_shadow_pipeline(&device, shadow_render_pass, shadow_pipeline_layout, &shadow_vert_spirv, &shadow_frag_spirv)?;
        let shadow_image = GpuImage::depth_array(
            &device,
            &alloc,
            shadow_extent,
            depth_format,
            4,
            "shadow_map",
        )?;
        let shadow_layer_views: Vec<vk::ImageView> = (0..4u32)
            .map(|i| pipeline::create_shadow_layer_view(&device, shadow_image.image, depth_format, i))
            .collect::<Result<Vec<_>>>()?;
        let shadow_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_shadow_sampler: {e:?}"))?;
        let shadow_framebuffers: Vec<vk::Framebuffer> = (0..4u32)
            .map(|i| {
                create_framebuffer_with(
                    &device,
                    shadow_render_pass,
                    &[shadow_layer_views[i as usize]],
                    shadow_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        {
            let cmd_init = begin_one_time(&device, command_pool)?;
            transition_image_layout(
                &device,
                cmd_init,
                shadow_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                1,
                4,
            );
            end_and_submit(&device, command_pool, graphics_queue, cmd_init)?;
        }

        // --- per-frame resources ---
        // Allocate FRAMES_IN_FLIGHT chunk material descriptor sets (one per
        // frame-in-flight, each set covers bindings 0..8 of the chunk
        // material pipeline layout). The companion tile_remap sets live
        // separately in `tile_remap_descriptor_sets`.
        let descriptor_sets = pipeline::allocate_descriptor_sets(
            &device,
            descriptor_pool,
            descriptor_set_layout,
            FRAMES_IN_FLIGHT,
        )?;
        // ----- Slice 2: scene_opaque_color image + linear sampler + transparent render pass -----
        // Allocates a single full-resolution color image (TRANSFER_DST +
        // SAMPLED, NO_COLOR_ATTACHMENT) that the transparent chunk pipeline
        // samples as binding 6. We create it BEFORE the descriptor set
        // updates below so the (view, sampler) pair is available at bind time.
        // MVP simplification: a single shared image across frames; per-frame
        // ring-buffer swap is a follow-up.
        let scene_opaque_color = GpuImage::scene_opaque(
            &device,
            &alloc,
            swapchain_extent,
            swapchain_format,
            "scene_opaque_color",
        )?;
        let scene_opaque_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_scene_opaque_sampler: {e:?}"))?;
        // Transition the new image from UNDEFINED to SHADER_READ_ONLY_OPTIMAL
        // so the very first descriptor binding of binding 6 is valid (no
        // undefined layout). The COPY each frame will reset the layout
        // back to TRANSFER_DST, do its work, and barrier it to
        // SHADER_READ_ONLY_OPTIMAL before the transparent pass starts.
        {
            let cmd_init = begin_one_time(&device, command_pool)?;
            transition_image_layout(
                &device,
                cmd_init,
                scene_opaque_color.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            end_and_submit(&device, command_pool, graphics_queue, cmd_init)?;
        }
        // ----- Slice 3: scene_opaque_depth image + reflection UBO -----
        // The depth companion to scene_opaque_color (binding 7). Same initial
        // transition dance: start in SHADER_READ_ONLY_OPTIMAL so the first
        // descriptor binding is valid; the per-frame copy re-barriers through
        // TRANSFER_DST.
        let scene_opaque_depth = GpuImage::scene_opaque_depth(
            &device,
            &alloc,
            swapchain_extent,
            depth_format,
            "scene_opaque_depth",
        )?;
        {
            let cmd_init = begin_one_time(&device, command_pool)?;
            transition_image_layout(
                &device,
                cmd_init,
                scene_opaque_depth.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                1,
                1,
            );
            end_and_submit(&device, command_pool, graphics_queue, cmd_init)?;
        }
        // NEAREST clamp sampler for the scene-depth copy (binding 7). NEAREST
        // because ray-marching wants the exact texel depth (linear filtering
        // would smear depth across silhouette edges → false SSR hits), and
        // D32 depth formats are not guaranteed to be linear-filterable.
        let scene_depth_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_scene_depth_sampler: {e:?}"))?;
        // Reflection/environment UBO (binding 8). Single shared buffer
        // updated once per frame after the frame-fence wait, same pattern as
        // the tile material UBO.
        let reflection_ubo = GpuBuffer::host_visible(
            &device,
            &alloc,
            std::mem::size_of::<pipeline::ReflectionUbo>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "reflection_ubo",
        )?;

        // Second render pass: LOAD_OP color + depth, samples scene_opaque via
        // binding 6. One framebuffer per offscreen image (uniform extent).
        // NOTE: `transparent_render_pass` itself was created earlier, right
        // after the main render pass, so `transparent_pipeline` could be
        // built against it.
        // The transparent pass is always single-sample (MSAA=1) to avoid
        // the GPU timeout from running the expensive water SSR shader at
        // 4x resolution. Framebuffers always use the non-MSAA path.
        let transparent_framebuffers = offscreen_images
            .iter()
            .map(|img| {
                create_framebuffer_with(
                    &device,
                    transparent_render_pass,
                    &[img.view, depth.view],
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for (idx_within_fif, &descriptor_set) in
            descriptor_sets.iter().take(FRAMES_IN_FLIGHT).enumerate()
        {
            let cmd = {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                unsafe { device.allocate_command_buffers(&alloc_info) }?[0]
            };
            let in_flight_fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }?;
            let image_available =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            let render_finished =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            let camera_ubo = GpuBuffer::host_visible(
                &device,
                &alloc,
                std::mem::size_of::<CameraUbo>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "camera_ubo",
            )?;
            let shadow_ubo = GpuBuffer::host_visible(
                &device,
                &alloc,
                std::mem::size_of::<ShadowUbo>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "shadow_ubo",
            )?;
            // Bind this frame's camera UBO + shared atlas + shared fog + shadow
            // map + shadow UBO + material table + slot-6/7 scene_opaque
            // color+depth samplers + slot-8 reflection UBO.
            pipeline::update_descriptor_set(
                &device,
                descriptor_set,
                camera_ubo.buffer,
                fog_ubo.buffer,
                atlas.view,
                atlas.sampler,
                shadow_image.view,
                shadow_sampler,
                shadow_ubo.buffer,
                tile_material_ubo.buffer,
                MaterialTable::SIZE_BYTES as vk::DeviceSize,
                scene_opaque_color.view,
                scene_opaque_sampler,
                scene_opaque_depth.view,
                scene_depth_sampler,
                reflection_ubo.buffer,
            );
            // Tile-remap UBO: 256 u32 entries (= 1024 B). Currently an
            // identity map so the chunk shader's `tile_remap.map[frag_tile]`
            // returns the canonical tile index; future animated-tile logic
            // updates this in place.
            let mut tile_remap_ubo = GpuBuffer::host_visible(
                &device,
                &alloc,
                std::mem::size_of::<pipeline::TileRemapUbo>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "tile_remap_ubo",
            )?;
            {
                let mut map = [0u32; 256];
                for (i, slot) in map.iter_mut().enumerate() {
                    *slot = i as u32;
                }
                let remap = pipeline::TileRemapUbo { map };
                tile_remap_ubo.upload(&device, bytemuck::bytes_of(&remap))?;
            }
            pipeline::update_tile_remap_descriptor_set(
                &device,
                tile_remap_descriptor_sets[idx_within_fif],
                tile_remap_ubo.buffer,
            );
            frames.push(Frame {
                cmd,
                in_flight_fence,
                image_available,
                render_finished,
                camera_ubo,
                shadow_ubo,
                descriptor_set,
                tile_remap_ubo,
                tile_remap_descriptor_set: tile_remap_descriptor_sets[idx_within_fif],
            });
        }

        // â”€â”€ UI pipeline â”€â”€
        let font = crate::ui::FontAtlas::new();
        let font_texture =
            AtlasTexture::new(&device, &alloc, command_pool, graphics_queue, &font.atlas)?;
        let ui_descriptor_set_layout = pipeline::create_ui_descriptor_set_layout(&device)?;
        // Separate pool for the UI descriptor set (3 image samplers: block + font + minimap).
        let ui_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 3,
        }];
        let ui_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&ui_pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let ui_descriptor_pool = unsafe { device.create_descriptor_pool(&ui_pool_info, None) }
            .map_err(|e| anyhow!("create_ui_descriptor_pool: {e:?}"))?;
        let ui_descriptor_set =
            pipeline::allocate_ui_descriptor_set(&device, ui_descriptor_pool, ui_descriptor_set_layout)?;

        // Create a placeholder minimap texture (256×256, will be uploaded later).
        let minimap_texture = crate::dynamic_texture::DynamicAtlasTexture::new(
            &device, &alloc, command_pool, graphics_queue, 256, 256,
        )?;

        pipeline::update_ui_descriptor_set(
            &device,
            ui_descriptor_set,
            atlas.view,
            atlas.sampler,
            font_texture.view,
            font_texture.sampler,
            minimap_texture.view,
            minimap_texture.sampler,
        );
        let ui_pipeline_layout = pipeline::create_ui_pipeline_layout(&device, ui_descriptor_set_layout)?;
        let ui_pipeline = pipeline::create_ui_pipeline(&device, render_pass, ui_pipeline_layout, &ui_vert_spirv, &ui_frag_spirv, msaa_samples)?;

        // Persistent host-visible buffers for UI vertices/indices (re-uploaded
        // each frame). 256 KB each is way more than a simple HUD needs.
        let ui_vbo = GpuBuffer::host_visible(
            &device,
            &alloc,
            256 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "ui_vbo",
        )?;
        let ui_ibo = GpuBuffer::host_visible(
            &device,
            &alloc,
            256 * 1024,
            vk::BufferUsageFlags::INDEX_BUFFER,
            "ui_ibo",
        )?;

        // â”€â”€ Sky pipeline â”€â”€
        let sky_descriptor_set_layout = pipeline::create_sky_descriptor_set_layout(&device)?;
        let sky_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];
        let sky_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sky_pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let sky_descriptor_pool = unsafe { device.create_descriptor_pool(&sky_pool_info, None) }
            .map_err(|e| anyhow!("create_sky_descriptor_pool: {e:?}"))?;
        let sky_layouts = [sky_descriptor_set_layout];
        let sky_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(sky_descriptor_pool)
            .set_layouts(&sky_layouts);
        let sky_sets = unsafe { device.allocate_descriptor_sets(&sky_alloc_info) }
            .map_err(|e| anyhow!("allocate_sky_descriptor_set: {e:?}"))?;
        let sky_descriptor_set = sky_sets[0];
        let sky_ubo = GpuBuffer::host_visible(
            &device,
            &alloc,
            std::mem::size_of::<SkyUbo>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "sky_ubo",
        )?;
        // Bind sky UBO to the descriptor set.
        let sky_buf_info = vk::DescriptorBufferInfo::default()
            .buffer(sky_ubo.buffer)
            .offset(0)
            .range(std::mem::size_of::<SkyUbo>() as u64);
        let sky_buf_infos = [sky_buf_info];
        let sky_writes = [vk::WriteDescriptorSet::default()
            .dst_set(sky_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&sky_buf_infos)];
        unsafe { device.update_descriptor_sets(&sky_writes, &[]) };
        let sky_pipeline_layout = pipeline::create_sky_pipeline_layout(&device, sky_descriptor_set_layout)?;
        let sky_pipeline = pipeline::create_sky_pipeline(&device, render_pass, sky_pipeline_layout, &sky_vert_spirv, &sky_frag_spirv, msaa_samples)?;

        // Panorama pipeline (title screen cubemap)
        let panorama_vert_spirv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/panorama.vert.spv")).to_vec();
        let panorama_frag_spirv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/panorama.frag.spv")).to_vec();
        let panorama_descriptor_set_layout = pipeline::create_panorama_descriptor_set_layout(&device)?;
        let panorama_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        let panorama_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&panorama_pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let panorama_descriptor_pool = unsafe { device.create_descriptor_pool(&panorama_pool_info, None) }
            .map_err(|e| anyhow!("create_panorama_descriptor_pool: {e:?}"))?;
        let panorama_layouts = [panorama_descriptor_set_layout];
        let panorama_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(panorama_descriptor_pool)
            .set_layouts(&panorama_layouts);
        let panorama_sets = unsafe { device.allocate_descriptor_sets(&panorama_alloc_info) }
            .map_err(|e| anyhow!("allocate_panorama_descriptor_set: {e:?}"))?;
        let panorama_descriptor_set = panorama_sets[0];

        // Try to load panorama cubemap from assets/textures/panorama/.
        let panorama_dir = std::path::Path::new("assets/textures/panorama");
        let panorama = crate::panorama::Panorama::load(&device, &alloc, command_pool, graphics_queue, panorama_dir);

        // Create a 1x1 placeholder cubemap if panorama didn't load (so the
        // descriptor set always has a valid image view bound). The
        // placeholder is owned by the renderer (not a `let` local) so
        // [`Renderer::drop`] can release the four underlying Vulkan
        // resources + `gpu_allocator` allocation; pre-refactor these
        // silently leaked at shutdown.
        let panorama_view_for_binding;
        let panorama_sampler_for_binding;
        let panorama_placeholder: Option<PanoramaPlaceholder>;
        if panorama.loaded {
            panorama_view_for_binding = panorama.view;
            panorama_sampler_for_binding = panorama.sampler;
            panorama_placeholder = None;
        } else {
            // Create a tiny 1x1x6 cubemap as placeholder.
            let placeholder_create = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
                .mip_levels(1)
                .array_layers(6)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);
            let placeholder_img = unsafe { device.create_image(&placeholder_create, None) }
                .map_err(|e| anyhow!("panorama placeholder image: {e:?}"))?;
            let req = unsafe { device.get_image_memory_requirements(placeholder_img) };
            let placeholder_alloc = alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "panorama_placeholder",
                requirements: req,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe {
                device.bind_image_memory(placeholder_img, placeholder_alloc.memory(), placeholder_alloc.offset())
                    .map_err(|e| anyhow!("panorama placeholder bind: {e:?}"))?;
            }
            // Transition to SHADER_READ.
            let cmd_ph = begin_one_time(&device, command_pool)?;
            transition_image_layout(&device, cmd_ph, placeholder_img,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR, 1, 6);
            end_and_submit(&device, command_pool, graphics_queue, cmd_ph)?;

            let view_info = vk::ImageViewCreateInfo::default()
                .image(placeholder_img)
                .view_type(vk::ImageViewType::CUBE)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 6,
                });
            let placeholder_view = unsafe { device.create_image_view(&view_info, None) }
                .map_err(|e| anyhow!("panorama placeholder view: {e:?}"))?;
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            let placeholder_sampler = unsafe { device.create_sampler(&sampler_info, None) }
                .map_err(|e| anyhow!("panorama placeholder sampler: {e:?}"))?;
            panorama_view_for_binding = placeholder_view;
            panorama_sampler_for_binding = placeholder_sampler;
            panorama_placeholder = Some(PanoramaPlaceholder {
                image: placeholder_img,
                view: placeholder_view,
                sampler: placeholder_sampler,
                allocation: Some(placeholder_alloc),
            });
        }

        // Write the panorama descriptor set.
        let panorama_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(panorama_view_for_binding)
            .sampler(panorama_sampler_for_binding);
        let panorama_img_infos = [panorama_img_info];
        let panorama_writes = [vk::WriteDescriptorSet::default()
            .dst_set(panorama_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&panorama_img_infos)];
        unsafe { device.update_descriptor_sets(&panorama_writes, &[]) };

        let panorama_pipeline_layout = pipeline::create_panorama_pipeline_layout(&device, panorama_descriptor_set_layout)?;
        let panorama_pipeline = pipeline::create_panorama_pipeline(&device, render_pass, panorama_pipeline_layout, &panorama_vert_spirv, &panorama_frag_spirv, msaa_samples)?;

        // Entity pipeline (shares chunk descriptor set layout).
        let entity_pipeline_layout = crate::entity::create_entity_pipeline_layout(&device, descriptor_set_layout)?;
        let entity_pipeline = crate::entity::create_entity_pipeline(&device, render_pass, entity_pipeline_layout, &entity_vert_spirv, &entity_frag_spirv, msaa_samples)?;
        let entity_vbo = GpuBuffer::host_visible(&device, &alloc, 64 * 1024, vk::BufferUsageFlags::VERTEX_BUFFER, "entity_vbo")?;
        let entity_ibo = GpuBuffer::host_visible(&device, &alloc, 64 * 1024, vk::BufferUsageFlags::INDEX_BUFFER, "entity_ibo")?;
        let entity_held_pipeline = crate::entity::create_held_item_pipeline(&device, render_pass, entity_pipeline_layout, &entity_vert_spirv, &entity_frag_spirv, msaa_samples)?;

        // ── Particle pipeline (subpass 1) ──
        // Uses the dedicated particle pipeline layout created earlier in this
        // function (`particle_pipeline_layout`) whose set 0 is the chunk
        // layout and set 1 is the particle depth-input layout.
        let particle_pipeline = pipeline::create_particle_pipeline(
            &device,
            render_pass,
            particle_pipeline_layout,
            &particle_vert_spirv,
            &particle_frag_spirv,
            msaa_samples,
        )?;
        // MAX_PARTICLES (4096) * 32 B = 128 KB. Matches the particle_module
        // cap so we never overflow. Host-visible (no staging needed; we
        // rebuild contents each frame and the GPU reads on the same frame).
        let particle_instance_size: vk::DeviceSize = (crate::particle::MAX_PARTICLES
            * std::mem::size_of::<crate::particle::ParticleInstance>()) as vk::DeviceSize;
        let particle_instance_vbo = GpuBuffer::host_visible(
            &device,
            &alloc,
            particle_instance_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "particle_instance_vbo",
        )?;
        let overlay_pipeline_layout = crate::overlay::create_overlay_pipeline_layout(&device)?;
        let overlay_pipeline = crate::overlay::create_overlay_pipeline(&device, render_pass, overlay_pipeline_layout, &overlay_vert_spirv, &overlay_frag_spirv, msaa_samples)?;
        let overlay_vbo = GpuBuffer::host_visible(&device, &alloc, 8192 * 16, vk::BufferUsageFlags::VERTEX_BUFFER, "overlay_vbo")?;

        // -- Occlusion culling init --
        let occlusion_pipeline = pipeline::create_occlusion_pipeline(
            &device, render_pass, pipeline_layout, &aabb_vert_spirv, &aabb_frag_spirv, msaa_samples,
        )?;
        let aabb_index_buffer = pipeline::create_aabb_index_buffer(&device, &alloc)?;

        let mut occlusion_frames_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::OCCLUSION)
                .query_count(MAX_OCCLUSION_QUERIES);
            let pool = unsafe { device.create_query_pool(&pool_info, None) }
                .map_err(|e| anyhow!("create occlusion query_pool: {e:?}"))?;
            // Initial reset: puts all queries in "unavailable" state, satisfying
            // VUID-vkCmdBeginQuery-None-00807 ("query not reset") for the first
            // frame. Per-frame resets in record_chunk_passes keep it valid
            // thereafter.
            unsafe { device.reset_query_pool(pool, 0, MAX_OCCLUSION_QUERIES) };
            occlusion_frames_vec.push(OcclusionFrameData {
                query_pool: pool,
                used_queries: Vec::new(),
            });
        }
        log::info!("occlusion culling: {} queries per frame", MAX_OCCLUSION_QUERIES);

        // â”€â”€ Post pass init â”€â”€
        let post_render_pass = pipeline::create_post_render_pass(&device, swapchain_format)?;
        let post_descriptor_set_layout = pipeline::create_post_descriptor_set_layout(&device)?;
        let post_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: swapchain_images.len() as u32 * 2,  // scene_color + depth (SSAO)
        }];
        let post_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&post_pool_sizes)
            .max_sets(swapchain_images.len() as u32)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let post_descriptor_pool = unsafe { device.create_descriptor_pool(&post_pool_info, None) }
            .map_err(|e| anyhow!("create_post_descriptor_pool: {e:?}"))?;
        let post_layouts = vec![post_descriptor_set_layout; swapchain_images.len()];
        let post_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(post_descriptor_pool)
            .set_layouts(&post_layouts);
        let post_sets = unsafe { device.allocate_descriptor_sets(&post_alloc_info) }
            .map_err(|e| anyhow!("allocate_post_descriptor_sets: {e:?}"))?;
        let post_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_post_sampler: {e:?}"))?;
        let depth_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_depth_sampler: {e:?}"))?;
        let post_descriptor_sets: Vec<vk::DescriptorSet> = post_sets;
        {
            let img_infos: Vec<vk::DescriptorImageInfo> = post_descriptor_sets
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(offscreen_images[i].view)
                        .sampler(post_sampler)
                })
                .collect();
            let depth_img_infos: Vec<vk::DescriptorImageInfo> = post_descriptor_sets
                .iter()
                .map(|_| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(depth.view)
                        .sampler(depth_sampler)
                })
                .collect();
            let mut writes: Vec<vk::WriteDescriptorSet> = post_descriptor_sets
                .iter()
                .enumerate()
                .map(|(i, &set)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&img_infos[i..i + 1])
                })
                .collect();
            // Add depth buffer binding (binding 1) for SSAO.
            for (i, &set) in post_descriptor_sets.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&depth_img_infos[i..i + 1]),
                );
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let post_pipeline_layout =
            pipeline::create_post_pipeline_layout(&device, post_descriptor_set_layout)?;
        let post_pipeline = pipeline::create_post_pipeline(&device, post_render_pass, post_pipeline_layout, &post_vert_spirv, &post_frag_spirv)?;
        let post_framebuffers: Vec<vk::Framebuffer> = swapchain_image_views
            .iter()
            .map(|&view| {
                create_framebuffer_with(&device, post_render_pass, &[view], swapchain_extent)
            })
            .collect::<Result<Vec<_>>>()?;

        // â”€â”€ GPU timestamp query pool â”€â”€
        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;
        let timestamp_period = limits.timestamp_period;
        let query_pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(GPU_TIMESTAMP_COUNT * FRAMES_IN_FLIGHT as u32);
        let query_pool = unsafe { device.create_query_pool(&query_pool_info, None) }
            .map_err(|e| anyhow!("create_query_pool: {e:?}"))?;
        // Initial reset for the timestamp pool — required before the first
        // vkGetQueryPoolResults call (VUID-vkGetQueryPoolResults-None-09401).
        unsafe {
            device.reset_query_pool(
                query_pool,
                0,
                GPU_TIMESTAMP_COUNT * FRAMES_IN_FLIGHT as u32,
            )
        };

        // Capture config fields referenced *after* the `config,` shorthand
        // in the struct literal below moves `config` into Self.
        let cfg_particle_softness = config.particle_softness;
        let cfg_occlusion_culling = config.occlusion_culling;
        // Phase-1 GPU-driven pipeline: build before the struct literal so
        // `device`/`alloc` (moved into Self below) are still borrowable.
        let gpu_driven = if config.gpu_driven {
            match indirect::GpuDriven::new(
                &device, &alloc, render_pass, descriptor_set_layout, msaa_samples,
                config.gpu_meshing, command_pool, graphics_queue,
            ) {
                Ok(g) => Some(g),
                Err(e) => {
                    log::error!("GPU-driven init failed, falling back to legacy path: {e}");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            config,
            _entry: entry,
            instance,
            debug_messenger,
            physical_device,
            device,
            queues,
            graphics_queue,
            present_queue,
            surface,
            surface_instance,
            swapchain_device,
            alloc,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            depth: Some(depth),
            msaa_color: msaa_color_img,
            msaa_depth: msaa_depth_img,
            msaa_samples,
            offscreen_framebuffers,
            render_pass,
            pipeline_layout,
            pipeline,
            wireframe_pipeline,
            transparent_pipeline,
            wireframe_enabled: false,
            descriptor_pool,
            descriptor_set_layout,
            tile_remap_set_layout,
            command_pool,
            atlas,
            chunk_vert_spirv, chunk_frag_spirv,
            tile_material_ubo,
            pending_material_table: MaterialTable::empty(),
            material_table_dirty: true,
            ui_vert_spirv, ui_frag_spirv,
            sky_vert_spirv, sky_frag_spirv,
            shadow_vert_spirv, shadow_frag_spirv,
            post_vert_spirv, post_frag_spirv,
            entity_vert_spirv, entity_frag_spirv,
            fog_ubo,
            ui_pipeline,
            ui_pipeline_layout,
            ui_descriptor_set_layout,
            ui_descriptor_pool,
            ui_descriptor_set,
            font_texture,
            minimap_texture,
            ui_vbo,
            ui_ibo,
            sky_pipeline,
            sky_pipeline_layout,
            sky_descriptor_set_layout,
            sky_descriptor_pool,
            sky_descriptor_set,
            sky_ubo,
            panorama_pipeline,
            panorama_pipeline_layout,
            panorama_descriptor_set_layout,
            panorama_descriptor_pool,
            panorama_descriptor_set,
            panorama,
            panorama_placeholder,
            panorama_vert_spirv,
            panorama_frag_spirv,
            entity_pipeline,
            entity_pipeline_layout,
            entity_vbo,
            entity_ibo,
            entity_held_pipeline,
            particle_pipeline,
            particle_pipeline_layout,
            particle_depth_set_layout,
            particle_depth_descriptor_pool,
            particle_depth_descriptor_sets,
            particle_vert_spirv,
            particle_frag_spirv,
            particle_instance_vbo,
            particle_manager: crate::particle::ParticleManager::new(
                crate::particle::DEFAULT_MAX_PARTICLES,
            ),
            particle_softness: cfg_particle_softness,
            overlay_pipeline,
            overlay_pipeline_layout,
            overlay_vbo,
            overlay_vert_spirv,
            overlay_frag_spirv,
            overlay_data: crate::overlay::OverlayData::default(),
            occlusion_pipeline,
            aabb_index_buffer,
            aabb_vert_spirv,
            aabb_frag_spirv,
            occlusion_state: RwLock::new(HashMap::new()),
            occlusion_frames: RwLock::new(occlusion_frames_vec),
            occlusion_culling_enabled: cfg_occlusion_culling,
            models: Vec::new(),
            shadow_render_pass,
            shadow_pipeline,
            shadow_pipeline_layout,
            shadow_image,
            shadow_layer_views,
            shadow_sampler,
            shadow_framebuffers,
            shadow_ubo_data: ShadowUbo::default(),
            offscreen_images,
            post_render_pass,
            post_pipeline,
            post_pipeline_layout,
            post_descriptor_set_layout,
            post_descriptor_pool,
            post_descriptor_sets,
            post_sampler,
            depth_sampler,
            post_framebuffers,
            post_params: [1.0, 0.0, 0.0, 0.0],
            ssao_params: [1.5, 0.025, 1.0, 1.0],
            proj_params: [0.1, 500.0, 0.0, 0.0],
            // Slice 2 fields
            scene_opaque_color,
            scene_opaque_sampler,
            scene_depth_sampler,
            transparent_render_pass,
            transparent_framebuffers,
            // Slice 3 (reflections) fields
            scene_opaque_depth,
            reflection_ubo,
            reflection_strength: 0.85,
            ssr_depth_valid,
            depth_resolve_active: msaa_samples != vk::SampleCountFlags::TYPE_1
                && depth_resolve_mode.is_some(),
            frames,
            chunks: RwLock::new(HashMap::new()),
            gpu_driven,
            query_pool,
            timestamp_period,
            timings: GpuTimings::default(),
            needs_resize: false,
            frame_counter: 0,
            sky_horizon: [0.52, 0.72, 0.95],
            sky_zenith: [0.35, 0.55, 0.90],
            sky_fog: [0.62, 0.80, 0.96],
            sky_ambient: 1.0,
            sky_underwater: false,
            sun_dir: [0.0, 1.0, 0.0],
            pending_destruction: Vec::new(),
        })
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    /// Mark the swapchain for recreation on the next draw (call on window resize).
    pub fn resize(&mut self) {
        self.needs_resize = true;
    }

    /// Current swapchain extent (window drawable size in pixels).
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    /// Latest GPU timing results (1-2 frame lag).
    pub fn latest_timings(&self) -> GpuTimings {
        self.timings
    }

    /// Rebuild the texture atlas from `config.textures_dir` and swap it into
    /// the chunk + UI descriptor sets in place. Safe to call repeatedly;
    /// failures leave the existing atlas untouched.
    pub fn reload_atlas(&mut self) -> Result<()> {
        unsafe { self.device.device_wait_idle()?; }
        let dir: &Path = self
            .config
            .textures_dir
            .as_deref()
            .filter(|d| d.is_dir())
            .unwrap_or_else(|| Path::new(""));
        let pack_mapping = load_pack_mapping(
            dir,
            self.config.texture_packs_dir.as_deref(),
        );
        let atlas_pixels = build_atlas_with_packs(dir, pack_mapping.as_ref(), self.config.texture_packs_dir.as_deref());
        let new_atlas = AtlasTexture::new(
            &self.device,
            &self.alloc,
            self.command_pool,
            self.graphics_queue,
            &atlas_pixels,
        )
        .context("reload_atlas: AtlasTexture::new")?;
        // Re-bind chunk descriptor sets (binding 1 = atlas sampler, binding 5 = material table).
        for frame in self.frames.iter() {
            pipeline::update_descriptor_set(
                &self.device,
                frame.descriptor_set,
                frame.camera_ubo.buffer,
                self.fog_ubo.buffer,
                new_atlas.view,
                new_atlas.sampler,
                self.shadow_image.view,
                self.shadow_sampler,
                frame.shadow_ubo.buffer,
                self.tile_material_ubo.buffer,
                MaterialTable::SIZE_BYTES as vk::DeviceSize,
                self.scene_opaque_color.view,
                self.scene_opaque_sampler,
                self.scene_opaque_depth.view,
                self.scene_depth_sampler,
                self.reflection_ubo.buffer,
            );
        }
        // UI descriptor set (binding 0 = block atlas, binding 1 = font atlas, binding 2 = minimap).
        pipeline::update_ui_descriptor_set(
            &self.device,
            self.ui_descriptor_set,
            new_atlas.view,
            new_atlas.sampler,
            self.font_texture.view,
            self.font_texture.sampler,
            self.minimap_texture.view,
            self.minimap_texture.sampler,
        );
        self.atlas.destroy_in_place(&self.device, &self.alloc);
        self.atlas = new_atlas;
        log::info!("atlas hot-reloaded");
        Ok(())
    }

    /// Hot-swap a subset of `RendererConfig` at runtime. Fields that would
    /// require a full re-init (vsync, validation, swapchain dimensions) are
    /// logged but not changed - those need a restart.
    pub fn reload_config(&mut self, new_config: &RendererConfig) -> Result<()> {
        unsafe { self.device.device_wait_idle()?; }
        let textures_changed = new_config.textures_dir != self.config.textures_dir;
        let packs_changed = new_config.texture_packs_dir != self.config.texture_packs_dir;
        let fog_changed = new_config.fog_color != self.config.fog_color
            || (new_config.fog_distance - self.config.fog_distance).abs() > f32::EPSILON;

        self.config.fog_color = new_config.fog_color;
        self.config.fog_distance = new_config.fog_distance;
        self.config.textures_dir = new_config.textures_dir.clone();
        self.config.texture_packs_dir = new_config.texture_packs_dir.clone();
        self.config.shader_dir = new_config.shader_dir.clone();
        self.config.clear_color = new_config.clear_color;
        self.config.validation = new_config.validation;
        self.config.vsync = new_config.vsync;

        if textures_changed || packs_changed {
            self.reload_atlas()?;
        } else if fog_changed {
            log::info!(
                "renderer fog config hot-reloaded: {:?} distance={}",
                self.config.fog_color,
                self.config.fog_distance
            );
        }
        if fog_changed || textures_changed {
            log::info!("renderer config hot-reloaded");
        }
        Ok(())
    }

    /// Recompile a single shader source file via `glslangValidator` and
    /// destroy + recreate the affected Vulkan pipeline(s) in place. Maps
    /// `name` to its pipeline family:
    ///   * `chunk.{vert,frag}` -> all 3 chunk pipelines (FILL, LINE, transparent)
    ///   * `ui.{vert,frag}` -> ui pipeline
    ///   * `sky.{vert,frag}` -> sky pipeline
    ///   * `shadow.{vert,frag}` -> shadow pipeline
    ///   * `post.{vert,frag}` -> post pipeline
    /// Callers must invoke this on the renderer's own thread (i.e. from the
    /// engine's frame loop, NOT from the file-watcher thread).
    pub fn reload_shader(&mut self, name: &str) -> Result<()> {
        let dir = self
            .config
            .shader_dir
            .as_ref()
            .ok_or_else(|| anyhow!("reload_shader: no shader_dir configured"))?;
        let src = dir.join(name);
        let bytes = crate::hot_reload::compile_shader(&src)
            .with_context(|| format!("reload_shader({name})"))?;
        unsafe { self.device.device_wait_idle()?; }
        match name {
            "chunk.vert" => {
                self.chunk_vert_spirv = bytes;
                self.recreate_chunk_pipelines()
            }
            "chunk.frag" => {
                self.chunk_frag_spirv = bytes;
                self.recreate_chunk_pipelines()
            }
            "ui.vert" => {
                self.ui_vert_spirv = bytes;
                self.recreate_ui_pipeline()
            }
            "ui.frag" => {
                self.ui_frag_spirv = bytes;
                self.recreate_ui_pipeline()
            }
            "sky.vert" => {
                self.sky_vert_spirv = bytes;
                self.recreate_sky_pipeline()
            }
            "sky.frag" => {
                self.sky_frag_spirv = bytes;
                self.recreate_sky_pipeline()
            }
            "shadow.vert" => {
                self.shadow_vert_spirv = bytes;
                self.recreate_shadow_pipeline()
            }
            "shadow.frag" => {
                self.shadow_frag_spirv = bytes;
                self.recreate_shadow_pipeline()
            }
            "post.vert" => {
                self.post_vert_spirv = bytes;
                self.recreate_post_pipeline()
            }
            "post.frag" => {
                self.post_frag_spirv = bytes;
                self.recreate_post_pipeline()
            }
            _ => Err(anyhow!("reload_shader: unknown shader family for {name}")),
        }
        .map(|_| {
            log::info!(
                "reload_shader({name}) ok: pipeline(s) recreated; expect 1 frame stutter"
            );
        })
    }

    fn recreate_chunk_pipelines(&mut self) -> Result<()> {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline(self.wireframe_pipeline, None);
            self.device.destroy_pipeline(self.transparent_pipeline, None);
        }
        self.pipeline = vk::Pipeline::null();
        self.wireframe_pipeline = vk::Pipeline::null();
        self.transparent_pipeline = vk::Pipeline::null();
        self.pipeline = pipeline::create_graphics_pipeline(
            &self.device,
            self.render_pass,
            self.pipeline_layout,
            vk::PolygonMode::FILL,
            vk::CullModeFlags::BACK,
            &self.chunk_vert_spirv,
            &self.chunk_frag_spirv,
            self.msaa_samples,
            true,
        )?;
        self.wireframe_pipeline = pipeline::create_graphics_pipeline(
            &self.device,
            self.render_pass,
            self.pipeline_layout,
            vk::PolygonMode::LINE,
            vk::CullModeFlags::BACK,
            &self.chunk_vert_spirv,
            &self.chunk_frag_spirv,
            self.msaa_samples,
            true,
        )?;
        self.transparent_pipeline = pipeline::create_graphics_pipeline(
            &self.device,
            self.transparent_render_pass,
            self.pipeline_layout,
            vk::PolygonMode::FILL,
            vk::CullModeFlags::NONE,
            &self.chunk_vert_spirv,
            &self.chunk_frag_spirv,
            vk::SampleCountFlags::TYPE_1,
            false,
        )?;
        Ok(())
    }

    fn recreate_ui_pipeline(&mut self) -> Result<()> {
        unsafe {
            self.device.destroy_pipeline(self.ui_pipeline, None);
        }
        self.ui_pipeline = vk::Pipeline::null();
        self.ui_pipeline = pipeline::create_ui_pipeline(
            &self.device,
            self.render_pass,
            self.ui_pipeline_layout,
            &self.ui_vert_spirv,
            &self.ui_frag_spirv,
            self.msaa_samples,
        )?;
        Ok(())
    }

    fn recreate_sky_pipeline(&mut self) -> Result<()> {
        unsafe {
            self.device.destroy_pipeline(self.sky_pipeline, None);
        }
        self.sky_pipeline = vk::Pipeline::null();
        self.sky_pipeline = pipeline::create_sky_pipeline(
            &self.device,
            self.render_pass,
            self.sky_pipeline_layout,
            &self.sky_vert_spirv,
            &self.sky_frag_spirv,
            self.msaa_samples,
        )?;
        Ok(())
    }

    fn recreate_shadow_pipeline(&mut self) -> Result<()> {
        unsafe {
            self.device.destroy_pipeline(self.shadow_pipeline, None);
        }
        self.shadow_pipeline = vk::Pipeline::null();
        self.shadow_pipeline = pipeline::create_shadow_pipeline(
            &self.device,
            self.shadow_render_pass,
            self.shadow_pipeline_layout,
            &self.shadow_vert_spirv,
            &self.shadow_frag_spirv,
        )?;
        Ok(())
    }

    fn recreate_post_pipeline(&mut self) -> Result<()> {
        unsafe {
            self.device.destroy_pipeline(self.post_pipeline, None);
        }
        self.post_pipeline = vk::Pipeline::null();
        self.post_pipeline = pipeline::create_post_pipeline(
            &self.device,
            self.post_render_pass,
            self.post_pipeline_layout,
            &self.post_vert_spirv,
            &self.post_frag_spirv,
        )?;
        Ok(())
    }

    /// Set dynamic sky parameters for the day/night cycle. Called each frame
    /// by the engine. Updates the fog UBO's colour + ambient brightness, and
    /// the clear colour used for the sky.
    pub fn set_sky(
        &mut self,
        horizon: [f32; 3],
        zenith: [f32; 3],
        fog: [f32; 3],
        ambient: f32,
        underwater: bool,
    ) {
        self.sky_horizon = horizon;
        self.sky_zenith = zenith;
        self.sky_fog = fog;
        self.sky_ambient = ambient;
        self.sky_underwater = underwater;

        // Update the clear colour to match the horizon sky colour, dimmed by ambient.
        if underwater {
            self.config.clear_color = [0.01, 0.05, 0.15, 1.0];
        } else {
            self.config.clear_color = [
                horizon[0] * ambient.max(0.05),
                horizon[1] * ambient.max(0.05),
                horizon[2] * ambient.max(0.05),
                1.0,
            ];
        }
    }

    /// Set the sun direction explicitly (called by the engine with the day params).
    pub fn set_sun_dir(&mut self, dir: [f32; 3]) {
        self.sun_dir = dir;
    }

    /// Set cascaded shadow map data (4 light-space VP matrices + cascade splits).
    pub fn set_shadow_data(
        &mut self,
        cascade_vps: [[f32; 16]; 4],
        cascade_splits: [f32; 4],
        light_dir_and_bias: [f32; 4],
    ) {
        self.shadow_ubo_data = ShadowUbo {
            cascade_vps,
            cascade_splits,
            light_dir_and_bias,
        };
    }

    /// Set post-processing parameters (exposure, vignette strength, time, underwater).
    pub fn set_post_params(&mut self, exposure: f32, vignette: f32, time: f32, underwater: bool) {
        self.post_params = [exposure, vignette, time, if underwater { 1.0 } else { 0.0 }];
    }

    /// Set SSAO parameters (radius, bias, strength, enabled).
    pub fn set_ssao_params(&mut self, radius: f32, bias: f32, strength: f32, enabled: bool) {
        // SSAO requires single-sample depth buffer; disable when MSAA is active.
        let effective = enabled && self.msaa_samples == vk::SampleCountFlags::TYPE_1;
        self.ssao_params = [radius, bias, strength, if effective { 1.0 } else { 0.0 }];
    }

    /// Push the tile material lookup table (descriptor binding 5) for the
    /// next frame. The engine typically builds this once per frame from the
    /// block registry + the water-level/strength scalars from the active
    /// [`EngineConfig`]. Cheap ownership copy (≈4 KB) so no aliasing
    /// concerns. The actual GPU upload happens in `flush_pending_ubos` so
    /// callers can drop their reference to the table immediately.
    pub fn set_tile_material_table(&mut self, table: MaterialTable) {
        self.pending_material_table = table;
        self.material_table_dirty = true;
    }

    /// Set projection parameters for SSAO depth linearization.
    pub fn set_proj_params(&mut self, near: f32, far: f32, screen_w: f32, screen_h: f32) {
        self.proj_params = [near, far, screen_w, screen_h];
    }

    /// Set the master reflection strength in [0, 1] (chunk shader binding 8,
    /// `sun_dir_str.w`). 0 disables every reflection path (water, glass,
    /// opaque REFLECTIVE tiles). Written into the reflection UBO by the next
    /// `flush_pending_ubos`.
    pub fn set_reflection_strength(&mut self, strength: f32) {
        self.reflection_strength = strength.clamp(0.0, 1.0);
    }

    /// Flush pending sky/fog/sun UBO data to the GPU.
    /// Must be called after the frame fence wait to avoid data races.
    fn flush_pending_ubos(&mut self) {
        // Tile material table: copy the engine-pushed scratch to the UBO if
        // anything was pushed this frame. We only write when dirty so the
        // per-frame perf cost is zero in the steady state.
        if self.material_table_dirty {
            if let Ok(slice) = self.tile_material_ubo.mapped_slice_mut() {
                let bytes: &[u8] = bytemuck::bytes_of(&self.pending_material_table);
                slice[..bytes.len()].copy_from_slice(bytes);
                if let Err(e) = self.tile_material_ubo.flush_whole(&self.device) {
                    log::warn!("tile_material_ubo flush failed: {e}");
                }
            }
            self.material_table_dirty = false;
        }


        // Fog UBO
        let (fog_color, ambient_val) = if self.sky_underwater {
            ([0.05, 0.15, 0.35], self.sky_ambient * 0.6)
        } else {
            (self.sky_fog, self.sky_ambient)
        };
        let fog_data = FogUbo {
            color_and_density: [fog_color[0], fog_color[1], fog_color[2], 1.0],
            ambient_and_sun: [ambient_val, 0.0, 1.0, 0.0],
        };
        if let Ok(slice) = self.fog_ubo.mapped_slice_mut() {
            let bytes: &[u8] = bytemuck::bytes_of(&fog_data);
            slice[..bytes.len()].copy_from_slice(bytes);
            if let Err(e) = self.fog_ubo.flush_whole(&self.device) {
                log::warn!("fog_ubo flush failed: {e}");
            }
        }

        // Sky UBO
        let data = SkyUbo {
            horizon: [
                self.sky_horizon[0],
                self.sky_horizon[1],
                self.sky_horizon[2],
                1.0,
            ],
            zenith: [
                self.sky_zenith[0],
                self.sky_zenith[1],
                self.sky_zenith[2],
                1.0,
            ],
            sun_dir: [self.sun_dir[0], self.sun_dir[1], self.sun_dir[2], 0.0],
        };
        if let Ok(slice) = self.sky_ubo.mapped_slice_mut() {
            let bytes: &[u8] = bytemuck::bytes_of(&data);
            slice[..bytes.len()].copy_from_slice(bytes);
            if let Err(e) = self.sky_ubo.flush_whole(&self.device) {
                log::warn!("sky_ubo flush failed: {e}");
            }
        }

        // Reflection/environment UBO (chunk binding 8). Mirrors the sky UBO
        // colours so the chunk shader can evaluate the same analytic sky for
        // reflected rays; near/far come from `proj_params` (set via
        // `set_proj_params` each frame by the engine).
        let refl_data = pipeline::ReflectionUbo {
            sky_horizon: [self.sky_horizon[0], self.sky_horizon[1], self.sky_horizon[2], 1.0],
            sky_zenith: [self.sky_zenith[0], self.sky_zenith[1], self.sky_zenith[2], 1.0],
            sun_dir_str: [
                self.sun_dir[0],
                self.sun_dir[1],
                self.sun_dir[2],
                self.reflection_strength,
            ],
            proj_misc: [
                self.proj_params[0],
                self.proj_params[1],
                if self.sky_underwater { 1.0 } else { 0.0 },
                if self.ssr_depth_valid { 1.0 } else { 0.0 },
            ],
        };
        if let Ok(slice) = self.reflection_ubo.mapped_slice_mut() {
            let bytes: &[u8] = bytemuck::bytes_of(&refl_data);
            slice[..bytes.len()].copy_from_slice(bytes);
            if let Err(e) = self.reflection_ubo.flush_whole(&self.device) {
                log::warn!("reflection_ubo flush failed: {e}");
            }
        }

        // Shadow UBO (per-frame).
        let shadow_bytes: &[u8] = bytemuck::bytes_of(&self.shadow_ubo_data);
        for frame in self.frames.iter_mut() {
            if let Ok(slice) = frame.shadow_ubo.mapped_slice_mut() {
                slice[..shadow_bytes.len()].copy_from_slice(shadow_bytes);
                if let Err(e) = frame.shadow_ubo.flush_whole(&self.device) {
                    log::warn!("shadow_ubo flush failed: {e}");
                }
            }
        }
    }

    /// Upload (or replace) a batch of chunk meshes. Done via a single one-time
    /// staging command buffer for efficiency.
    /// Upload new RGBA pixel data to the minimap GPU texture.
    /// `data.len()` must equal `width * height * 4` (currently 256×256 = 262144 bytes).
    pub fn upload_minimap_texture(&mut self, data: &[u8]) {
        if let Err(e) = self.minimap_texture.upload(
            data,
            &self.device,
            self.command_pool,
            self.graphics_queue,
        ) {
            log::warn!("upload_minimap_texture: {e}");
        }
    }

    pub fn upload_chunks(&mut self, uploads: Vec<ChunkUpload>) {
        if uploads.is_empty() {
            return;
        }
        // Phase-1 GPU-driven path: route to the mega-buffer subsystem.
        if let Some(gpu) = self.gpu_driven.as_mut() {
            gpu.upload(
                &self.device,
                &self.alloc,
                self.command_pool,
                self.graphics_queue,
                uploads,
            );
            return;
        }
        let device = &self.device;
        let alloc = &self.alloc;
        let pool = self.command_pool;
        let queue = self.graphics_queue;

        // Create device-local buffers + staging buffers for each upload.
        struct Pending {
            pos: ChunkPos,
            pass: MeshPass,
            vbo: GpuBuffer,
            ibo: GpuBuffer,
            staging: GpuBuffer,
            v_offset: vk::DeviceSize,
            i_size: vk::DeviceSize,
            index_count: u32,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(uploads.len());

        // Build one big staging buffer per chunk (vertices then indices packed).
        let cmd = match begin_one_time(device, pool) {
            Ok(c) => c,
            Err(e) => {
                log::error!("begin_one_time failed: {e}");
                return;
            }
        };

        for u in uploads {
            if u.vertices.is_empty() || u.indices.is_empty() {
                continue;
            }
            let v_size = u.vertices.len() as vk::DeviceSize;
            let i_size = u.indices.len() as vk::DeviceSize;
            let staging = match GpuBuffer::host_visible(
                device,
                alloc,
                v_size + i_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                "chunk_staging",
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("staging alloc failed: {e}");
                    // End the recording into EXECUTABLE state before freeing
                    // — `vkFreeCommandBuffers` requires the cmd's state to be
                    // INITIAL or EXECUTABLE, and validation reports "in use"
                    // (VUID-vkFreeCommandBuffers-pCommandBuffers-00047) for
                    // primary cmds still in RECORDING after `begin_command_buffer`.
                    // An un-endable cmd is reset to INITIAL via pool reset.
                    let _ = unsafe { device.end_command_buffer(cmd) };
                    unsafe { device.free_command_buffers(pool, &[cmd]); }
                    continue;
                }
            };
            let mut staging = staging;
            if let Err(e) = staging.upload(device, &u.vertices) {
                log::error!("staging vertex upload: {e}");
                staging.destroy(device, alloc);
                let _ = unsafe { device.end_command_buffer(cmd) };
                unsafe { device.free_command_buffers(pool, &[cmd]); }
                continue;
            }
            // Copy indices after vertices in the staging buffer. We bypass
            // `upload()` here because indices follow vertices in the same
            // buffer at a non-zero offset, so the helper can't write both
            // with one call. The flush below covers BOTH the vertex run
            // (already written+flushed by `upload()`) and this index tail;
            // flushing twice in the same frame is safe and idempotent so
            // this isn't worth optimizing unless we add a `write_range()`
            // helper.
            {
                let slice = match staging.mapped_slice_mut() {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("staging map: {e}");
                        staging.destroy(device, alloc);
                        let _ = unsafe { device.end_command_buffer(cmd) };
                        unsafe { device.free_command_buffers(pool, &[cmd]); }
                        continue;
                    }
                };
                slice[v_size as usize..(v_size + i_size) as usize].copy_from_slice(&u.indices);
                if let Err(e) = staging.flush_whole(device) {
                    log::error!("staging flush: {e}");
                }
            }

            let vbo = match GpuBuffer::device_local(
                device,
                alloc,
                v_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                "chunk_vbo",
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("vbo alloc: {e}");
                    staging.destroy(device, alloc);
                    let _ = unsafe { device.end_command_buffer(cmd) };
                    unsafe { device.free_command_buffers(pool, &[cmd]); }
                    continue;
                }
            };
            let ibo = match GpuBuffer::device_local(
                device,
                alloc,
                i_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                "chunk_ibo",
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("ibo alloc: {e}");
                    staging.destroy(device, alloc);
                    vbo.destroy(device, alloc);
                    let _ = unsafe { device.end_command_buffer(cmd) };
                    unsafe { device.free_command_buffers(pool, &[cmd]); }
                    continue;
                }
            };

            // Record copies.
            let v_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(v_size);
            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, vbo.buffer, &[v_region]);
            }
            let i_region = vk::BufferCopy::default()
                .src_offset(v_size)
                .dst_offset(0)
                .size(i_size);
            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, ibo.buffer, &[i_region]);
            }

            pending.push(Pending {
                pos: u.pos,
                pass: u.pass,
                vbo,
                ibo,
                staging,
                v_offset: v_size,
                i_size,
                index_count: u.index_count,
            });
        }

        // Use a per-batch fence instead of queue_wait_idle to avoid stalling
        // the entire GPU queue (which kills frame rate during chunk loading).
        // Submit WITHOUT a fence wait. The previous code created an
        // `upload_fence` per batch and called `device.wait_for_fences(..., u64::MAX)`
        // which stalled the main thread for many seconds during initial
        // world streaming ("freeze #2" in the user reports). We still submit
        // in-order on the same graphics queue, so the GPU processes the upload
        // copies before any subsequent `draw_frame` records. The buffers
        // and the per-batch fence (gone now — we drop it entirely) are
        // reclaimed asynchronously in `drain_pending_destructions` once
        // `FRAMES_IN_FLIGHT` frames have elapsed since submission.
        unsafe {
            if let Err(e) = device.end_command_buffer(cmd) {
                log::error!("end_command_buffer (upload) failed: {e:?}");
                return;
            }
        }
        let submit_frame = self.frame_counter;
        // Defer cmd buffer reclamation to the existing
        // `drain_pending_destructions` path (the same one used for
        // staging + chunk VBO reclaim below). The earlier
        // `queue_wait_idle` + `free_command_buffers` pair was trying
        // to satisfy VUID-vkFreeCommandBuffers-pCommandBuffers-00047
        // ("cmd in use") AND avoid the STATUS_ACCESS_VIOLATION segfault
        // we saw after `spawn ready`, but it DEADLOCKED the engine:
        // `vkQueueWaitIdle` blocks the main thread, which is the same
        // thread that pumps winit window events. With the main thread
        // stalled, the swapchain `image_available` semaphore never gets
        // a present-paired signal (the OS won't present without the
        // event loop advancing), so the queue never drains, so the
        // wait never returns — TDR / force-close after ~2 s.
        //
        // The fix is to submit with a DEDICATED per-batch fence, push
        // both the cmd and the fence to `pending_destruction`, and let
        // the next frame's drain `wait_for_fences` on it. That avoids
        // both the deadlock (no main-thread wait) AND the spec
        // violation (we only `vkFreeCommandBuffers` after the fence
        // has signalled, satisfying the "not pending" requirement).
        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        let batch_fence = unsafe {
            match device.create_fence(&vk::FenceCreateInfo::default(), None) {
                Ok(f) => f,
                Err(e) => {
                    log::error!("upload create_fence failed: {e:?}");
                    // Submission failure path: we're about to return
                    // without recording, so free the cmd buffer
                    // immediately (no fence was ever registered).
                    device.free_command_buffers(pool, &[cmd]);
                    return;
                }
            }
        };
        if let Err(e) = unsafe { device.queue_submit(queue, &[submit_info], batch_fence) } {
            log::error!("upload queue_submit failed: {e}");
            unsafe {
                device.destroy_fence(batch_fence, None);
                device.free_command_buffers(pool, &[cmd]);
            }
            return;
        }
        self.pending_destruction.push((
            submit_frame,
            PendingDestroy::CommandBuffer {
                cmd,
                fence: batch_fence,
            },
        ));

        // Insert into the chunk map. Each chunk can have both an opaque and a
        // transparent pass â€” store them in the same ChunkBuffers entry rather
        // than overwriting one with the other.
        let mut chunks = self.chunks.write();
        for p in pending {
            let entry = chunks.entry(p.pos).or_insert_with(ChunkBuffers::new);
            let slot = match p.pass {
                MeshPass::Opaque => &mut entry.opaque,
                MeshPass::Transparent => &mut entry.transparent,
            };
            // Replace old buffers for this pass — defer destruction so the
            // GPU doesn't see its memory yanked mid-frame.
            if let Some(old) = slot.take() {
                self.pending_destruction
                    .push((submit_frame, PendingDestroy::ChunkValue(old)));
            }
            *slot = Some(PassBuffers {
                vbo: p.vbo,
                ibo: p.ibo,
                index_count: p.index_count,
            });
            // Staging: defer too.
            self.pending_destruction
                .push((submit_frame, PendingDestroy::Staging(p.staging)));
            let _ = (p.v_offset, p.i_size); // already used above

            // Reset occlusion state so the chunk is treated as visible next frame.
            if self.occlusion_culling_enabled {
                let mut occ = self.occlusion_state.write();
                if let Some(state) = occ.get_mut(&p.pos) {
                    state.was_visible = true;
                    state.consecutive_invisible = 0;
                }
            }
        }
    }

    /// Remove a chunk's GPU buffers (called when the streamer unloads it).
    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        if let Some(gpu) = self.gpu_driven.as_mut() {
            gpu.remove(pos);
            return;
        }
        let removed = self.chunks.write().remove(&pos);
        // Defer destruction of the GPU buffers so we don't yank memory out
        // from under the GPU while it's mid-draw with these chunks. The
        // same `drain_pending_destructions` reclaim path that `upload_chunks`
        // uses will pick these up in a couple of frames.
        let submit_frame = self.frame_counter;
        if let Some(bufs) = removed {
            if let Some(opaque) = bufs.opaque {
                self.pending_destruction
                    .push((submit_frame, PendingDestroy::ChunkValue(opaque)));
            }
            if let Some(transparent) = bufs.transparent {
                self.pending_destruction
                    .push((submit_frame, PendingDestroy::ChunkValue(transparent)));
            }
        }
        // Clean up occlusion state so the query index can be reused.
        if self.occlusion_culling_enabled {
            let mut occ = self.occlusion_state.write();
            occ.remove(&pos);
        }
    }

    /// Number of chunks currently on the GPU.
    pub fn chunk_count(&self) -> usize {
        if let Some(gpu) = self.gpu_driven.as_ref() {
            return gpu.chunk_count();
        }
        self.chunks.read().len()
    }

    /// Update the block properties table for the Phase-2 GPU compute mesher.
    /// Call after the block registry is loaded or changes. No-op if GPU
    /// meshing is disabled.
    pub fn set_block_properties(&mut self, props: &[crate::BlockPropertiesGpu]) {
        if let Some(gpu) = self.gpu_driven.as_mut() {
            gpu.set_block_properties(&self.device, &self.alloc, props);
        }
    }

    /// GPU-mesh a chunk from raw voxel data (18³ u16, 16³ chunk + 1-voxel
    /// border). Dispatches the compute mesher and inserts the result into the
    /// mega VBO/IBO. No-op (returns false) if GPU meshing is disabled.
    pub fn upload_chunk_gpu_mesh(
        &mut self, pos: voxel_core::ChunkPos, pass: MeshPass, voxels: &[u16],
    ) -> bool {
        if let Some(gpu) = self.gpu_driven.as_mut() {
            return gpu.upload_chunk_gpu_mesh(
                &self.device, &self.alloc, self.command_pool, self.graphics_queue,
                pos, pass, voxels,
            );
        }
        false
    }

    /// Load a glTF model from disk and register it in the model registry.
    /// Returns the model_id (index) for use with `ModelRef` components.
    pub fn load_model(&mut self, path: &std::path::Path) -> Result<u32> {
        let model = crate::model::load_model(
            &self.device,
            &self.alloc,
            self.command_pool,
            self.graphics_queue,
            path,
        )?;
        let id = self.models.len() as u32;
        self.models.push(model);
        log::info!("loaded model '{}' (id={})", path.display(), id);
        Ok(id)
    }

    /// Get a reference to a loaded model by id.

    /// Set overlay data for the current frame (brush wireframe, etc).
    pub fn set_overlay(&mut self, data: crate::overlay::OverlayData) {
        self.overlay_data = data;
    }

    /// Record the overlay pass: draw wireframe lines.
    fn record_overlay_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        view_proj: glam::Mat4,
    ) {
        if self.overlay_data.lines.is_empty() {
            return;
        }
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.overlay_pipeline,
            );
            let vbo = [self.overlay_vbo.buffer];
            device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
            let push = crate::overlay::OverlayPushConstants {
                view_proj: view_proj.to_cols_array_2d(),
            };
            device.cmd_push_constants(
                cmd,
                self.overlay_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
            let vertex_count = (self.overlay_data.lines.len() * 2) as u32;
            device.cmd_draw(cmd, vertex_count, 1, 0, 0);
        }
    }
    pub fn get_model(&self, id: u32) -> Option<&crate::model::Model> {
        self.models.get(id as usize)
    }

    /// Returns (total_allocated_bytes, total_reserved_bytes) from the
    /// GPU memory allocator.
    pub fn allocator_stats(&self) -> (u64, u64) {
        self.alloc.stats()
    }

    /// Total index count across all chunk buffers (opaque + transparent).
    /// Returns (vertex_count_estimate, index_count).
    pub fn chunk_buffer_stats(&self) -> (u32, u32) {
        let chunks = self.chunks.read();
        let mut index_count = 0u32;
        for (_, bufs) in chunks.iter() {
            if let Some(ref opaque) = bufs.opaque {
                index_count += opaque.index_count;
            }
            if let Some(ref transparent) = bufs.transparent {
                index_count += transparent.index_count;
            }
        }
        // Estimate vertex count: chunk vertices are ~4 per face, indices are
        // 6 per face (2 triangles), so vertices ≈ indices * 4/6.
        let vertex_estimate = index_count * 4 / 6;
        (vertex_estimate, index_count)
    }

    /// Toggle wireframe rendering.
    pub fn toggle_wireframe(&mut self) {
        self.wireframe_enabled = !self.wireframe_enabled;
    }

    /// Whether wireframe rendering is active.
    pub fn is_wireframe(&self) -> bool {
        self.wireframe_enabled
    }

    /// Get the active chunk pipeline (fill or wireframe).
    fn active_pipeline(&self) -> vk::Pipeline {
        if self.wireframe_enabled {
            self.wireframe_pipeline
        } else {
            self.pipeline
        }
    }

    /// Destroy any GPU resources enqueued by previous `upload_chunks` /
    /// `remove_chunk` calls that are now at least `FRAMES_IN_FLIGHT` frames
    /// old. Safe to call repeatedly; each call is O(N) over a queue that
    /// holds at most a handful of items in steady state (most chunks are
    /// uploaded once and stay uploaded until unloaded).
    fn drain_pending_destructions(&mut self) {
        // An item submitted at `submitted_at` is safe to destroy once
        // `self.frame_counter >= submitted_at + FRAMES_IN_FLIGHT` because by
        // that point the corresponding per-frame in-flight fence (which was
        // submitted-after the upload on the same queue) has already
        // signalled — i.e. the GPU is past every resource that batch used.
        let cutoff = self.frame_counter.saturating_sub(FRAMES_IN_FLIGHT);
        let device = &self.device;
        let alloc = &self.alloc;
        self.pending_destruction.retain_mut(|(submitted_at, item)| {
            if *submitted_at <= cutoff {
                // `GpuBuffer::destroy_in_place` takes `&mut self` (vs.
                // `destroy` which takes `self` by move). We can only use
                // `&mut` here because `item` is borrowed mutably by
                // `retain_mut`'s closure — moving out of a borrow isn't
                // allowed. The two operations are semantically the same:
                // release the underlying Vulkan buffer + alloc.
                match item {
                    PendingDestroy::ChunkValue(pb) => {
                        pb.vbo.destroy_in_place(device, alloc);
                        pb.ibo.destroy_in_place(device, alloc);
                    }
                    PendingDestroy::Staging(b) => {
                        b.destroy_in_place(device, alloc);
                    }
                    PendingDestroy::CommandBuffer { cmd, fence } => {
                        // Wait for the upload's dedicated fence. This
                        // is what guarantees the cmd is no longer in
                        // the "pending" state (VUID 00047 — required
                        // for `vkFreeCommandBuffers`). The wait is on
                        // the caller's main thread, but on a normal
                        // frame the GPU has had ~16 ms (one frame
                        // period) to complete the upload, so this
                        // returns immediately. Only stalls on heavy
                        // backed-up GPU — and even then, only inside
                        // the `drain_pending_destructions` pre-amble
                        // (one entry per chunk, batched), not at the
                        // top of every frame.
                        unsafe {
                            // `retain_mut`'s closure borrows the
                            // fields as `&mut`, so deref into the
                            // handles before passing to Vulkan FFI.
                            let cmd_handle = *cmd;
                            let fence_handle = *fence;
                            if let Err(e) =
                                device.wait_for_fences(&[fence_handle], true, u64::MAX)
                            {
                                log::warn!(
                                    "pending destroy: wait_for_fences(upload) failed: {e:?}"
                                );
                            }
                            device.free_command_buffers(self.command_pool, &[cmd_handle]);
                            device.destroy_fence(fence_handle, None);
                        }
                    }
                }
                false
            } else {
                true
            }
        });
    }

    /// Render one frame and present it. `camera` drives view-projection + culling.
    /// `ui` provides optional overlay vertices (crosshair, hotbar, pause menu).
    pub fn draw_frame(
        &mut self,
        camera: Camera,
        ui: Option<&UiDrawData>,
        game_time: f32,
        underwater: bool,
        world_entities: &[crate::entity::EntityRenderData],
        held_items: &[crate::entity::EntityRenderData],
        show_panorama: bool,
        panorama_rotation: f32,
    ) -> Result<()> {
        // Resize-check, upload UI, precompute camera matrices. Everything we need
        // before submitting any GPU work.
        // First: drain any GPU resources queued for deferred destruction from
        // previous `upload_chunks` / `remove_chunk` calls. By this point the
        // per-frame in-flight fences (only re-awaited just below) have
        // signalled, so it's safe to free their associated buffers.
        self.drain_pending_destructions();
        let (view_proj, vp_cols, frustum, ui_index_count) = self.prepare_frame(&camera, ui)?;

        let frame_idx = self.frame_counter % FRAMES_IN_FLIGHT;
        // Copy out the per-frame handles (all `Copy`) so we don't hold an
        // immutable borrow of `self.frames` across the mutable UBO update below.
        let (cmd, in_flight_fence, image_available, render_finished, descriptor_set, tile_remap_descriptor_set) = {
            let f = &self.frames[frame_idx];
            (
                f.cmd,
                f.in_flight_fence,
                f.image_available,
                f.render_finished,
                f.descriptor_set,
                f.tile_remap_descriptor_set,
            )
        };

        // Wait for this frame's previous use to finish.
        self.wait_for_fence_reset(in_flight_fence)?;

        // Write UBO data now that the previous frame is done with this slot.
        self.flush_pending_ubos();

        // Read back occlusion query results from 2 frames ago.
        self.readback_occlusion_results(frame_idx);

        // Read back the previous frame's GPU timestamps (now available after fence wait).
        // On the first 1-2 frames the queries haven't been written yet, so we
        // tolerate VK_NOT_READY and just skip the update.
        {
            // Read frame N-1's GPU timestamps (i.e. the slot that was
            // **just written** by the previous frame's cmd buffer). The
            // pattern is: `frame_idx` is the slot THIS frame will write;
            // the previous frame's data is at `(frame_idx + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT`.
            // Reading from this frame's own slot would trip the
            // `get_query_pool_results(WAIT)` infinite wait, because the GPU
            // hasn't written those queries yet. (Match the
            // `readback_occlusion_results` pattern at line 3384.)
            //
            // Skip the readback on the very first frame: there is no prior
            // frame whose timestamps to read, and querying uninitialised
            // pool slots trips VUID-vkGetQueryPoolResults-None-09401 even
            // when the pool was host-reset at startup.
            let prev_offset = (((frame_idx + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT) as u32) * GPU_TIMESTAMP_COUNT;
            if frame_idx > 0 {
                let mut timestamps = [0u64; GPU_TIMESTAMP_COUNT as usize];
            // WAIT flag: blocks the CPU until the GPU has finished processing
            // the previous frame's cmd_reset_query_pool + timestamps. Without
            // it, frame N's readback can race the GPU and trip
            // VUID-vkGetQueryPoolResults-None-09401 ("query not reset") on
            // the first query of the batch.
            let read_ok = unsafe {
                self.device.get_query_pool_results(
                    self.query_pool,
                    prev_offset,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            };
            if let Ok(()) = read_ok {
                let ns_to_ms = self.timestamp_period / 1_000_000.0;
                let t = &timestamps;
                self.timings = GpuTimings {
                    shadow_ms: (t[1] - t[0]) as f32 * ns_to_ms,
                    sky_ms: (t[2] - t[1]) as f32 * ns_to_ms,
                    opaque_ms: (t[3] - t[2]) as f32 * ns_to_ms,
                    transparent_ms: (t[4] - t[3]) as f32 * ns_to_ms,
                    ui_ms: (t[5] - t[4]) as f32 * ns_to_ms,
                    post_ms: (t[7] - t[6]) as f32 * ns_to_ms,
                    frame_ms: (t[7] - t[0]) as f32 * ns_to_ms,
                };
            }
            // On error (queries not yet available), keep previous timings.
            }
        }

        // Acquire the next swapchain image. NOT_READY / OUT_OF_DATE / SUBOPTIMAL
        // are non-fatal: skip this frame and retry next time.
        let acquire_result = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        };
        let (image_index, _suboptimal) = match acquire_result {
            Ok(pair) => pair,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.needs_resize = true;
                return Ok(());
            }
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                // No image available right now; try again next frame.
                return Ok(());
            }
            Err(e) => return Err(anyhow!("acquire_next_image: {e:?}")),
        };

        // Update camera UBO.
        self.update_camera_ubo(&camera, underwater, frame_idx)?;

        let device = &self.device;

        // Record command buffer (cmd was copied out above).
        // Wrap query_offset by the pool size so long-running sessions
        // (frame_counter > FRAMES_IN_FLIGHT) don't write past the pool
        // capacity. The pool is sized for `FRAMES_IN_FLIGHT` slots; older
        // timestamps get overwritten by the wrap, which is fine for a
        // running frame profiler.
        let query_pool_size = GPU_TIMESTAMP_COUNT * FRAMES_IN_FLIGHT as u32;
        let query_offset = ((frame_idx as u32) * GPU_TIMESTAMP_COUNT) % query_pool_size;
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // Reset query pool for this frame's slice and write frame start timestamp.
            device.cmd_reset_query_pool(cmd, self.query_pool, query_offset, GPU_TIMESTAMP_COUNT);
            // Reset occlusion query pool for this frame.
            self.reset_occlusion_queries(device, cmd, frame_idx);
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                query_offset, // timestamp 0: frame start
            );
        }

        // â”€â”€ Shadow pass: render chunk depth from the light's perspective â”€â”€
        self.record_shadow_pass(device, cmd, Some(query_offset + 1));

        self.record_main_pass_setup(device, cmd, image_index, descriptor_set, tile_remap_descriptor_set);

        // Sky or panorama pass: draw background before chunks.
        if show_panorama && self.panorama.loaded {
            self.record_panorama_pass(
                device,
                cmd,
                view_proj,
                camera.pos,
                panorama_rotation,
                Some(query_offset + 2),
            );
        } else {
            self.record_sky_pass(
                device,
                cmd,
                view_proj,
                camera.pos,
                descriptor_set,
                Some(query_offset + 2),
            );
        }



        // Chunk passes: Phase-1 GPU-driven indirect path or legacy per-chunk loop.
        if let Some(gpu) = self.gpu_driven.as_mut() {
            gpu.record(
                &self.device,
                cmd,
                descriptor_set,
                &vp_cols,
                game_time,
                camera.pos,
                self.query_pool,
                Some(query_offset + 3),
                Some(query_offset + 4),
            );
        } else {
            self.record_chunk_passes(
                device,
                cmd,
                &frustum,
                &vp_cols,
                game_time,
                Some(query_offset + 3),
                Some(query_offset + 4),
                camera.pos,
            );
        }

        // Entity pass: upload quad vertices then draw entity meshes.
        if !world_entities.is_empty() {
            let quad_verts = crate::entity::unit_quad_vertices();
            let vert_bytes = bytemuck::cast_slice(&quad_verts);
            if let Ok(slice) = self.entity_vbo.mapped_slice_mut() {
                let len = vert_bytes.len().min(slice.len());
                slice[..len].copy_from_slice(&vert_bytes[..len]);
            }
            if let Err(e) = self.entity_vbo.flush_whole(&self.device) {
                log::warn!("entity_vbo flush failed: {e}");
            }
            self.record_entity_pass(device, cmd, world_entities, view_proj);
        }

        // Held item pass: ALWAYS depth compare, rendered after world entities.
        if !held_items.is_empty() {
            self.record_held_item_pass(device, cmd, held_items, view_proj);
        }

        // Overlay pass: wireframe lines (brush preview, selection boxes).
        if !self.overlay_data.lines.is_empty() {
            let overlay_verts = self.overlay_data.to_vertices();
            let vert_bytes = bytemuck::cast_slice(&overlay_verts);
            if let Ok(slice) = self.overlay_vbo.mapped_slice_mut() {
                let len = vert_bytes.len().min(slice.len());
                slice[..len].copy_from_slice(&vert_bytes[..len]);
            }
            if let Err(e) = self.overlay_vbo.flush_whole(&self.device) {
                log::warn!("overlay_vbo flush failed: {e}");
            }
            self.record_overlay_pass(device, cmd, view_proj);
        }

        // UI overlay pass
        if ui_index_count > 0 {
            self.record_ui(device, cmd, ui_index_count);
        }

        unsafe {
            // Timestamp 5: UI end.
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                self.query_pool,
                query_offset + 5,
            );
            // Transition to subpass 1 (particles) so the chunks' depth
            // attachment is read as input (READ_ONLY_OPTIMAL) and the
            // particles composite onto the colour attachment above the
            // scene UI.
            device.cmd_next_subpass(cmd, vk::SubpassContents::INLINE);
        }
        // Subpass 1: particles. Emits a draw only when there's at least one
        // live particle; the pipeline's depth test occludes them where the
        // scene geometry blocks them.
        self.record_particle_subpass(
            device,
            cmd,
            descriptor_set,
            view_proj,
            camera.near,
            camera.far,
            frame_idx,
        );

        unsafe {
            device.cmd_end_render_pass(cmd);
        }

        // Slice 2 (draw_frame): scene_opaque_color copy + transparent pass.
        self.record_scene_opaque_copy(device, cmd, image_index);
        self.record_transparent_pass(
            device,
            cmd,
            image_index,
            &frustum,
            &vp_cols,
            game_time,
            descriptor_set,
            tile_remap_descriptor_set,
            camera.pos,
        );

        // Timestamp 6: main pass end.
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                query_offset + 6,
            );
        }

        // â”€â”€ Post-processing pass: sample offscreen â†’ swapchain â”€â”€

        // Cluster B fix: depth -> SHADER_READ_ONLY transition before the
        // post pass. The post pass binds `depth_tex` at set 0 binding 1
        // unconditionally (the binding exists for SSAO + depth-of-field +
        // any future consumer, so it must always be in a shader-readable
        // layout). Previously this barrier was gated on
        // `ssao_params[3] > 0.5`, which skipped the transition when SSAO
        // was disabled and tripped VUID-00344 on the post pass's
        // `vkCmdDraw` ("doesn't match the previous known layout
        // DEPTH_STENCIL_ATTACHMENT_OPTIMAL").
        if let Some(ref depth_img) = self.depth {
            crate::texture::transition_image_layout(
                device, cmd, depth_img.image,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                1, 1,
            );
        }

                self.record_post_pass(device, cmd, image_index, Some(query_offset + 7));
        unsafe {
            device.end_command_buffer(cmd)?;
        }

        // Submit.
        self.record_submit(
            device,
            cmd,
            &[image_available],
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[render_finished],
            in_flight_fence,
        )?;

        // Present.
        self.record_present(image_index, &[render_finished], true)?;

        self.frame_counter += 1;
        Ok(())
    }

    /// Render one frame without presenting and read the colour attachment back
    /// as RGBA8 pixels (used by the engine to save a verification screenshot).
    pub fn capture_frame(
        &mut self,
        camera: Camera,
        ui: Option<&UiDrawData>,
        game_time: f32,
        underwater: bool,
        world_entities: &[crate::entity::EntityRenderData],
        held_items: &[crate::entity::EntityRenderData],
        show_panorama: bool,
        panorama_rotation: f32,
    ) -> Result<Vec<u8>> {
        // Resize-check, upload UI, precompute camera matrices.
        let (view_proj, vp_cols, frustum, ui_index_count) = self.prepare_frame(&camera, ui)?;

        // Wait for frame 0's previous GPU work to complete before we
        // write to its UBO and descriptor set.
        self.wait_for_fence_reset(self.frames[0].in_flight_fence)?;

        // Write UBO data now that the previous frame is done.
        self.flush_pending_ubos();

        // Read back occlusion query results (uses frame 0 since capture uses frame 0).
        self.readback_occlusion_results(0);

        let device = &self.device;
        let cmd = begin_one_time(device, self.command_pool)?;

        // Acquire an image. A fence signals when the acquisition is safe; we
        // destroy it as soon as the acquire completes. The submit fence (used
        // later to wait for the readback command buffer) is created just before
        // queue_submit so it only needs cleanup on a narrow window of errors.
        let acquire_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let (image_index, _) = match unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                acquire_fence,
            )
        } {
            Ok(pair) => pair,
            Err(e) => {
                unsafe { device.destroy_fence(acquire_fence, None) };
                return Err(anyhow!("capture acquire: {e:?}"));
            }
        };
        // Wait for the acquisition to complete before using the image.
        unsafe {
            if let Err(e) = device.wait_for_fences(&[acquire_fence], true, u64::MAX) {
                device.destroy_fence(acquire_fence, None);
                return Err(anyhow!("capture wait_for_fences: {e:?}"));
            }
            device.destroy_fence(acquire_fence, None);
        }

        // Update camera UBO (frame 0).
        self.update_camera_ubo(&camera, underwater, 0)?;
        let device = &self.device;

        // â”€â”€ Shadow pass (capture) â”€â”€
        self.record_shadow_pass(device, cmd, None);

        self.record_main_pass_setup(device, cmd, image_index, self.frames[0].descriptor_set, self.frames[0].tile_remap_descriptor_set);

        // Sky or panorama pass (capture).
        if show_panorama && self.panorama.loaded {
            self.record_panorama_pass(device, cmd, view_proj, camera.pos, panorama_rotation, None);
        } else {
            self.record_sky_pass(
                device,
                cmd,
                view_proj,
                camera.pos,
                self.frames[0].descriptor_set,
                None,
            );
        }

        // Chunk passes: Phase-1 GPU-driven indirect path or legacy per-chunk loop.
        let chunk_ds = self.frames[0].descriptor_set;
        if let Some(gpu) = self.gpu_driven.as_mut() {
            gpu.record(&self.device, cmd, chunk_ds, &vp_cols, game_time, camera.pos, self.query_pool, None, None);
        } else {
            self.record_chunk_passes(&self.device, cmd, &frustum, &vp_cols, game_time, None, None, camera.pos);
        }

        // â”€â”€ UI overlay pass â”€â”€
        if ui_index_count > 0 {
            self.record_ui(device, cmd, ui_index_count);
        }

        unsafe {
            device.cmd_end_render_pass(cmd);
        }

        // Slice 2 (capture_frame): scene_opaque color+depth copies +
        // transparent pass. Runs BEFORE the post pass (matching draw_frame)
        // so captured screenshots actually contain water/glass; previously
        // this ran after post and captures silently dropped all translucent
        // geometry.
        self.record_scene_opaque_copy(device, cmd, image_index);
        self.record_transparent_pass(
            device,
            cmd,
            image_index,
            &frustum,
            &vp_cols,
            game_time,
            self.frames[0].descriptor_set,
            self.frames[0].tile_remap_descriptor_set,
            camera.pos,
        );

        // SSAO depth barrier for capture.
        if self.ssao_params[3] > 0.5 {
            if let Some(ref depth_img) = self.depth {
                crate::texture::transition_image_layout(
                    device,
                    cmd,
                    depth_img.image,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageAspectFlags::DEPTH,
                    1,
                    1,
                );
            }
        }

        // â”€â”€ Post pass (capture) â”€â”€
        self.record_post_pass(device, cmd, image_index, None);

        // Transition the swapchain image from PRESENT_SRC (the post pass's
        // final layout) to TRANSFER_SRC so we can copy it back to the host.
        transition_image_layout(
            device,
            cmd,
            self.swapchain_images[image_index as usize],
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );

        self.record_entity_pass(device, cmd, world_entities, view_proj);
        self.record_held_item_pass(device, cmd, held_items, view_proj);

        // Overlay pass in capture.
        if !self.overlay_data.lines.is_empty() {
            let overlay_verts = self.overlay_data.to_vertices();
            let vert_bytes = bytemuck::cast_slice(&overlay_verts);
            if let Ok(slice) = self.overlay_vbo.mapped_slice_mut() {
                let len = vert_bytes.len().min(slice.len());
                slice[..len].copy_from_slice(&vert_bytes[..len]);
            }
            if let Err(e) = self.overlay_vbo.flush_whole(&self.device) {
                log::warn!("overlay_vbo flush failed: {e}");
            }
            self.record_overlay_pass(device, cmd, view_proj);
        }

        let extent = self.swapchain_extent;
        let pixel_count = (extent.width * extent.height) as usize;
        let row_pitch = extent.width * 4; // RGBA8
        let buf_size = (row_pitch * extent.height) as vk::DeviceSize;
        let mut readback = GpuBuffer::host_visible(
            device,
            &self.alloc,
            buf_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            "capture_readback",
        )?;

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(extent.width)
            .buffer_image_height(extent.height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                self.swapchain_images[image_index as usize],
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &[region],
            );
            device.end_command_buffer(cmd)?;
        }

        let submit_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| anyhow!("capture submit_fence: {e:?}"))?;
        self.record_submit(device, cmd, &[], &[], &[], submit_fence)?;
        unsafe {
            if let Err(e) = device.wait_for_fences(&[submit_fence], true, u64::MAX) {
                device.destroy_fence(submit_fence, None);
                return Err(anyhow!("capture wait_for_fences (submit): {e:?}"));
            }
            device.destroy_fence(submit_fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
        }

        let slice = readback.mapped_slice_mut()?;
        let mut out = vec![0u8; pixel_count * 4];
        out.copy_from_slice(&slice[..pixel_count * 4]);
        readback.destroy(device, &self.alloc);

        // Transition the image back to PRESENT_SRC and present it so the
        // swapchain image is returned to the pool (otherwise repeated captures
        // leak images until acquire_next_image deadlocks/crashes).
        let present_cmd = begin_one_time(device, self.command_pool)?;
        transition_image_layout(
            device,
            present_cmd,
            self.swapchain_images[image_index as usize],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );
        unsafe {
            device.end_command_buffer(present_cmd)?;
        }

        let present_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let command_buffers = [present_cmd];
        let signal_semaphores = [present_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let submit_infos = [submit_info];
        unsafe {
            device.queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())?;
        }

        // `queue_submit` was the last OLD `device` use; the OLD `&self.device`
        // borrow ends here. `record_present` takes `&mut self` next.
        self.record_present(image_index, &signal_semaphores, false)?;

        // Fresh borrow for the post-present cleanup (record_present held
        // &mut self, so we need an immutable re-acquire here).
        let device = &self.device;

        unsafe {
            device.device_wait_idle()?;
            device.destroy_semaphore(present_semaphore, None);
            device.free_command_buffers(self.command_pool, &[present_cmd]);
        }

        // The swapchain is typically B8G8R8A8 (sRGB or unorm); the readback is
        // in that channel order. Convert to RGBA8 so callers can save it directly.
        let is_bgra = matches!(
            self.swapchain_format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
        );
        if is_bgra {
            for px in out.chunks_exact_mut(4) {
                px.swap(0, 2); // B <-> R
            }
        }

        // Keep frame_counter in sync so the next draw_frame picks the right slot.
        self.frame_counter += 1;

        Ok(out)
    }

    /// Record both chunk draw passes (opaque then transparent) into `cmd`.
    ///
    /// Both `draw_frame` and `capture_frame` need identical chunk rendering;
    /// this is the shared implementation. Uses the **collect-then-drop** lock
    /// pattern: acquire `self.chunks.read()` once to build the list of
    /// visible chunks, release it before recording draws. This keeps chunk
    /// uploads (`upload_chunks` taking `self.chunks.write()`) from blocking
    /// for the duration of push-constant filling + draw command recording.
    ///
    /// `opaque_end_timestamp_query` and `transparent_end_timestamp_query`
    /// are `Some(query_offset + 3)` and `Some(query_offset + 4)` from
    /// `draw_frame` (GPU profiling); `capture_frame` passes `None` for both
    /// since it doesn't run the timestamp pool.
    ///
    /// `vp_cols` is the 16-float view-projection matrix in column-major order.

    /// Read back occlusion query results from the previous frame and update
    /// per-chunk visibility state. Must be called with `&mut self` (in
    /// `draw_frame` / `capture_frame`) after the fence wait.
    fn readback_occlusion_results(&mut self, current_frame_idx: usize) {
        if !self.occlusion_culling_enabled {
            return;
        }
        // The OTHER frame slot was last written 2 frames ago (fence already
        // signalled). Read its results.
        let read_idx = (current_frame_idx + 1) % FRAMES_IN_FLIGHT;
        let frames = self.occlusion_frames.write();
        let frame = &frames[read_idx];
        let queries = &frame.used_queries;
        if queries.is_empty() {
            return;
        }

        // Allocate a results buffer.
        let mut results = vec![0u32; queries.len()];
        let mut all_ok = true;
        for (ri, &qi) in queries.iter().enumerate() {
            let read = unsafe {
                self.device.get_query_pool_results(
                    frame.query_pool,
                    qi,
                    &mut results[ri..ri + 1],
                    vk::QueryResultFlags::WAIT,
                )
            };
            if read.is_err() {
                all_ok = false;
                break;
            }
        }
        if !all_ok {
            return;
        }

        // Update visibility state.
        let mut state = self.occlusion_state.write();
        for (i, &query_idx) in queries.iter().enumerate() {
            // Find the chunk that owns this query index.
            for (_pos, oc) in state.iter_mut() {
                if oc.query_index == query_idx {
                    if results[i] > 0 {
                        oc.was_visible = true;
                        oc.consecutive_invisible = 0;
                    } else {
                        oc.was_visible = false;
                        oc.consecutive_invisible += 1;
                    }
                    break;
                }
            }
        }
    }

    /// Reset the occlusion query pool for the current frame and clear the
    /// used_queries list. Called at the start of command recording.
    fn reset_occlusion_queries(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_idx: usize) {
        if !self.occlusion_culling_enabled {
            return;
        }
        let mut frames = self.occlusion_frames.write();
        let frame = &mut frames[frame_idx];
        // Reset ALL queries in the pool (not just the previously-used ones)
        // so the very first frame after init has every slot in the
        // "unavailable" state required by VUID-vkCmdBeginQuery-None-00807.
        // The min/max optimization only worked on frame N>=2; frame 0
        // had an empty used_queries list and skipped the reset entirely.
        unsafe {
            device.cmd_reset_query_pool(cmd, frame.query_pool, 0, MAX_OCCLUSION_QUERIES);
        }
        frame.used_queries.clear();
    }

    /// Allocate a free occlusion query index for a chunk. Returns None if the
    /// pool is exhausted.
    fn record_chunk_passes(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frustum: &Frustum,
        vp_cols: &[f32],
        game_time: f32,
        opaque_end_timestamp_query: Option<u32>,
        transparent_end_timestamp_query: Option<u32>,
        cam_pos: glam::Vec3,
    ) {
        // Collect visible chunk buffer handles under the read lock, then
        // drop the lock so upload_chunks can proceed while we record.
        let (mut opaque_draws, transparent_draws): (
            Vec<(ChunkPos, vk::Buffer, vk::Buffer, u32)>,
            Vec<(ChunkPos, vk::Buffer, vk::Buffer, u32)>,
        ) = {
            let chunks = self.chunks.read();
            let mut opaque = Vec::new();
            let mut transparent = Vec::new();
            for (&pos, bufs) in chunks.iter() {
                let origin = chunk_origin(pos);
                let min = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
                let max = min + Vec3::splat(voxel_core::CHUNK_SIZE as f32);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                if let Some(b) = &bufs.opaque {
                    opaque.push((pos, b.vbo.buffer, b.ibo.buffer, b.index_count));
                }
                if let Some(b) = &bufs.transparent {
                    transparent.push((pos, b.vbo.buffer, b.ibo.buffer, b.index_count));
                }
            }
            (opaque, transparent)
        };

        // Sort opaque chunks front-to-back for better early-Z rejection and
        // occlusion query efficiency.
        opaque_draws.sort_by(|a, b| {
            let origin_a = chunk_origin(a.0);
            let origin_b = chunk_origin(b.0);
            let center_a = Vec3::new(
                origin_a.x as f32 + 8.0,
                origin_a.y as f32 + 8.0,
                origin_a.z as f32 + 8.0,
            );
            let center_b = Vec3::new(
                origin_b.x as f32 + 8.0,
                origin_b.y as f32 + 8.0,
                origin_b.z as f32 + 8.0,
            );
            let dist_a = (center_a - cam_pos).length_squared();
            let dist_b = (center_b - cam_pos).length_squared();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let occlusion_enabled = self.occlusion_culling_enabled;

        // Pre-allocate persistent occlusion query indices for new chunks.
        // This ensures each chunk has a stable query_index that persists
        // across frames, which is required for correct readback mapping.
        if occlusion_enabled {
            let mut occ = self.occlusion_state.write();
            let used: std::collections::HashSet<u32> = occ.values().map(|s| s.query_index).collect();
            let mut next_idx = 0u32;
            for (pos, _, _, _) in &opaque_draws {
                if !occ.contains_key(pos) {
                    while next_idx < MAX_OCCLUSION_QUERIES && used.contains(&next_idx) {
                        next_idx += 1;
                    }
                    if next_idx < MAX_OCCLUSION_QUERIES {
                        occ.insert(*pos, OcclusionState {
                            query_index: next_idx,
                            was_visible: true,
                            consecutive_invisible: 0,
                        });
                        next_idx += 1;
                    }
                }
            }
        }


        let issue_draw =
            |cmd: vk::CommandBuffer,
             pos: ChunkPos,
             vbo_buf: vk::Buffer,
             ibo_buf: vk::Buffer,
             index_count: u32| {
                let origin = chunk_origin(pos);
                let mut push = [0f32; 24];
                push[0] = origin.x as f32;
                push[1] = origin.y as f32;
                push[2] = origin.z as f32;
                push[4..20].copy_from_slice(vp_cols);
                push[20] = game_time;
                unsafe {
                    device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        // `self.pipeline_layout` declares the push-constant range
                        // as VERTEX|FRAGMENT (96 B, layout.rs `create_pipeline_layout`),
                        // so the call site must echo both stages or validation
                        // reports VUID-VkCmdPushConstants-offset-01796 ("missing
                        // stageFlags from the overlapping VkPushConstantRange").
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    let vbo = [vbo_buf];
                    device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                    device.cmd_bind_index_buffer(cmd, ibo_buf, 0, vk::IndexType::UINT32);
                    device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
                }
            };

        // ── Opaque pass ──
        if occlusion_enabled {
            let occ_state = self.occlusion_state.read();
            let mut frames = self.occlusion_frames.write();
            let frame_idx = self.frame_counter % FRAMES_IN_FLIGHT;
            let frame = &mut frames[frame_idx];
            let query_pool = frame.query_pool;

            for &(pos, vbo_buf, ibo_buf, index_count) in &opaque_draws {
                let state = occ_state.get(&pos);
                let should_draw_mesh = match state {
                    None => true, // New chunk, no query yet -> draw full mesh
                    Some(s) => {
                        s.was_visible
                            || s.consecutive_invisible < OCCLUSION_INVISIBLE_THRESHOLD
                    }
                };

                if should_draw_mesh {
                    // Draw full mesh wrapped in an occlusion query.
                    let qi = match occ_state.get(&pos) {
                        Some(s) if s.query_index < MAX_OCCLUSION_QUERIES => s.query_index,
                        _ => continue, // skip if no valid query index
                    };
                    if qi < MAX_OCCLUSION_QUERIES {
                        frame.used_queries.push(qi);
                        unsafe {
                            device.cmd_begin_query(
                                cmd,
                                query_pool,
                                qi,
                                vk::QueryControlFlags::empty(),
                            );
                        }
                        issue_draw(cmd, pos, vbo_buf, ibo_buf, index_count);
                        unsafe {
                            device.cmd_end_query(cmd, query_pool, qi);
                        }
                    } else {
                        // Query pool exhausted; draw without query.
                        issue_draw(cmd, pos, vbo_buf, ibo_buf, index_count);
                    }
                } else {
                    // Chunk was invisible for too long — draw AABB proxy only.
                    // Use a cheap occlusion query to re-check visibility.
                    let qi = match occ_state.get(&pos) {
                        Some(s) if s.query_index < MAX_OCCLUSION_QUERIES => s.query_index,
                        _ => continue, // skip if no valid query index
                    };
                    if qi < MAX_OCCLUSION_QUERIES {
                        frame.used_queries.push(qi);
                        let origin = chunk_origin(pos);
                        let min = Vec3::new(
                            origin.x as f32,
                            origin.y as f32,
                            origin.z as f32,
                        );
                        let max = min + Vec3::splat(voxel_core::CHUNK_SIZE as f32);
                        // Push constants: min.xyz in first vec4, VP matrix, max.xyz in last vec4.
                        let mut push = [0f32; 24];
                        push[0] = min.x;
                        push[1] = min.y;
                        push[2] = min.z;
                        push[3] = 0.0;
                        push[4..20].copy_from_slice(vp_cols);
                        push[20] = max.x;
                        push[21] = max.y;
                        push[22] = max.z;
                        push[23] = 0.0;
                        unsafe {
                            device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.occlusion_pipeline,
                            );
                            device.cmd_push_constants(
                                cmd,
                                self.pipeline_layout,
                                // chunk material layout declares VERTEX|FRAGMENT (96 B);
                                // see VUID-VkCmdPushConstants-offset-01796 above.
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&push),
                            );
                            device.cmd_bind_index_buffer(
                                cmd,
                                self.aabb_index_buffer.buffer,
                                0,
                                vk::IndexType::UINT16,
                            );
                            device.cmd_begin_query(
                                cmd,
                                query_pool,
                                qi,
                                vk::QueryControlFlags::empty(),
                            );
                            device.cmd_draw_indexed(cmd, 36, 1, 0, 0, 0);
                            device.cmd_end_query(cmd, query_pool, qi);
                            // Restore the main chunk pipeline.
                            device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.active_pipeline(),
                            );
                        }
                    }
                    // If query pool exhausted, silently skip this chunk.
                }
            }
            drop(occ_state);
            drop(frames);
        } else {
            for &(pos, vbo_buf, ibo_buf, index_count) in &opaque_draws {
                issue_draw(cmd, pos, vbo_buf, ibo_buf, index_count);
            }
        }

        unsafe {
            if let Some(q) = opaque_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }
            // NOTE: transparent chunks are NOT drawn in this (main) pass
            // anymore. They render exclusively in the slice-2 transparent
            // pass (`record_transparent_pass`), which runs after the
            // scene_opaque color+depth copies so water/glass can refract and
            // reflect the CURRENT frame's opaque scene. The old code drew
            // them here too, which double-blended them AND depth-occluded the
            // slice-2 draw (same depth, LESS compare → slice-2 fully hidden).
            // The transparent_end timestamp is still written here so the
            // t[3]..t[5] ordering the GpuTimings subtraction expects stays
            // monotonic (main-pass transparent time is now ~0 by design).
            if let Some(q) = transparent_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }
        }
        let _ = transparent_draws; // collected but unused; see note above.
    }

    /// Record the entity pass: draw all entity meshes (billboards/cubes).
    fn record_entity_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        entities: &[crate::entity::EntityRenderData],
        view_proj: glam::Mat4,
    ) {
        if entities.is_empty() {
            return;
        }

        // Bind entity pipeline and VBO.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.entity_pipeline,
            );
            let vbo = [self.entity_vbo.buffer];
            device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
        }

        // Draw each entity as a quad.
        for entity in entities {
            let translate = glam::Mat4::from_translation(entity.pos);
            let rotation = glam::Mat4::from_quat(entity.rot);
            let entity_model = translate * rotation;
            let mvp = view_proj * entity_model;
            let mvp_cols = mvp.to_cols_array_2d();

            let billboard: u32 = if entity.billboard { 1 } else { 0 };
            let push = crate::entity::EntityPushConstants {
                model: mvp_cols,
                tile: entity.tile,
                half_size: entity.half_size,
                billboard,
                _pad: [0; 2],
            };
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    self.entity_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_draw(cmd, 6, 1, 0, 0);
            }
        }
    }

    /// Record the particle subpass (subpass 1). Reads the depth-written by
    /// subpass 0 (chunks + entities + held items + overlay + UI), enabling
    /// hard-edged scene occlusion via the fixed-function depth test in
    /// subpass 1's pipeline. Phase 2 will swap the depth attachment read
    /// mode to an input-attachment read so the fragment shader can compute
    /// a soft fade as the particle approaches geometry.
    ///
    /// `view_proj` is the current frame's view-projection matrix; we push
    /// its inverse so the vertex shader can billboard quad vertices
    /// against the camera (right/up basis derived from the upper-3x3 of
    /// inverse(view_proj)) and project them back to clip space.
    fn record_particle_subpass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        chunk_descriptor_set: vk::DescriptorSet,
        view_proj: glam::Mat4,
        camera_near: f32,
        camera_far: f32,
        frame_idx: usize,
    ) {
        let instances = self.particle_manager.instances();
        if instances.is_empty() {
            return;
        }

        // Inverse VP for vertex shader billboarding. Determinant guard
        // mirrors the sky shader's behaviour (avoid a NaN matrix on rare
        // degenerate matrices).
        let inv_vp = if view_proj.determinant().abs() > 1e-10 {
            view_proj.inverse()
        } else {
            glam::Mat4::IDENTITY
        };
        // mat4 (64 B) + vec4 soft_near_far (16 B) = 80 B; chunk range is 96 B
        // so this fits without changing the layout.
        let mut push = [0f32; 24];
        push[..16].copy_from_slice(&inv_vp.to_cols_array());
        push[16] = self.particle_softness;
        push[17] = camera_near;
        push[18] = camera_far;
        push[19] = 0.0;

        let pipeline_layout = self.particle_pipeline_layout;
        let depth_set = match self.particle_depth_descriptor_sets.get(frame_idx % FRAMES_IN_FLIGHT) {
            Some(&set) => set,
            None => return, // Descriptor sets not ready yet, skip particle rendering.
        };
        let sets = [chunk_descriptor_set, depth_set];
        let vbos = [self.entity_vbo.buffer, self.particle_instance_vbo.buffer];
        let offsets = [0u64, 0u64];
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.particle_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &vbos, &offsets);
            // 6 vertices per quad, N instances for N particles.
            device.cmd_draw(cmd, 6, instances.len() as u32, 0, 0);
        }
    }

    /// Update CPU-side particle simulation and upload the per-instance
    /// vertex stream. Called once per frame, before any cmd recording.
    /// Clamps the upload to [`crate::particle::MAX_PARTICLES`] so the
    /// HOST_VISIBLE VBO is never overrun.
    pub fn update_particles(&mut self, dt: f32) {
        self.particle_manager.update(dt);
        // Snap physics → GPU layout.
        let instances = self.particle_manager.instances();
        let bytes = bytemuck::cast_slice(&instances);
        if let Ok(slice) = self.particle_instance_vbo.mapped_slice_mut() {
            // `slice.len()` is exactly MAX_PARTICLES * stride; the
            // snapshot can be shorter if fewer particles are alive, we
            // only copy what we have. Anything we don't update is
            // stale but irrelevant because `instances.len()` controls
            // the draw count.
            let len = bytes.len().min(slice.len());
            slice[..len].copy_from_slice(&bytes[..len]);
        }
        if let Err(e) = self.particle_instance_vbo.flush_whole(&self.device) {
            log::warn!("particle_instance_vbo flush failed: {e}");
        }
    }

    /// Spawn break particles at a block breakage site. Called by the
    /// engine on block break; the renderer owns the simulation queue.
    pub fn spawn_particles_break(&mut self, pos: glam::Vec3, color: [u8; 4], normal: glam::Vec3) {
        self.particle_manager.emit_break(pos, color, normal);
    }

    /// Spawn place particles at a block placement site.
    pub fn spawn_particles_place(&mut self, pos: glam::Vec3, color: [u8; 4]) {
        self.particle_manager.emit_place(pos, color);
    }

    /// Number of live particles (mostly for telemetry / debug).
    pub fn particle_count(&self) -> usize {
        self.particle_manager.len()
    }

    /// Record the held item pass: draw held items with ALWAYS depth compare.
    fn record_held_item_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        entities: &[crate::entity::EntityRenderData],
        view_proj: glam::Mat4,
    ) {
        if entities.is_empty() {
            return;
        }

        // Bind held item pipeline (ALWAYS depth compare).
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.entity_held_pipeline,
            );
            let vbo = [self.entity_vbo.buffer];
            device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
        }

        for entity in entities {
            let translate = glam::Mat4::from_translation(entity.pos);
            let rotation = glam::Mat4::from_quat(entity.rot);
            let entity_model = translate * rotation;
            let mvp = view_proj * entity_model;
            let mvp_cols = mvp.to_cols_array_2d();

            let billboard: u32 = if entity.billboard { 1 } else { 0 };
            let push = crate::entity::EntityPushConstants {
                model: mvp_cols,
                tile: entity.tile,
                half_size: entity.half_size,
                billboard,
                _pad: [0; 2],
            };
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    self.entity_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_draw(cmd, 6, 1, 0, 0);
            }
        }
    }

    fn record_shadow_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        shadow_end_timestamp_query: Option<u32>,
    ) {
        let shadow_extent = vk::Extent2D {
            width: 2048,
            height: 2048,
        };
        let shadow_vp = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(2048.0)
            .height(2048.0)
            .min_depth(0.0)
            .max_depth(1.0);
        let shadow_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: shadow_extent,
        };
        let chunks = self.chunks.read();
        for cascade in 0..4u32 {
            let clear_values = [vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }];
            let shadow_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.shadow_render_pass)
                .framebuffer(self.shadow_framebuffers[cascade as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: shadow_extent,
                })
                .clear_values(&clear_values);
            unsafe {
                device.cmd_begin_render_pass(cmd, &shadow_begin, vk::SubpassContents::INLINE);
                device.cmd_set_viewport(cmd, 0, &[shadow_vp]);
                device.cmd_set_scissor(cmd, 0, &[shadow_scissor]);
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.shadow_pipeline,
                );

                let vp = self.shadow_ubo_data.cascade_vps[cascade as usize];
                for (&pos, b) in chunks.iter() {
                    let Some(opaque) = &b.opaque else { continue };
                    let origin = chunk_origin(pos);
                    let mut push = [0f32; 20];
                    push[..16].copy_from_slice(&vp);
                    push[16] = origin.x as f32;
                    push[17] = origin.y as f32;
                    push[18] = origin.z as f32;
                    push[19] = 0.0;
                    device.cmd_push_constants(
                        cmd,
                        self.shadow_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::cast_slice(&push),
                    );
                    let vbo = [opaque.vbo.buffer];
                    device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                    device.cmd_bind_index_buffer(cmd, opaque.ibo.buffer, 0, vk::IndexType::UINT32);
                    device.cmd_draw_indexed(cmd, opaque.index_count, 1, 0, 0, 0);
                }
                device.cmd_end_render_pass(cmd);
            }
        }
        drop(chunks);

        if let Some(q) = shadow_end_timestamp_query {
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    q,
                );
            }
        }
    }

    /// Record the sky pass (full-screen gradient) into `cmd` and then
    /// re-bind the chunk graphics pipeline + descriptor set so the
    /// subsequent chunk draws see the right state.
    ///
    /// Both `draw_frame` and `capture_frame` need identical sky rendering;
    /// this is the shared implementation. If `sky_end_timestamp_query` is
    /// `Some`, a `cmd_write_timestamp` is emitted on `COLOR_ATTACHMENT_OUTPUT`
    /// after the sky draw (used by the GPU profiling path in `draw_frame`;
    /// capture skips it).
    ///
    /// `chunk_descriptor_set` is restored after the sky draw so the chunk
    /// pass binds to the right per-frame resources.
    fn record_sky_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        view_proj: Mat4,
        camera_pos: Vec3,
        chunk_descriptor_set: vk::DescriptorSet,
        sky_end_timestamp_query: Option<u32>,
    ) {
        let vp = vk::Viewport::default()
            .x(0.0)
            .y(self.swapchain_extent.height as f32)
            .width(self.swapchain_extent.width as f32)
            .height(-(self.swapchain_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };

        // Compute inverse view-projection for the sky shader.
        let inv_vp = if view_proj.determinant().abs() > 1e-10 {
            view_proj.inverse()
        } else {
            Mat4::IDENTITY
        };
        let inv_vp_cols = inv_vp.to_cols_array();
        // Pack inverse VP (64 bytes) + camera position (16 bytes) = 80 bytes.
        let mut sky_push_data = [0.0f32; 20];
        sky_push_data[..16].copy_from_slice(&inv_vp_cols);
        sky_push_data[16] = camera_pos.x;
        sky_push_data[17] = camera_pos.y;
        sky_push_data[18] = camera_pos.z;
        sky_push_data[19] = 0.0;

        let sky_desc_sets = [self.sky_descriptor_set];

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[vp]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.sky_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.sky_pipeline_layout,
                0,
                &sky_desc_sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.sky_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&sky_push_data),
            );
            // Draw 3 vertices = full-screen triangle (no vertex buffer).
            device.cmd_draw(cmd, 3, 1, 0, 0);
            if let Some(q) = sky_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }

            // Restore the chunk pipeline + descriptor set so chunk draws
            // see the right state. The viewport is already correct (same
            // negative-height viewport) so it doesn't need to be reset.
            // Bind BOTH descriptor sets 0 (chunk material UBO group) and 1
            // (tile_remap UBO) because the chunk material pipeline statically
            // references both — binding only set 0 trips
            // VUID-vkCmdDrawIndexed-pDescriptorSets-04616 ("set 1 out of bounds
            // for the number of sets bound").
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.active_pipeline());
            let tile_remap_ds = self.frames[self.frame_counter % FRAMES_IN_FLIGHT]
                .tile_remap_descriptor_set;
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[chunk_descriptor_set, tile_remap_ds],
                &[],
            );
        }
    }

    /// Record the panorama pass (cubemap background) into `cmd`.
    /// Used on the title screen when panorama textures are loaded.
    /// `rotation` is a yaw angle in radians applied to the view direction.
    fn record_panorama_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        view_proj: Mat4,
        camera_pos: Vec3,
        rotation: f32,
        end_timestamp_query: Option<u32>,
    ) {
        let vp = vk::Viewport::default()
            .x(0.0)
            .y(self.swapchain_extent.height as f32)
            .width(self.swapchain_extent.width as f32)
            .height(-(self.swapchain_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };

        // Build a rotated view-projection: apply yaw rotation to the view matrix.
        let rot_mat = Mat4::from_rotation_y(rotation);
        let rotated_vp = view_proj * rot_mat;
        let inv_vp = if rotated_vp.determinant().abs() > 1e-10 {
            rotated_vp.inverse()
        } else {
            Mat4::IDENTITY
        };
        let inv_vp_cols = inv_vp.to_cols_array();

        let mut push_data = [0.0f32; 20];
        push_data[..16].copy_from_slice(&inv_vp_cols);
        push_data[16] = camera_pos.x;
        push_data[17] = camera_pos.y;
        push_data[18] = camera_pos.z;
        push_data[19] = 0.0;

        let desc_sets = [self.panorama_descriptor_set];

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[vp]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.panorama_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.panorama_pipeline_layout,
                0,
                &desc_sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.panorama_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&push_data),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            if let Some(q) = end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }

            // Restore chunk pipeline.
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.active_pipeline(),
            );
            // Bind both chunk descriptor sets: set 0 (chunk material UBO group)
            // and set 1 (tile_remap UBO). The chunk material pipeline statically
            // references both layouts — binding only set 0 trips the
            // "set 1 out of bounds" validation.
            let cur_frame = self.frame_counter % FRAMES_IN_FLIGHT;
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[
                    self.frames[cur_frame].descriptor_set,
                    self.frames[cur_frame].tile_remap_descriptor_set,
                ],
                &[],
            );
        }
    }

    /// Record the post-processing pass into `cmd`. Both `draw_frame` and
    /// `capture_frame` need the same offscreen â†’ swapchain blit; this is
    /// the shared implementation.
    ///
    /// `post_end_timestamp_query` is `Some(query_offset + 7)` from
    /// `draw_frame` (GPU profiling). Capture frames pass `None` because
    /// they don't run the timing pool.
    fn record_post_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        post_end_timestamp_query: Option<u32>,
    ) {
        let post_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.post_render_pass)
            .framebuffer(self.post_framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&[]);
        let post_vp = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let post_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        unsafe {
            device.cmd_begin_render_pass(cmd, &post_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, &[post_vp]);
            device.cmd_set_scissor(cmd, 0, &[post_scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.post_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.post_pipeline_layout,
                0,
                &[self.post_descriptor_sets[image_index as usize]],
                &[],
            );
            let mut post_push = [0.0f32; 12];
            post_push[0..4].copy_from_slice(&self.post_params);
            post_push[4..8].copy_from_slice(&self.ssao_params);
            post_push[8..12].copy_from_slice(&self.proj_params);
            device.cmd_push_constants(
                cmd,
                self.post_pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::cast_slice(&post_push),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            if let Some(q) = post_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    q,
                );
            }
        }
    }

    /// Begin the offscreen render pass for `image_index`, bind the active chunk
    /// pipeline, and bind the chunk descriptor set. Both `draw_frame` and
    /// `capture_frame` call this with the appropriate descriptor set
    /// (`draw_frame` uses the per-frame set; `capture_frame` uses frame 0).
    /// The render area + clear values are derived from `self.swapchain_extent`
    /// and `self.config.clear_color` internally so callers don't have to pass
    /// the same literal block each time.
    fn record_main_pass_setup(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        descriptor_set: vk::DescriptorSet,
        tile_remap_descriptor_set: vk::DescriptorSet,
    ) {
        // Always 3 clear values. When MSAA is off the render pass has only
        // 2 attachments, so the 3rd entry is harmlessly ignored by Vulkan.
        let resolve_clear = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: self.config.clear_color,
                },
            }
        } else {
            vk::ClearValue::default()
        };
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: self.config.clear_color,
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            resolve_clear,
        ];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.offscreen_framebuffers[image_index as usize])
            .render_area(render_area)
            .clear_values(&clear_values);
        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.active_pipeline());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                // Two-set layout since cluster A wired in `tile_remap` at
                // set 1 binding 0: without binding both descriptor sets in a
                // single `cmd_bind_descriptor_sets` call the validation
                // layer fires VUID-08600 ("set (1) is out of bounds for the
                // number of sets bound (1)") on every chunk draw.
                &[descriptor_set, tile_remap_descriptor_set],
                &[],
            );
        }
    }

    /// Update `frames[frame_index].camera_ubo` with the current `camera` position
    /// and the underwater-aware fog distance. Required to be called after the
    /// frame's previous GPU work has finished (e.g. after `wait_for_fences`), so
    /// writes don't race GPU reads. Both `draw_frame` and `capture_frame` call
    /// this; only the `frame_index` differs (per-frame in draw, frame 0 in capture).
    fn update_camera_ubo(
        &mut self,
        camera: &Camera,
        underwater: bool,
        frame_index: usize,
    ) -> Result<()> {
        let mut cam_ubo = CameraUbo::default();
        let fog_dist = if underwater {
            self.config.fog_distance * 0.05 // Very close fog underwater for Minecraft-like look.
        } else {
            self.config.fog_distance
        };
        cam_ubo.cam_pos_and_maxdist = [camera.pos.x, camera.pos.y, camera.pos.z, fog_dist];
        let frame = &mut self.frames[frame_index];
        {
            let slice = frame.camera_ubo.mapped_slice_mut()?;
            let bytes: &[u8] = bytemuck::bytes_of(&cam_ubo);
            slice[..bytes.len()].copy_from_slice(bytes);
        }
        // Flush so the GPU sees the latest camera pos + fog distance.
        // Without this, non-coherent memory can serve stale camera data,
        // causing fog/lighting miscalculations visible as border artifacts.
        if let Err(e) = frame.camera_ubo.flush_whole(&self.device) {
            log::warn!("camera_ubo flush failed: {e}");
        }
        Ok(())
    }

    /// Common preamble for both `draw_frame` and `capture_frame`: handle a
    /// pending swapchain resize, upload any UI data, and precompute the
    /// camera matrices used by all `record_*_pass` helpers. Returns every
    /// derived value callers need before submitting any GPU work.
    ///
    /// **Note:** `view_proj`/`vp_cols`/`frustum` are computed and returned
    /// BEFORE `update_camera_ubo` is called later in each method. This is
    /// safe because `view_projection` is a pure function of `camera` and
    /// does not read `self` state.
    ///
    /// **Return-tuple order:** `(Mat4, [f32; 16], Frustum, u32)` â€”
    /// `draw_frame` and `capture_frame` destructure into
    /// `(view_proj, vp_cols, frustum, ui_index_count)`. Keep this order in
    /// sync with both call sites if you ever add a return field.
    fn prepare_frame(
        &mut self,
        camera: &Camera,
        ui: Option<&UiDrawData>,
    ) -> Result<(Mat4, [f32; 16], Frustum, u32)> {
        if self.needs_resize {
            self.recreate_swapchain()?;
            self.needs_resize = false;
        }
        let ui_index_count = ui.map(|u| self.upload_ui(u)).unwrap_or(0);
        let view_proj = camera.view_projection();
        let vp_cols = view_proj.to_cols_array(); // 16 floats, column-major
        let frustum = Frustum::from_view_projection(view_proj);
        Ok((view_proj, vp_cols, frustum, ui_index_count))
    }

    /// Wait for `fence` to signal (the previous frame's GPU work using this
    /// fence is done) and reset it back to unsignalled state. Called once per
    /// frame at the start of both `draw_frame` (uses the per-frame
    /// `in_flight_fence`) and `capture_frame` (uses `self.frames[0].in_flight_fence`).
    fn wait_for_fence_reset(&self, fence: vk::Fence) -> Result<()> {
        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence])?;
        }
        Ok(())
    }

    /// Build a `vk::SubmitInfo` from the given command buffer + wait/signal
    /// semaphores and submit it to `self.graphics_queue`. Used by both
    /// `draw_frame` and `capture_frame`. The caller owns `fence` (passing
    /// `vk::Fence::null()` is fine if no completion signal is needed).
    fn record_submit(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> Result<()> {
        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);
        let submit_infos = [submit_info];
        unsafe {
            device.queue_submit(self.graphics_queue, &submit_infos, fence)?;
        }
        Ok(())
    }

    /// Submit the present request for `image_index` and handle the result.
    /// If `set_resize_on_out_of_date` is true and the result is
    /// `ERROR_OUT_OF_DATE_KHR` or `SUBOPTIMAL_KHR`, `self.needs_resize` is
    /// set so the next frame triggers a swapchain recreate. If false (used by
    /// `capture_frame`), those errors are silently ignored.
    fn record_present(
        &mut self,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
        set_resize_on_out_of_date: bool,
    ) -> Result<()> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let result = unsafe {
            self.swapchain_device
                .queue_present(self.present_queue, &present_info)
        };
        match result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR)
                if set_resize_on_out_of_date =>
            {
                self.needs_resize = true;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                // capture_frame: silently ignore OUT_OF_DATE/SUBOPTIMAL.
            }
            Err(e) => return Err(anyhow!("queue_present: {e:?}")),
        }
        Ok(())
    }

    /// Recreate the swapchain + dependent resources (called on resize).
    fn recreate_swapchain(&mut self) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;
        }
        // Clean up old swapchain-dependent objects.
        for &fb in &self.offscreen_framebuffers {
            unsafe { self.device.destroy_framebuffer(fb, None) };
        }
        self.offscreen_framebuffers.clear();
        for img in self.offscreen_images.drain(..) {
            img.destroy(&self.device, &self.alloc);
        }
        // Destroy MSAA images.
        if let Some(img) = self.msaa_color.take() {
            img.destroy(&self.device, &self.alloc);
        }
        if let Some(img) = self.msaa_depth.take() {
            img.destroy(&self.device, &self.alloc);
        }
        for &fb in &self.post_framebuffers {
            unsafe { self.device.destroy_framebuffer(fb, None) };
        }
        self.post_framebuffers.clear();
        // Slice 2/3: destroy the transparent pass framebuffers + the
        // scene_opaque color/depth copy images. Both are sized to the
        // swapchain, so they must be recreated below (previously they were
        // leaked stale across resizes: the copy would target an image of the
        // old extent and the framebuffers referenced destroyed views).
        for &fb in &self.transparent_framebuffers {
            unsafe { self.device.destroy_framebuffer(fb, None) };
        }
        self.transparent_framebuffers.clear();
        self.scene_opaque_color
            .destroy_in_place(&self.device, &self.alloc);
        self.scene_opaque_depth
            .destroy_in_place(&self.device, &self.alloc);
        for &v in &self.swapchain_image_views {
            unsafe { self.device.destroy_image_view(v, None) };
        }
        if let Some(depth) = self.depth.take() {
            depth.destroy(&self.device, &self.alloc);
        }
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }

        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) = swapchain::create_swapchain(
            &self.device,
            &self.swapchain_device,
            &self.surface_instance,
            self.physical_device,
            self.surface,
            self.config.vsync,
        )?;
        let swapchain_image_views =
            swapchain::create_image_views(&self.device, &swapchain_images, swapchain_format)?;
        let depth_format = device::find_depth_format(&self.instance, self.physical_device);
        let depth = GpuImage::depth(&self.device, &self.alloc, swapchain_extent, depth_format)?;

        // Recreate MSAA images.
        if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            self.msaa_color = Some(GpuImage::color_attachment_msaa(
                &self.device,
                &self.alloc,
                swapchain_extent,
                swapchain_format,
                self.msaa_samples,
                "msaa_color",
            )?);
            self.msaa_depth = Some(GpuImage::depth_msaa(
                &self.device,
                &self.alloc,
                swapchain_extent,
                depth_format,
                self.msaa_samples,
            )?);
        }

        // Recreate offscreen images + framebuffers.
        let mut offscreen_images = Vec::with_capacity(swapchain_images.len());
        for _ in 0..swapchain_images.len() {
            let img = GpuImage::color_attachment(
                &self.device,
                &self.alloc,
                swapchain_extent,
                self.swapchain_format,
                "offscreen",
            )?;
            let cmd_init = begin_one_time(&self.device, self.command_pool)?;
            transition_image_layout(
                &self.device,
                cmd_init,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            end_and_submit(
                &self.device,
                self.command_pool,
                self.graphics_queue,
                cmd_init,
            )?;
            offscreen_images.push(img);
        }
                let offscreen_framebuffers = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            let msaa_c = self.msaa_color.as_ref().unwrap();
            let msaa_d = self.msaa_depth.as_ref().unwrap();
            let depth_resolve_active = self.depth_resolve_active;
            offscreen_images
                .iter()
                .map(|img| {
                    if depth_resolve_active {
                        create_framebuffer_with(
                            &self.device,
                            self.render_pass,
                            &[msaa_c.view, msaa_d.view, img.view, depth.view],
                            swapchain_extent,
                        )
                    } else {
                        create_framebuffer_with(
                            &self.device,
                            self.render_pass,
                            &[msaa_c.view, msaa_d.view, img.view],
                            swapchain_extent,
                        )
                    }
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            offscreen_images
                .iter()
                .map(|img| {
                    create_framebuffer_with(
                        &self.device,
                        self.render_pass,
                        &[img.view, depth.view],
                        swapchain_extent,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Recreate the scene_opaque copy images at the new extent (same
        // initial SHADER_READ_ONLY transition as the constructor).
        let scene_opaque_color = GpuImage::scene_opaque(
            &self.device,
            &self.alloc,
            swapchain_extent,
            self.swapchain_format,
            "scene_opaque_color",
        )?;
        let scene_opaque_depth = GpuImage::scene_opaque_depth(
            &self.device,
            &self.alloc,
            swapchain_extent,
            depth_format,
            "scene_opaque_depth",
        )?;
        {
            let cmd_init = begin_one_time(&self.device, self.command_pool)?;
            transition_image_layout(
                &self.device,
                cmd_init,
                scene_opaque_color.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            transition_image_layout(
                &self.device,
                cmd_init,
                scene_opaque_depth.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                1,
                1,
            );
            end_and_submit(&self.device, self.command_pool, self.graphics_queue, cmd_init)?;
        }
        self.scene_opaque_color = scene_opaque_color;
        self.scene_opaque_depth = scene_opaque_depth;

        // Recreate the transparent pass framebuffers against the new
        // offscreen images (they referenced the destroyed ones before).
        // Transparent pass is always single-sample (MSAA=1) — see comment
        // in Renderer::new() for rationale.
        self.transparent_framebuffers = offscreen_images
            .iter()
            .map(|img| {
                create_framebuffer_with(
                    &self.device,
                    self.transparent_render_pass,
                    &[img.view, depth.view],
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Recreate post framebuffers (one per swapchain image view).
        let post_framebuffers = swapchain_image_views
            .iter()
            .map(|&view| {
                create_framebuffer_with(
                    &self.device,
                    self.post_render_pass,
                    &[view],
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;
        // Resize hook: depth.view changed; re-write every per-frame
        // particle depth descriptor so `subpassLoad(depth_input)` reads
        // from the just-recreated view. Capture the view BEFORE moving
        // `depth` into `self.depth`.
        // With MSAA the particle subpass reads the multisampled depth.
        let particle_depth_view = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            self.msaa_depth.as_ref().unwrap().view
        } else {
            depth.view
        };
        self.depth = Some(depth);
        for &set in &self.particle_depth_descriptor_sets {
            pipeline::update_particle_descriptor_set(
                &self.device,
                set,
                particle_depth_view,
            );
        }
        self.offscreen_images = offscreen_images;
        self.offscreen_framebuffers = offscreen_framebuffers;
        self.post_framebuffers = post_framebuffers;

        // Re-write the descriptors that referenced the recreated images:
        //   - chunk sets: bindings 6/7 (scene_opaque color+depth copies)
        //   - post sets: binding 0 (offscreen color), binding 1 (depth)
        // Previously neither was refreshed on resize, so the first resized
        // frame sampled destroyed image views.
        for frame in self.frames.iter() {
            pipeline::update_scene_copy_descriptors(
                &self.device,
                frame.descriptor_set,
                self.scene_opaque_color.view,
                self.scene_opaque_sampler,
                self.scene_opaque_depth.view,
                self.scene_depth_sampler,
            );
        }
        {
            let depth_view = self.depth.as_ref().unwrap().view;
            for (i, &set) in self.post_descriptor_sets.iter().enumerate() {
                let color_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(self.offscreen_images[i].view)
                    .sampler(self.post_sampler);
                let depth_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(depth_view)
                    .sampler(self.depth_sampler);
                let color_infos = [color_info];
                let depth_infos = [depth_info];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&color_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&depth_infos),
                ];
                unsafe { self.device.update_descriptor_sets(&writes, &[]) };
            }
        }
        Ok(())
    }

    /// Record UI draw commands into `cmd`. Must be called inside the render pass
    /// (after the chunk pass). Uploads vertices to the persistent host-visible
    /// buffer, binds the UI pipeline, and draws.
    /// Upload UI vertex/index data to the persistent mapped buffers. Call before
    /// the render pass begins. Returns the index count to draw, or 0 if skipped.
    fn upload_ui(&mut self, ui: &UiDrawData) -> u32 {
        let vbytes = bytemuck::cast_slice(&ui.vertices);
        let ibytes = bytemuck::cast_slice(&ui.indices);
        {
            let vslice = match self.ui_vbo.mapped_slice_mut() {
                Ok(s) => s,
                Err(e) => {
                    log::error!("ui vbo map: {e}");
                    return 0;
                }
            };
            if vbytes.len() > vslice.len() {
                log::warn!(
                    "UI vertices {} exceed buffer {}",
                    vbytes.len(),
                    vslice.len()
                );
                return 0;
            }
            vslice[..vbytes.len()].copy_from_slice(vbytes);
        }
        if let Err(e) = self.ui_vbo.flush_whole(&self.device) {
            log::warn!("ui_vbo flush failed: {e}");
        }
        {
            let islice = match self.ui_ibo.mapped_slice_mut() {
                Ok(s) => s,
                Err(e) => {
                    log::error!("ui ibo map: {e}");
                    return 0;
                }
            };
            if ibytes.len() > islice.len() {
                log::warn!("UI indices {} exceed buffer {}", ibytes.len(), islice.len());
                return 0;
            }
            islice[..ibytes.len()].copy_from_slice(ibytes);
        }
        if let Err(e) = self.ui_ibo.flush_whole(&self.device) {
            log::warn!("ui_ibo flush failed: {e}");
        }
        ui.indices.len() as u32
    }

    /// Record UI draw commands into `cmd`. Must be called inside the render pass.
    fn record_ui(&self, device: &ash::Device, cmd: vk::CommandBuffer, index_count: u32) {
        let ui_viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let ui_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        let push = [
            self.swapchain_extent.width as f32,
            self.swapchain_extent.height as f32,
        ];
        let ui_desc_sets = [self.ui_descriptor_set];
        let vbo = [self.ui_vbo.buffer];

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[ui_viewport]);
            device.cmd_set_scissor(cmd, 0, &[ui_scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.ui_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ui_pipeline_layout,
                0,
                &ui_desc_sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.ui_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
            device.cmd_bind_index_buffer(cmd, self.ui_ibo.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
        }
    }
/// Slice 2 helper: copy the just-rendered main-pass color into the
    /// `scene_opaque_color` sidecar that `TRANSLUCENT_ABSORB` reads via
    /// descriptor binding 6.
    fn record_scene_opaque_copy(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
    ) {
        let offscreen = &self.offscreen_images[image_index as usize];
        let extent3d = vk::Extent3D {
            width: self.swapchain_extent.width,
            height: self.swapchain_extent.height,
            depth: 1,
        };
        unsafe {
            crate::texture::transition_image_layout(
                device, cmd, offscreen.image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageAspectFlags::COLOR, 1, 1,
            );
            crate::texture::transition_image_layout(
                device, cmd, self.scene_opaque_color.image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR, 1, 1,
            );
            let copy = vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0, base_array_layer: 0, layer_count: 1,
                })
                .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0, base_array_layer: 0, layer_count: 1,
                })
                .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .extent(extent3d);
            device.cmd_copy_image(
                cmd, offscreen.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.scene_opaque_color.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            crate::texture::transition_image_layout(
                device, cmd, self.scene_opaque_color.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR, 1, 1,
            );
            crate::texture::transition_image_layout(
                device, cmd, offscreen.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR, 1, 1,
            );
        }

        // ── Depth copy (SSR scene depth, binding 7) ──
        // The single-sample `depth` image holds the scene depth at this point:
        // without MSAA it was the main pass's depth attachment; with MSAA the
        // render pass resolved the multisampled depth into it
        // (`VkSubpassDescriptionDepthStencilResolve`). Copy it texel-for-texel
        // into `scene_opaque_depth` and leave both images in the layouts their
        // other consumers expect (`depth` back to DEPTH_STENCIL_ATTACHMENT so
        // the later SSAO transition stays valid; the copy target to
        // SHADER_READ_ONLY for the transparent pass).
        if self.ssr_depth_valid {
            if let Some(ref depth_img) = self.depth {
                unsafe {
                    crate::texture::transition_image_layout(
                        device, cmd, depth_img.image,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageAspectFlags::DEPTH, 1, 1,
                    );
                    crate::texture::transition_image_layout(
                        device, cmd, self.scene_opaque_depth.image,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageAspectFlags::DEPTH, 1, 1,
                    );
                    let depth_copy = vk::ImageCopy::default()
                        .src_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            mip_level: 0, base_array_layer: 0, layer_count: 1,
                        })
                        .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .dst_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            mip_level: 0, base_array_layer: 0, layer_count: 1,
                        })
                        .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .extent(extent3d);
                    device.cmd_copy_image(
                        cmd, depth_img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.scene_opaque_depth.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[depth_copy],
                    );
                    crate::texture::transition_image_layout(
                        device, cmd, self.scene_opaque_depth.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageAspectFlags::DEPTH, 1, 1,
                    );
                    crate::texture::transition_image_layout(
                        device, cmd, depth_img.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        vk::ImageAspectFlags::DEPTH, 1, 1,
                    );
                }
            }
        }
    }

    /// Slice 2 helper: the second render pass that LOADS color/depth from
    /// `offscreen_images[image_index]`, binds `transparent_pipeline` and the
    /// shared chunk descriptor set (binding 6 = scene_opaque sampler), draws
    /// every chunk's `transparent` PassBuffers entry, then ends.
    fn record_transparent_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        frustum: &voxel_core::Frustum,
        vp_cols: &[f32],
        game_time: f32,
        descriptor_set: vk::DescriptorSet,
        tile_remap_descriptor_set: vk::DescriptorSet,
        cam_pos: glam::Vec3,
    ) {
        let any_transparent = self.chunks.read().iter().any(|(_, b)| b.transparent.is_some());
        if !any_transparent {
            // No transparent draws this frame. The scene_opaque copy restored
            // the offscreen image to COLOR_ATTACHMENT_OPTIMAL in anticipation
            // of this pass; since the pass won't run (its resolve attachment
            // final layout would have produced SHADER_READ_ONLY_OPTIMAL),
            // transition explicitly so the post pass samples a valid layout.
            let offscreen = &self.offscreen_images[image_index as usize];
            crate::texture::transition_image_layout(
                device,
                cmd,
                offscreen.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            return;
        }

        const CHUNK_SIZE: f32 = 16.0;
        // (origin, vbo_buf, ibo_buf, index_count, dist_sq) — copy everything
        // out so we drop the chunks lock before the unsafe draw loop.
        // Frustum-cull transparent chunks so off-screen water is not drawn.
        let mut draws: Vec<(glam::Vec3, vk::Buffer, vk::Buffer, u32, f32)> = Vec::new();
        for (pos, bufs) in self.chunks.read().iter() {
            if let Some(ref t) = bufs.transparent {
                let origin = glam::Vec3::new(
                    pos.0.x as f32 * CHUNK_SIZE,
                    pos.0.y as f32 * CHUNK_SIZE,
                    pos.0.z as f32 * CHUNK_SIZE,
                );
                let min = origin;
                let max = origin + glam::Vec3::splat(CHUNK_SIZE);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                let center = origin + glam::Vec3::splat(CHUNK_SIZE * 0.5);
                draws.push((
                    origin,
                    t.vbo.buffer,
                    t.ibo.buffer,
                    t.index_count,
                    (center - cam_pos).length_squared(),
                ));
            }
        }
        draws.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

        let clear_values = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.transparent_render_pass)
            .framebuffer(self.transparent_framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        // NEGATIVE-height viewport (y = H, height = -H) — the same Y-flip
        // convention every other pass in this renderer uses (see the sky pass,
        // whose viewport the opaque chunk draws inherit). The previous
        // positive-height viewport rendered all transparent geometry
        // vertically mirrored against the projection matrix.
        let vp = vk::Viewport {
            x: 0.0, y: self.swapchain_extent.height as f32,
            width: self.swapchain_extent.width as f32,
            height: -(self.swapchain_extent.height as f32),
            min_depth: 0.0, max_depth: 1.0,
        };
        let sc = vk::Rect2D { offset: vk::Offset2D::default(), extent: self.swapchain_extent };
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, &[vp]);
            device.cmd_set_scissor(cmd, 0, &[sc]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.transparent_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0,
                // Two-set bind: `transparent_pipeline` is created against the
                // chunk material layout (which now has set 1 = tile_remap
                // from cluster A), and `chunk.frag`'s `tile_remap` UBO read
                // would otherwise fail with VUID-08600. Same one-shot bind
                // pattern used in `record_main_pass_setup`.
                &[descriptor_set, tile_remap_descriptor_set], &[],
            );
            for (origin, vbo_buffer, ibo_buffer, index_count, _dist_sq) in draws.iter() {
                let mut pc = [0.0f32; 24];
                pc[0] = origin.x; pc[1] = origin.y; pc[2] = origin.z; pc[3] = 0.0;
                if vp_cols.len() >= 16 { pc[4..20].copy_from_slice(&vp_cols[..16]); }
                pc[20] = game_time; pc[21] = 0.0; pc[22] = 0.0; pc[23] = 0.0;
                device.cmd_push_constants(
                    cmd, self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0, bytemuck::cast_slice(&pc),
                );
                let vbo = [*vbo_buffer];
                device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                device.cmd_bind_index_buffer(cmd, *ibo_buffer, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd, *index_count, 1, 0, 0, 0);
            }
            device.cmd_end_render_pass(cmd);
        }
    }

}

    impl Drop for Renderer {
    fn drop(&mut self) {
        // Wait for the GPU to finish, then drain any pending cmd+fence
        // pairs while the handles are still valid. `device_wait_idle`
        // signals every upload-batch fence, so the `wait_for_fences`
        // inside `drain_pending_destructions` is a no-op for surviving
        // entries, and `free_command_buffers` + `destroy_fence` then
        // run on still-valid handles before pool/device teardown.
        // Without this, every Renderer shutdown would silently leak
        // the per-frame tail of pending cmd+fence pairs.
        if let Err(e) = unsafe { self.device.device_wait_idle() } {
            log::warn!("Renderer::drop: device_wait_idle failed: {e:?}");
        }
        self.drain_pending_destructions();
        let device = &self.device;

        // Chunk buffers.
        let chunks = std::mem::take(&mut self.chunks).into_inner();
        for (_, bufs) in chunks {
            bufs.destroy(device, &self.alloc);
        }
        // Phase-1 GPU-driven pipeline.
        if let Some(mut gpu) = self.gpu_driven.take() {
            gpu.destroy(device, &self.alloc);
        }

        // Loaded models.
        for model in self.models.drain(..) {
            model.destroy(device, &self.alloc);
        }

        // Per-frame.
        for f in self.frames.drain(..) {
            unsafe {
                device.destroy_semaphore(f.image_available, None);
                device.destroy_semaphore(f.render_finished, None);
                device.destroy_fence(f.in_flight_fence, None);
            }
            f.camera_ubo.destroy(device, &self.alloc);
            f.shadow_ubo.destroy(device, &self.alloc);
            // Tile-remap UBO was added in cluster A so the chunk pipeline
            // layout has set-1 binding 0 wired up; without this destroy it
            // leaks `FRAMES_IN_FLIGHT` host-visible UBOs (each 1024 B + a
            // GPU allocation) every Renderer::drop.
            f.tile_remap_ubo.destroy(device, &self.alloc);
        }

        // Atlas + fog UBO + UI resources + sky resources.
        self.atlas.destroy_in_place(device, &self.alloc);
        self.fog_ubo.destroy_in_place(device, &self.alloc);
        self.font_texture.destroy_in_place(device, &self.alloc);
        self.minimap_texture.destroy_in_place(device, &self.alloc);
        self.ui_vbo.destroy_in_place(device, &self.alloc);
        self.ui_ibo.destroy_in_place(device, &self.alloc);
        self.sky_ubo.destroy_in_place(device, &self.alloc);
        // Tile material lookup table (chunk binding 5). Was added in the
        // cluster A fix wave (engine pushes a fresh table each frame via
        // `set_tile_material_table`) but no Drop cleanup — leaked 4 KB
        // host-visible every Renderer::drop.
        self.tile_material_ubo.destroy_in_place(device, &self.alloc);

        // MSAA images.
        if let Some(mut img) = self.msaa_color.take() {
            img.destroy_in_place(device, &self.alloc);
        }
        if let Some(mut img) = self.msaa_depth.take() {
            img.destroy_in_place(device, &self.alloc);
        }

        // Depth attachment (the main render pass's depth image).
        if let Some(mut depth) = self.depth.take() {
            depth.destroy_in_place(device, &self.alloc);
        }

        // Entity pipeline VBO/IBO (shared between world entities and held items).
        self.entity_vbo.destroy_in_place(device, &self.alloc);
        self.entity_ibo.destroy_in_place(device, &self.alloc);

        // Overlay (brush wireframe) VBO.
        self.overlay_vbo.destroy_in_place(device, &self.alloc);

        // Particle instance VBO (the source of the leak shown in the warning).
        self.particle_instance_vbo.destroy_in_place(device, &self.alloc);

        // Panorama cubemap image/view/sampler.
        self.panorama.destroy(device, &self.alloc);
        // Placeholder cubemap (only allocated when no `assets/textures/panorama/*.png`
        // files were present this launch). Pre-refactor these four resources
        // lived in `let` locals inside the constructor's else-branch and
        // leaked 3 KB device-local every Renderer::drop.
        if let Some(mut pp) = self.panorama_placeholder.take() {
            pp.destroy(device, &self.alloc);
        }

        // Post-processing pass resources.
        for &fb in &self.post_framebuffers {
            unsafe { device.destroy_framebuffer(fb, None); }
        }
        self.post_framebuffers.clear();
        unsafe {
            device.destroy_pipeline(self.post_pipeline, None);
            device.destroy_pipeline_layout(self.post_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.post_descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.post_descriptor_pool, None);
            device.destroy_sampler(self.post_sampler, None);
            device.destroy_sampler(self.depth_sampler, None);
            device.destroy_render_pass(self.post_render_pass, None);
        }

        // Sky + panorama descriptor pool/layout (sets freed with their pools).
        unsafe {
            device.destroy_descriptor_set_layout(self.sky_descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.sky_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.panorama_descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.panorama_descriptor_pool, None);
        }

        // Shadow sampler.
        unsafe { device.destroy_sampler(self.shadow_sampler, None); }

        // Shadow resources.
        for &fb in &self.shadow_framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        self.shadow_framebuffers.clear();
        for &v in &self.shadow_layer_views {
            unsafe { device.destroy_image_view(v, None) };
        }
        self.shadow_layer_views.clear();
        self.shadow_image.destroy_in_place(device, &self.alloc);

        // Offscreen resources.
        for &fb in &self.offscreen_framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        self.offscreen_framebuffers.clear();
        for img in self.offscreen_images.drain(..) {
            img.destroy(device, &self.alloc);
        }

        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_query_pool(self.query_pool, None);/*SLICE2_DROP*/
            // --- Slice 2 cleanup: scene_opaque_color + transparent render pass (framebuffers first, then render_pass, sampler, image) ---
            for fb in self.transparent_framebuffers.drain(..) {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_render_pass(self.transparent_render_pass, None);
            device.destroy_sampler(self.scene_opaque_sampler, None);
            self.scene_opaque_color.destroy_in_place(&self.device, &self.alloc);
            // --- Slice 3 (reflections) cleanup ---
            device.destroy_sampler(self.scene_depth_sampler, None);
            self.scene_opaque_depth.destroy_in_place(&self.device, &self.alloc);
            self.reflection_ubo.destroy_in_place(&self.device, &self.alloc);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.wireframe_pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.ui_pipeline, None);
            device.destroy_pipeline_layout(self.ui_pipeline_layout, None);
            device.destroy_descriptor_pool(self.ui_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.ui_descriptor_set_layout, None);
            device.destroy_pipeline(self.sky_pipeline, None);
            device.destroy_pipeline_layout(self.sky_pipeline_layout, None);
            device.destroy_pipeline(self.entity_pipeline, None);
            device.destroy_pipeline_layout(self.entity_pipeline_layout, None);
            device.destroy_pipeline(self.entity_held_pipeline, None);
            device.destroy_pipeline(self.overlay_pipeline, None);

        // Occlusion culling cleanup.
        self.device.destroy_pipeline(self.occlusion_pipeline, None);
        self.aabb_index_buffer.destroy_in_place(device, &self.alloc);
        for frame in self.occlusion_frames.get_mut().drain(..) {
            self.device.destroy_query_pool(frame.query_pool, None);
        }

            // Particle subpass resources.
            // Sets owned by `particle_depth_descriptor_pool` are freed when
            // the pool is destroyed; only the pool + layout + pipeline layout
            // need explicit destroy calls here.
            device.destroy_pipeline(self.particle_pipeline, None);
            device.destroy_pipeline_layout(self.particle_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.particle_depth_set_layout, None);
            device.destroy_descriptor_pool(self.particle_depth_descriptor_pool, None);
        }

        // Drop the allocator (frees remaining allocations) BEFORE destroying the device.
        unsafe {
            ManuallyDrop::drop(&mut self.alloc);
        }

        // Device, surface, debug messenger, instance.
        unsafe {
            device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            if let Some(m) = self.debug_messenger.take() {
                let du = ash::ext::debug_utils::Instance::new(&self._entry, &self.instance);
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

