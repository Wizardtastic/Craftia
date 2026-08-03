use anyhow::{anyhow, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};

// ── UBO types ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct CameraUbo {
    /// xyz = camera position, w = fog max distance.
    pub(super) cam_pos_and_maxdist: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct FogUbo {
    /// rgb = fog colour, a = unused.
    pub(super) color_and_density: [f32; 4],
    /// x = ambient brightness (day/night), yzw = sun direction (for future use).
    pub(super) ambient_and_sun: [f32; 4],
}

/// Per-frame sky uniform: horizon colour, zenith colour, sun direction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct SkyUbo {
    pub(super) horizon: [f32; 4],
    pub(super) zenith: [f32; 4],
    pub(super) sun_dir: [f32; 4],
}

/// Per-frame shadow uniform for cascaded shadow maps (binding 4 of the chunk set).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct ShadowUbo {
    pub(super) cascade_vps: [[f32; 16]; 4],
    pub(super) cascade_splits: [f32; 4],
    pub(super) light_dir_and_bias: [f32; 4],
}

/// Per-frame tile remap (set 1, binding 0). Maps canonical tile IDs to
/// animated tile IDs for procedural texture variation. The chunk fragment
/// shader reads `tile_remap.map[frag_tile]` at line 257 of
/// `shaders/chunk.frag` so the descriptor set layout MUST declare set 1
/// binding 0 — the validation layer otherwise reports VUID-07988 ("binding
/// was not declared in VkPipelineLayoutCreateInfo::pSetLayouts[1]") at
/// pipeline creation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct TileRemapUbo {
    pub(super) map: [u32; 256],
}

// `[u32; 256]` doesn't implement `Default` (only fixed-size arrays up to a
// certain length do), so we provide one explicitly. Maps to an identity
// remap (every canonical tile ID renders to itself) — the animated-tile
// rotation code lives elsewhere and writes a different map into this UBO
// before binding.
impl Default for TileRemapUbo {
    fn default() -> Self {
        let mut map = [0u32; 256];
        for (i, slot) in map.iter_mut().enumerate() {
            *slot = i as u32;
        }
        Self { map }
    }
}

/// Per-frame reflection/environment uniform (binding 8 of the chunk set).
///
/// `shaders/chunk.frag`. The sky colours mirror the sky UBO so the shader can
/// evaluate the same analytic sky for reflected rays instead of rendering a
/// per-frame cubemap probe (see docs/notes/water_reflections.md).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct ReflectionUbo {
    /// rgb = horizon colour (matches sky UBO), a unused.
    pub(super) sky_horizon: [f32; 4],
    /// rgb = zenith colour (matches sky UBO), a unused.
    pub(super) sky_zenith: [f32; 4],
    /// xyz = sun direction (not necessarily normalised), w = master reflection
    /// strength in [0, 1] (0 disables every reflection path).
    pub(super) sun_dir_str: [f32; 4],
    /// x = camera near plane, y = camera far plane, z = camera-underwater flag
    /// (0/1), w = SSR-valid flag (0 when no single-sample scene depth exists,
    /// e.g. MSAA without depth-stencil-resolve support → sky-only fallback).
    pub(super) proj_misc: [f32; 4],
}

// ── Render pass / descriptor / pipeline helpers ─────────────────────────

/// Create the main render pass.
///
/// When `msaa_samples` is `TYPE_1`, the render pass uses 2 attachments
/// (color + depth) exactly as before. When `msaa_samples` is e.g. `TYPE_4`,
/// 3 attachments are used:
///   0 — MSAA color (transient, DONT_CARE store)
///   1 — MSAA depth (transient, DONT_CARE store)
///   2 — resolve color (single-sample, STORE, sampled by post pass)
///
/// The resolve attachment is only referenced by the last subpass (particles,
/// subpass 1) so that the final image includes both scene and particle writes.
///
/// When `depth_resolve_mode` is `Some(mode)` AND MSAA is active, a 4th
/// attachment is appended:
///   3 — resolve depth (single-sample, STORE, copied to `scene_opaque_depth`
///       after the pass for the water/glass SSR ray-march)
/// and subpass 0 resolves its multisampled depth into it via a
/// `VkSubpassDescriptionDepthStencilResolve` (Vulkan 1.2 core). Subpass 1
/// (particles) does not write depth, so only subpass 0 resolves it; this also
/// keeps `capture_frame`, which only runs subpass 0, producing resolved depth.
pub(super) fn create_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    depth_resolve_mode: Option<vk::ResolveModeFlags>,
) -> Result<vk::RenderPass> {
    let msaa = msaa_samples != vk::SampleCountFlags::TYPE_1;

    // Depth-stencil-resolve requires the Vulkan 1.2 `VkRenderPassCreateInfo2`
    // API (ash 0.38 only chains `SubpassDescriptionDepthStencilResolve` onto
    // `SubpassDescription2`). Dispatch to the v2 builder; the caller only
    // passes `Some(mode)` when the device is Vulkan 1.2+.
    if msaa {
        if let Some(mode) = depth_resolve_mode {
            return create_render_pass_depth_resolve(
                device,
                color_format,
                depth_format,
                msaa_samples,
                mode,
            );
        }
    }

    // ── Attachment 0: (MSAA) color ──
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(if msaa {
            vk::AttachmentStoreOp::DONT_CARE
        } else {
            vk::AttachmentStoreOp::STORE
        })
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(if msaa {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        });

    // ── Attachment 1: (MSAA) depth ──
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // ── Attachment 2 (MSAA only): resolve color ──
    let resolve_attachment = if msaa {
        Some(
            vk::AttachmentDescription::default()
                .format(color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        )
    } else {
        None
    };

    // ── Subpass 0: chunks + sky + entities + UI ──
    let color_ref_0 = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref_0 = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_refs_0 = [color_ref_0];
    // Resolve on BOTH subpasses so that both draw_frame (which runs both) and
    // capture_frame (which only runs subpass 0) produce a resolved image.
    // Subpass 1's resolve overwrites subpass 0's when particles run.
    let resolve_ref_0 = if msaa {
        let r = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        Some([r])
    } else {
        None
    };
    let mut subpass_0_desc = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs_0)
        .depth_stencil_attachment(&depth_ref_0);
    if let Some(ref r) = resolve_ref_0 {
        subpass_0_desc = subpass_0_desc.resolve_attachments(r);
    }
    let subpass_0 = subpass_0_desc;

    // ── Subpass 1: particles ──
    // Reads depth as an input attachment (MSAA or single-sample) for soft fade.
    // Writes to the MSAA color attachment and (when MSAA) resolves to the
    // single-sample resolve attachment at the end of this subpass.
    let color_ref_1 = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let input_ref_1 = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    let color_refs_1 = [color_ref_1];
    let input_refs_1 = [input_ref_1];
    let resolve_ref_1 = if msaa {
        let r = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        Some([r])
    } else {
        None
    };
    let mut subpass_1_desc = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs_1)
        .input_attachments(&input_refs_1);
    if let Some(ref r) = resolve_ref_1 {
        subpass_1_desc = subpass_1_desc.resolve_attachments(r);
    }
    let subpass_1 = subpass_1_desc;

    // External -> subpass 0 (initial layout transition + clear).
    let dep_ext_to_0 = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    // Subpass 0 -> subpass 1: chunks' finished writes feed the particle pass.
    let dep_0_to_1 = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(1)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
        .dependency_flags(vk::DependencyFlags::BY_REGION);

    // Build the attachment list.
    let mut attachments = vec![color_attachment, depth_attachment];
    if let Some(ra) = resolve_attachment {
        attachments.push(ra);
    }
    let subpasses = [subpass_0, subpass_1];
    let dependencies = [dep_ext_to_0, dep_0_to_1];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_render_pass: {e:?}"))
}

/// Vulkan 1.2 variant of [`create_render_pass`] used when MSAA is active AND
/// the device supports depth-stencil-resolve. Identical attachment/subpass
/// layout to the MSAA flavour of the v1 builder, plus a 4th attachment:
///   3 — resolve depth (single-sample, STORE)
/// Only subpass 0 resolves its multisampled depth into it via a chained
/// `VkSubpassDescriptionDepthStencilResolve` (Vulkan 1.2 core). Subpass 1
/// (particles) only reads depth as an input attachment, so it must not chain
/// the resolve struct (and its resolved depth from subpass 0 is preserved
/// untouched through subpass 1).
///
/// Subpass 0 chains a `VkSubpassDescriptionDepthStencilResolve` with
/// `depth_resolve_mode` (MIN when supported, else SAMPLE_ZERO). The resolved
/// depth is copied to `scene_opaque_depth` after the pass for the water/glass
/// SSR ray-march.
fn create_render_pass_depth_resolve(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    depth_resolve_mode: vk::ResolveModeFlags,
) -> Result<vk::RenderPass> {
    let attachments = [
        // 0: MSAA color (transient)
        vk::AttachmentDescription2::default()
            .format(color_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        // 1: MSAA depth (transient)
        vk::AttachmentDescription2::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        // 2: resolve color (single-sample, sampled by post pass)
        vk::AttachmentDescription2::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        // 3: resolve depth (single-sample, copied to scene_opaque_depth)
        vk::AttachmentDescription2::default()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref_0 = vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR);
    let depth_ref_0 = vk::AttachmentReference2::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH);
    let color_refs_0 = [color_ref_0];
    let resolve_ref_0 = [vk::AttachmentReference2::default()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR)];
    let depth_resolve_ref_0 = vk::AttachmentReference2::default()
        .attachment(3)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH);
    let mut dsr_0 = vk::SubpassDescriptionDepthStencilResolve::default()
        .depth_resolve_mode(depth_resolve_mode)
        .stencil_resolve_mode(vk::ResolveModeFlags::NONE)
        .depth_stencil_resolve_attachment(&depth_resolve_ref_0);
    let mut subpass_0_desc = vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs_0)
        .depth_stencil_attachment(&depth_ref_0)
        .resolve_attachments(&resolve_ref_0);
    subpass_0_desc = subpass_0_desc.push_next(&mut dsr_0);
    let subpass_0 = subpass_0_desc;

    let color_ref_1 = vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR);
    let input_ref_1 = vk::AttachmentReference2::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH);
    let color_refs_1 = [color_ref_1];
    let input_refs_1 = [input_ref_1];
    let resolve_ref_1 = [vk::AttachmentReference2::default()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR)];
    // NOTE: subpass 1 (particles) reads depth as an input attachment and does
    // NOT write it, so it has no depth-stencil attachment and must NOT chain a
    // SubpassDescriptionDepthStencilResolve. Chaining one there makes the
    // driver link the depth resolve onto a NULL depth-stencil attachment
    // (NULL+0x20 write, SIGSEGV in vkCreateRenderPass2).
    let subpass_1_desc = vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs_1)
        .input_attachments(&input_refs_1)
        .resolve_attachments(&resolve_ref_1);
    let subpass_1 = subpass_1_desc;

    let dependencies = [
        vk::SubpassDependency2::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
        vk::SubpassDependency2::default()
            .src_subpass(0)
            .dst_subpass(1)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION),
    ];

    let subpasses = [subpass_0, subpass_1];
    let create_info = vk::RenderPassCreateInfo2::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass2(&create_info, None) }
        .map_err(|e| anyhow!("create_render_pass2 (depth resolve): {e:?}"))
}

pub(super) fn create_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 5 = tile material lookup table (leaves SSS, wet-edge tint,
        // submerged-terrain caustics). Updated once per frame from the world
        // registry + the engine's water-level/strength scalars. See
        // `shaders/chunk.frag` for the std430 declaration consuming this.
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 6 = scene_opaque_color: a copy of the main render pass's
        // color attachment made BEFORE the transparent pass starts. Sampled
        // by the chunk fragment shader's TRANSLUCENT_ABSORB / WATER paths,
        // and by the water refraction path (`glass_idx`-style distortion of
        // the UV uses binding 6).
        //
        // Opaque chunks don't read this binding in their shader, but Vulkan
        // still requires a valid (or null) descriptor for every binding, so
        // we always bind scene_opaque_color[frame_idx] for both passes.
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 7 = scene_opaque_depth: single-sample copy of the scene
        // depth made alongside binding 6. Sampled (NEAREST) by the chunk
        // fragment shader's SSR ray-march + water-column absorption depth.
        vk::DescriptorSetLayoutBinding::default()
            .binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 8 = reflection/environment UBO (sky colours, sun direction,
        // near/far, master strength, SSR-valid flag). Drives every reflection
        // path (water, glass, opaque REFLECTIVE tiles).
        vk::DescriptorSetLayoutBinding::default()
            .binding(8)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_descriptor_set_layout: {e:?}"))
}

pub(super) fn create_descriptor_pool(
    device: &ash::Device,
    max_sets: usize,
) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        // 6 chunk UBOs (camera + fog + shadow + material table + reflection
        // + tile_remap) per frame set. The extra UBO is the set-1 tile_remap
        // binding the chunk fragment shader reads, see `TileRemapUbo`.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: (max_sets * 6) as u32,
        },
        // 4 combined image samplers per frame set (atlas + shadow +
        // scene_opaque_color + scene_opaque_depth).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: (max_sets * 4) as u32,
        },
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(max_sets as u32)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe { device.create_descriptor_pool(&create_info, None) }
        .map_err(|e| anyhow!("create_descriptor_pool: {e:?}"))
}

pub(super) fn allocate_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![layout; count];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| anyhow!("allocate_descriptor_sets: {e:?}"))
}

/// All buffer/image resources referenced by the chunk descriptor set.
/// Bundled so [`update_descriptor_set`] stays below the arg-count lint.
#[derive(Clone, Copy)]
pub(super) struct DescriptorResources {
    pub camera_buffer: vk::Buffer,
    pub fog_buffer: vk::Buffer,
    pub atlas_view: vk::ImageView,
    pub atlas_sampler: vk::Sampler,
    pub shadow_view: vk::ImageView,
    pub shadow_sampler: vk::Sampler,
    pub shadow_buffer: vk::Buffer,
    pub material_table_buffer: vk::Buffer,
    pub material_table_size: vk::DeviceSize,
    pub scene_opaque_view: vk::ImageView,
    pub scene_opaque_sampler: vk::Sampler,
    pub scene_depth_view: vk::ImageView,
    pub scene_depth_sampler: vk::Sampler,
    pub reflection_buffer: vk::Buffer,
}

pub(super) fn update_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    res: DescriptorResources,
) {
    let DescriptorResources {
        camera_buffer,
        fog_buffer,
        atlas_view,
        atlas_sampler,
        shadow_view,
        shadow_sampler,
        shadow_buffer,
        material_table_buffer,
        material_table_size,
        scene_opaque_view,
        scene_opaque_sampler,
        scene_depth_view,
        scene_depth_sampler,
        reflection_buffer,
    } = res;
    let cam_info = vk::DescriptorBufferInfo::default()
        .buffer(camera_buffer)
        .offset(0)
        .range(std::mem::size_of::<CameraUbo>() as u64);
    let fog_info = vk::DescriptorBufferInfo::default()
        .buffer(fog_buffer)
        .offset(0)
        .range(std::mem::size_of::<FogUbo>() as u64);
    let shadow_info = vk::DescriptorBufferInfo::default()
        .buffer(shadow_buffer)
        .offset(0)
        .range(std::mem::size_of::<ShadowUbo>() as u64);
    let material_info = vk::DescriptorBufferInfo::default()
        .buffer(material_table_buffer)
        .offset(0)
        .range(material_table_size);
    let reflection_info = vk::DescriptorBufferInfo::default()
        .buffer(reflection_buffer)
        .offset(0)
        .range(std::mem::size_of::<ReflectionUbo>() as u64);
    let atlas_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(atlas_view)
        .sampler(atlas_sampler);
    let shadow_img_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(shadow_view)
        .sampler(shadow_sampler);
    let scene_opaque_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(scene_opaque_view)
        .sampler(scene_opaque_sampler);
    let scene_depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(scene_depth_view)
        .sampler(scene_depth_sampler);

    let cam_infos = [cam_info];
    let atlas_infos = [atlas_info];
    let fog_infos = [fog_info];
    let shadow_img_infos = [shadow_img_info];
    let shadow_buf_infos = [shadow_info];
    let material_buf_infos = [material_info];
    let scene_opaque_infos = [scene_opaque_info];
    let scene_depth_infos = [scene_depth_info];
    let reflection_buf_infos = [reflection_info];

    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&cam_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&atlas_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&fog_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&shadow_img_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&shadow_buf_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&material_buf_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&scene_opaque_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&scene_depth_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(8)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&reflection_buf_infos),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Re-write ONLY the scene-copy bindings (6 = scene_opaque_color,
/// 7 = scene_opaque_depth) of a chunk descriptor set. Called from
/// `recreate_swapchain` after the scene images are recreated, where the other
/// bindings (buffers + atlas + shadow) are still valid.
pub(super) fn update_scene_copy_descriptors(
    device: &ash::Device,
    set: vk::DescriptorSet,
    scene_opaque_view: vk::ImageView,
    scene_opaque_sampler: vk::Sampler,
    scene_depth_view: vk::ImageView,
    scene_depth_sampler: vk::Sampler,
) {
    let scene_opaque_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(scene_opaque_view)
        .sampler(scene_opaque_sampler);
    let scene_depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(scene_depth_view)
        .sampler(scene_depth_sampler);
    let scene_opaque_infos = [scene_opaque_info];
    let scene_depth_infos = [scene_depth_info];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&scene_opaque_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&scene_depth_infos),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Create the chunk material set-1 descriptor set layout (sets binding 0
/// = the `tile_remap` UBO consumed by `shaders/chunk.frag`). The chunk
/// pipeline layout stitches this layout in at set 1 so the shader's
/// `layout(set = 1, binding = 0) uniform TileRemap` resolves.
pub(super) fn create_tile_remap_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_tile_remap_descriptor_set_layout: {e:?}"))
}

/// Bind the per-frame tile_remap UBO into the set-1 descriptor set at
/// binding 0. Called once per frame from `Renderer::new`.
pub(super) fn update_tile_remap_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    tile_remap_buffer: vk::Buffer,
) {
    let tile_remap_info = vk::DescriptorBufferInfo::default()
        .buffer(tile_remap_buffer)
        .offset(0)
        .range(std::mem::size_of::<TileRemapUbo>() as u64);
    let tile_remap_infos = [tile_remap_info];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&tile_remap_infos)];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

pub(super) fn create_pipeline_layout(
    device: &ash::Device,
    chunk_set_layout: vk::DescriptorSetLayout,
    tile_remap_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(96); // vec4 + mat4 + vec4 (origin, view_proj, time)
                   // TWO set layouts: `chunk_set_layout` (camera, fog, atlas, shadow,
                   // material table, scene-opaque color/depth, reflection) and
                   // `tile_remap_set_layout` (TileRemap UBO consumed by the chunk frag at
                   // `layout(set = 1, binding = 0) uniform TileRemap`). The chunk pipeline
                   // layout MUST include both — without tile_remap at set 1 binding 0,
                   // `vkCreateGraphicsPipelines` fails VUID-07988 on chunk.frag.
    let set_layouts = [chunk_set_layout, tile_remap_set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_pipeline_layout: {e:?}"))
}

/// Raster state + shaders for [`create_graphics_pipeline`].
/// Bundled so the pipeline builder stays below the arg-count lint.
#[derive(Clone, Copy)]
pub(super) struct GraphicsPipelineConfig<'a> {
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub vs_spirv: &'a [u8],
    pub fs_spirv: &'a [u8],
    pub msaa_samples: vk::SampleCountFlags,
    pub depth_write: bool,
}

pub(super) fn create_graphics_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    config: GraphicsPipelineConfig<'_>,
) -> Result<vk::Pipeline> {
    let GraphicsPipelineConfig {
        polygon_mode,
        cull_mode,
        vs_spirv,
        fs_spirv,
        msaa_samples,
        depth_write,
    } = config;
    let vert_spv: &[u8] = vs_spirv;
    let frag_spv: &[u8] = fs_spirv;
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("vert shader module: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("frag shader module: {e:?}"))?;

    let shadow_enabled = [0u32];
    let spec_map = [vk::SpecializationMapEntry::default()
        .constant_id(0)
        .size(std::mem::size_of::<u32>())
        .offset(0)];
    let spec_info = vk::SpecializationInfo::default()
        .map_entries(&spec_map)
        .data(bytemuck::cast_slice(&shadow_enabled));
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main")
            .specialization_info(&spec_info),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_UINT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(4)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(28),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .cull_mode(cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(depth_write)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines =
        result.map_err(|(_pipelines, e)| anyhow!("create_graphics_pipelines: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── UI pipeline helpers ──────────────────────────────────────────────────

/// Convert SPIR-V bytes (from `include_bytes!`) to a properly aligned `&[u32]`.
/// `include_bytes!` returns `&[u8]` (alignment 1), but Vulkan requires `&[u32]`
/// (alignment 4). We copy through an aligned `Vec<u32>` to avoid bytemuck panics.
pub(crate) fn spirv_to_u32(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

pub(super) fn create_ui_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_ui_descriptor_set_layout: {e:?}"))
}

pub(super) fn allocate_ui_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| anyhow!("allocate_ui_descriptor_set: {e:?}"))?;
    Ok(sets[0])
}

/// The three texture (view, sampler) pairs bound by the UI descriptor set.
/// Bundled so [`update_ui_descriptor_set`] stays below the arg-count lint.
#[derive(Clone, Copy)]
pub(super) struct UiTextures {
    pub block: (vk::ImageView, vk::Sampler),
    pub font: (vk::ImageView, vk::Sampler),
    pub minimap: (vk::ImageView, vk::Sampler),
}

pub(super) fn update_ui_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    textures: UiTextures,
) {
    let UiTextures {
        block: (block_view, block_sampler),
        font: (font_view, font_sampler),
        minimap: (minimap_view, minimap_sampler),
    } = textures;
    let block_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(block_view)
        .sampler(block_sampler);
    let font_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(font_view)
        .sampler(font_sampler);
    let minimap_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(minimap_view)
        .sampler(minimap_sampler);
    let block_infos = [block_info];
    let font_infos = [font_info];
    let minimap_infos = [minimap_info];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&block_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&font_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&minimap_infos),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

pub(super) fn create_ui_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(8); // vec2 screen_size
    let set_layouts = [set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_ui_pipeline_layout: {e:?}"))
}

pub(super) fn create_ui_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    let vert_spv: &[u8] = vs_spirv;
    let frag_spv: &[u8] = fs_spirv;
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("ui vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("ui frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::ui::UiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(16),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);

    // No depth test/write for UI — it draws on top of everything.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);

    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_ui_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── Sky pipeline helpers ─────────────────────────────────────────────────

pub(super) fn create_sky_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_sky_descriptor_set_layout: {e:?}"))
}

pub(super) fn create_sky_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80); // mat4 inverse view-proj + vec4 camera_pos
    let set_layouts = [set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_sky_pipeline_layout: {e:?}"))
}

pub(super) fn create_sky_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    let vert_spv: &[u8] = vs_spirv;
    let frag_spv: &[u8] = fs_spirv;
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("sky vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("sky frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    // No vertex input — the sky shader generates a full-screen triangle from
    // gl_VertexIndex.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);

    // Sky pass: depth test LESS_OR_EQUAL (so it draws at depth=1, behind everything),
    // no depth write (so chunks can overwrite it).
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    // No blending for the sky — it's the background.
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_sky_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── Panorama pipeline ─────────────────────────────────────────────────

/// Create the descriptor set layout for the panorama cubemap (1 samplerCube).
pub(super) fn create_panorama_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_panorama_descriptor_set_layout: {e:?}"))
}

/// Create the pipeline layout for the panorama pass (push constants only,
/// no UBO — the cubemap descriptor set is bound at set 0).
pub(super) fn create_panorama_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    // Push constants: mat4 inverse_view_proj (64 bytes) + vec4 camera_pos (16 bytes) = 80 bytes.
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80);
    let layouts = [descriptor_set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_panorama_pipeline_layout: {e:?}"))
}

/// Create the panorama graphics pipeline. Identical to the sky pipeline
/// (full-screen triangle, depth test LESS_OR_EQUAL, no depth write) but
/// uses a cubemap sampler instead of a UBO.
pub(super) fn create_panorama_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    let vert_code = spirv_to_u32(vs_spirv);
    let frag_code = spirv_to_u32(fs_spirv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("panorama vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("panorama frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisampling =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);
    // Same depth settings as sky: draws behind everything.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_panorama_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── Shadow + Post pipeline helpers ─────────────────────────────────────────

/// Create the transparent-chunk render pass (slice 2). It LOADS the color +
/// depth attachments of the previous main render pass into the same offscreen
/// image, draws the transparent chunk meshes (which sample `scene_opaque_color`
/// via descriptor binding 6 for absorption / refraction), and writes the result
/// back to the same image. This is the second of a three-stage pipeline:
///   1. main render pass (opaque chunks, sky, entities, UI, particles)
///   2. vkCmdCopyImage offscreen -> scene_opaque_color + barrier
///   3. THIS transparent render pass (transparent chunks only) -> STORE
pub(super) fn create_transparent_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::RenderPass> {
    let msaa = msaa_samples != vk::SampleCountFlags::TYPE_1;

    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(if msaa {
            vk::AttachmentStoreOp::DONT_CARE
        } else {
            vk::AttachmentStoreOp::STORE
        })
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(if msaa {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        });

    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_ref_0 = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref_0 = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_refs_0 = [color_ref_0];
    let resolve_ref_0 = if msaa {
        let r = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        Some([r])
    } else {
        None
    };
    let mut subpass_desc = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs_0)
        .depth_stencil_attachment(&depth_ref_0);
    if let Some(ref r) = resolve_ref_0 {
        subpass_desc = subpass_desc.resolve_attachments(r);
    }
    let subpass_0 = subpass_desc;

    let dep_ext_to_0 = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
        );

    let mut attachments = vec![color_attachment, depth_attachment];
    if msaa {
        // We resolve into attachment 2 so the post-processing pass can sample
        // a single-sample image even when MSAA is on.
        attachments.push(
            vk::AttachmentDescription::default()
                .format(color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        );
    }
    let subpasses = [subpass_0];
    let dependencies = [dep_ext_to_0];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_transparent_render_pass: {e:?}"))
}

pub(super) fn create_shadow_render_pass(
    device: &ash::Device,
    depth_format: vk::Format,
) -> Result<vk::RenderPass> {
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let depth_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref);

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

    let attachments = [depth_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_shadow_render_pass: {e:?}"))
}

pub(super) fn create_shadow_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_shadow_pipeline_layout: {e:?}"))
}

pub(super) fn create_shadow_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
) -> Result<vk::Pipeline> {
    let vert_spv: &[u8] = vs_spirv;
    let frag_spv: &[u8] = fs_spirv;
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("shadow vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("shadow frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_UINT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(4)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(28),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(2.0)
        .depth_bias_slope_factor(1.5);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_shadow_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

pub(super) fn create_post_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = [color_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_post_render_pass: {e:?}"))
}

pub(super) fn create_post_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    // 48 bytes: vec4 params (16) + vec4 ssao (16) + vec4 proj (16).
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(48);
    let push_ranges = [push_range];
    let set_layouts = [descriptor_set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_post_pipeline_layout: {e:?}"))
}

pub(super) fn create_post_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
) -> Result<vk::Pipeline> {
    let vert_spv: &[u8] = vs_spirv;
    let frag_spv: &[u8] = fs_spirv;
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("post vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("post frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_post_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

pub(super) fn create_post_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 1: depth buffer for SSAO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_post_descriptor_set_layout: {e:?}"))
}

// ── Particle pipeline helpers ─────────────────────────────────────────

/// Create the particle graphics pipeline (subpass 1 of the main render pass).
///
/// Reads the same `set 0` chunk descriptor set as chunks/entities (Camera UBO
/// at binding 0, atlas sampler at binding 1 — fog/shadow bindings are simply
/// unused by the particle shader). Push constants carry the inverse
/// view-projection matrix so the vertex shader can billboard quad vertices
/// against the camera and project them back to clip space.
///
/// Vertex input:
///   - `binding 0` (VERTEX rate, 32-byte stride): the unit quad in
///     `entity::unit_quad_vertices` (re-uses the existing renderer VBO).
///   - `binding 1` (INSTANCE rate, 32-byte stride): a
///     `crate::particle::ParticleInstance` layout, uploaded each frame.
///
/// Phase 1 depth behaviour: depth test `LESS` (so chunks occlude particles),
/// depth write disabled (particles don't write into the depth buffer — the
/// depth buffer is already correct after subpass 0). Premultiplied alpha
/// blending matches the shader's output.
pub(super) fn create_particle_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    let vert_code = spirv_to_u32(vs_spirv);
    let frag_code = spirv_to_u32(fs_spirv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("particle vert shader module: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("particle frag shader module: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    // Two bindings: the quad (binding 0, vertex-rate, 32-byte stride) and the
    // per-instance particle data (binding 1, instance-rate, 32-byte stride).
    // Quad attributes we read: pos (location 0, vec3, offset 0) and uv
    // (location 1, vec2, offset 12). The remaining 16 bytes of each 32-byte
    // vertex are ignored by the particle shader.
    let quad_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let inst_binding = vk::VertexInputBindingDescription::default()
        .binding(1)
        .stride(std::mem::size_of::<crate::particle::ParticleInstance>() as u32)
        .input_rate(vk::VertexInputRate::INSTANCE);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(3)
            .format(vk::Format::R32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(4)
            .format(vk::Format::R32_UINT)
            .offset(16),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(5)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(6)
            .format(vk::Format::R32_SFLOAT)
            .offset(24),
    ];
    let vertex_bindings = [quad_binding, inst_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE) // billboards face the camera always
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples);

    // Phase 2: depth test + write are BOTH disabled. The subpass 1
    // attachment is referenced as an input attachment, not a depth-stencil
    // attachment (Vulkan forbids both at the same time). The fragment
    // shader performs the depth comparison + soft fade via
    // `subpassLoad(depth_input)` and discards if the particle is behind
    // geometry.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Premultiplied alpha blend: the shader outputs (rgb * a, a) and we
    // composite as src.RGB * ONE + dst.RGB * (1 - src.A). This matches the
    // output of `shaders/particle.frag`.
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        // Bind against subpass 1 (see `create_render_pass` for the ordering).
        .subpass(1);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_particle_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

/// Layout for the particle-only descriptor set (set 1, binding 0). Holds the
/// depth attachment as an input attachment — the fragment shader fetches it
/// via `subpassLoad` to compute the soft-fade.
pub(super) fn create_particle_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_particle_descriptor_set_layout: {e:?}"))
}

/// Pipeline layout for the particle pipeline: takes TWO set layouts
///   - set 0 = chunk layout (camera + atlas + fog + shadow + shadow_ubo; shader
///     only reads camera + atlas)
///   - set 1 = particle layout (depth input attachment)
///
/// Push constants carry `mat4 inv_view_proj` (64 B) + `vec4 soft_near_far`
/// (16 B). Same 80-byte payload the chunk pipeline uses (chunk uses 96 B with
/// additional headroom) — fits inside the chunk layout's 96-B range.
pub(super) fn create_particle_pipeline_layout(
    device: &ash::Device,
    chunk_set_layout: vk::DescriptorSetLayout,
    particle_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(96);
    let set_layouts = [chunk_set_layout, particle_set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_particle_pipeline_layout: {e:?}"))
}

/// Pool + set allocation for the per-frame particle input attachment.
/// Separate pool (no UBOs/samplers needed) — INPUT_ATTACHMENT type is enough.
#[allow(dead_code)]
pub(super) fn create_particle_descriptor_pool(
    device: &ash::Device,
    max_sets: usize,
) -> Result<vk::DescriptorPool> {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::INPUT_ATTACHMENT,
        descriptor_count: max_sets as u32,
    }];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(max_sets as u32);
    unsafe { device.create_descriptor_pool(&create_info, None) }
        .map_err(|e| anyhow!("create_particle_descriptor_pool: {e:?}"))
}

#[allow(dead_code)]
pub(super) fn allocate_particle_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![layout; count];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| anyhow!("allocate_particle_descriptor_sets: {e:?}"))
}

/// Bind the depth image view into the particle input attachment descriptor
/// set at binding 0. Layout must be DEPTH_STENCIL_READ_ONLY_OPTIMAL to match
/// the subpass 1 VkAttachmentReference.
pub(super) fn update_particle_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    depth_view: vk::ImageView,
) {
    let info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .image_view(depth_view);
    let infos = [info];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .image_info(&infos)];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Create a depth-only occlusion query pipeline for AABB proxy cubes.
///
/// Uses the shared `pipeline_layout` (96-byte push constants). No vertex
/// input (procedural cube from `gl_VertexIndex`), no colour writes, depth
/// test + write enabled. The fragment shader is a no-op.
pub(super) fn create_occlusion_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    let vert_code = spirv_to_u32(vs_spirv);
    let frag_code = spirv_to_u32(fs_spirv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("occlusion vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("occlusion frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    // No vertex input — the AABB shader generates cube vertices procedurally.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    // 8 vertices → 12 triangles (36 indices implicit via TRIANGLE_LIST +
    // gl_VertexIndex-based index buffer of 36 entries, OR 8 unique vertices
    // with an index buffer). We use an index buffer approach: 36 u16 indices.
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    // No backface culling — the AABB is a box, and we want all faces tested.
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);

    // Depth test + write: standard LESS comparison.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // No colour writes — only the depth test matters for occlusion queries.
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::empty());

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_occlusion_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

/// 36-index buffer for a unit cube (12 triangles). Used to draw AABB proxies
/// for occlusion queries via `cmd_draw_indexed`. The vertex shader generates
/// the 8 cube corners procedurally from `gl_VertexIndex`, so the index buffer
/// just selects which 3 of the 8 vertices form each triangle.
pub(super) fn create_aabb_index_buffer(
    device: &ash::Device,
    alloc: &crate::alloc::Alloc,
) -> Result<crate::buffer::GpuBuffer> {
    // 12 triangles (2 per face), each triangle referencing 3 of the 8 cube
    // corners. The vertex shader maps gl_VertexIndex 0..7 to cube corners
    // via bit fields. We define the 12 triangles with winding order such
    // that front faces are counter-clockwise (matching the pipeline).
    #[rustfmt::skip]
    let indices: [u16; 36] = [
        // +X face
        1, 3, 5,   5, 3, 7,
        // -X face
        0, 4, 2,   2, 4, 6,
        // +Y face
        2, 6, 3,   3, 6, 7,
        // -Y face
        0, 1, 4,   4, 1, 5,
        // +Z face
        4, 5, 6,   6, 5, 7,
        // -Z face
        0, 2, 1,   1, 2, 3,
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&indices);
    use crate::buffer::GpuBuffer;
    use ash::vk;

    // Upload via staging (small buffer, not worth a separate function).
    let size = bytes.len() as vk::DeviceSize;
    let staging = GpuBuffer::host_visible(
        device,
        alloc,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        "aabb_idx_staging",
    )?;
    let mut staging = staging;
    staging.upload(device, bytes)?;

    // 72 bytes is small enough to keep the index buffer host-visible
    // (no need for the device_local + staging copy dance). The earlier
    // `_ibo` device_local attempt was dead-code leaking because the
    // variable was bound to `_` and never destroyed — its `GpuBuffer`
    // Drop emitted a leak warning and the `gpu_allocator` allocation
    // reported "aabb_idx" as leaked on shutdown.
    staging.destroy(device, alloc);

    let ibo = GpuBuffer::host_visible(
        device,
        alloc,
        size,
        vk::BufferUsageFlags::INDEX_BUFFER,
        "aabb_idx",
    )?;
    let mut ibo = ibo;
    ibo.upload(device, bytes)?;
    Ok(ibo)
}

pub(super) fn create_shadow_layer_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    base_array_layer: u32,
) -> Result<vk::ImageView> {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer,
            layer_count: 1,
        });
    unsafe { device.create_image_view(&create_info, None) }
        .map_err(|e| anyhow!("create_shadow_layer_view: {e:?}"))
}
