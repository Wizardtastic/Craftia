//! Renderer initialization helpers, extracted from `Renderer::new` to keep the
//! main module focused on per-frame logic.

use ash::vk;

/// All cached shader SPIR-V blobs loaded at startup from `build.rs`-compiled
/// defaults. The hot-reload path can update individual entries.
pub(crate) struct ShaderBlobs {
    pub chunk_vert: Vec<u8>,
    pub chunk_frag: Vec<u8>,
    pub ui_vert: Vec<u8>,
    pub ui_frag: Vec<u8>,
    pub sky_vert: Vec<u8>,
    pub sky_frag: Vec<u8>,
    pub post_vert: Vec<u8>,
    pub post_frag: Vec<u8>,
    pub entity_vert: Vec<u8>,
    pub entity_frag: Vec<u8>,
    pub particle_vert: Vec<u8>,
    pub particle_frag: Vec<u8>,
    pub overlay_vert: Vec<u8>,
    pub overlay_frag: Vec<u8>,
}

/// Load all built-in shader SPIR-V blobs from the `build.rs` output directory.
pub(crate) fn load_shader_blobs() -> ShaderBlobs {
    ShaderBlobs {
        chunk_vert: include_bytes!(concat!(env!("OUT_DIR"), "/chunk.vert.spv")).to_vec(),
        chunk_frag: include_bytes!(concat!(env!("OUT_DIR"), "/chunk.frag.spv")).to_vec(),
        ui_vert: include_bytes!(concat!(env!("OUT_DIR"), "/ui.vert.spv")).to_vec(),
        ui_frag: include_bytes!(concat!(env!("OUT_DIR"), "/ui.frag.spv")).to_vec(),
        sky_vert: include_bytes!(concat!(env!("OUT_DIR"), "/sky.vert.spv")).to_vec(),
        sky_frag: include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv")).to_vec(),
        post_vert: include_bytes!(concat!(env!("OUT_DIR"), "/post.vert.spv")).to_vec(),
        post_frag: include_bytes!(concat!(env!("OUT_DIR"), "/post.frag.spv")).to_vec(),
        entity_vert: include_bytes!(concat!(env!("OUT_DIR"), "/entity.vert.spv")).to_vec(),
        entity_frag: include_bytes!(concat!(env!("OUT_DIR"), "/entity.frag.spv")).to_vec(),
        particle_vert: include_bytes!(concat!(env!("OUT_DIR"), "/particle.vert.spv")).to_vec(),
        particle_frag: include_bytes!(concat!(env!("OUT_DIR"), "/particle.frag.spv")).to_vec(),
        overlay_vert: include_bytes!(concat!(env!("OUT_DIR"), "/overlay.vert.spv")).to_vec(),
        overlay_frag: include_bytes!(concat!(env!("OUT_DIR"), "/overlay.frag.spv")).to_vec(),
    }
}

/// Pick the offscreen scene colour format.
///
/// RGBA16F gives the lighting pipeline real HDR headroom: the sun disc, emissive
/// tiles and direct sunlight can exceed 1.0 and are tone-mapped once in the post
/// pass, instead of clipping in an 8-bit sRGB target. Falls back to the
/// swapchain format on the (very rare) device that cannot blend into it.
pub(crate) fn scene_color_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    fallback: vk::Format,
) -> vk::Format {
    let props = unsafe {
        instance.get_physical_device_format_properties(
            physical_device,
            vk::Format::R16G16B16A16_SFLOAT,
        )
    };
    let required = vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
        | vk::FormatFeatureFlags::SAMPLED_IMAGE;
    if props.optimal_tiling_features.contains(required) {
        vk::Format::R16G16B16A16_SFLOAT
    } else {
        log::warn!(
            "RGBA16F colour attachment unsupported; using swapchain format for the scene pass"
        );
        fallback
    }
}

/// Resolve the requested MSAA sample count against what the physical device
/// actually supports, clamping down to the next lower power of two.
pub(crate) fn resolve_msaa_samples(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    requested: u32,
) -> vk::SampleCountFlags {
    let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;
    let props = limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts;
    if requested >= 8 && props.contains(vk::SampleCountFlags::TYPE_8) {
        vk::SampleCountFlags::TYPE_8
    } else if requested >= 4 && props.contains(vk::SampleCountFlags::TYPE_4) {
        vk::SampleCountFlags::TYPE_4
    } else if requested >= 2 && props.contains(vk::SampleCountFlags::TYPE_2) {
        vk::SampleCountFlags::TYPE_2
    } else {
        vk::SampleCountFlags::TYPE_1
    }
}

/// Probe the device for Vulkan 1.2 depth-stencil-resolve support. Returns the
/// preferred resolve mode if available, `None` otherwise (pre-1.2 device).
pub(crate) fn probe_depth_resolve(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<vk::ResolveModeFlags> {
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    if props.api_version >= vk::make_api_version(0, 1, 2, 0) {
        let mut dsr_props = vk::PhysicalDeviceDepthStencilResolveProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut dsr_props);
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut props2);
        }
        let modes = dsr_props.supported_depth_resolve_modes;
        if modes.contains(vk::ResolveModeFlags::MIN) {
            Some(vk::ResolveModeFlags::MIN)
        } else if modes.contains(vk::ResolveModeFlags::SAMPLE_ZERO) {
            Some(vk::ResolveModeFlags::SAMPLE_ZERO)
        } else {
            None
        }
    } else {
        None
    }
}
