//! Entity rendering: pipeline creation, vertex layout, and draw data
//! for entities with a Mesh component.

use anyhow::{anyhow, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::renderer::spirv_to_u32;

/// Entity render data collected from the ECS after the schedule runs.
/// One entry per entity with a Mesh + Transform component.
#[derive(Clone, Copy, Debug)]
pub struct EntityRenderData {
    pub pos: glam::Vec3,
    pub rot: glam::Quat,
    pub tile: u32,
    pub billboard: bool,
    pub half_size: f32,
    /// Whether this entity uses alpha blending (rendered back-to-front).
    pub transparent: bool,
    /// Whether this is a held item (uses ALWAYS depth compare).
    pub held_item: bool,
}

/// Push constants for the entity pipeline (80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EntityPushConstants {
    pub model: [[f32; 4]; 4], // 64 bytes
    pub tile: u32,            // 4 bytes
    pub half_size: f32,       // 4 bytes
    pub billboard: u32,       // 4 bytes
    pub _pad: [u32; 2],       // 8 bytes
    // total: 84 bytes, but we round to 80 with vec4 alignment
}

/// Create the entity pipeline layout. Uses the same descriptor set
/// layout as chunks (camera UBO + atlas sampler + fog UBO at
/// bindings 0, 1, 2) but different push constants (80 bytes for
/// model matrix + metadata).
pub fn create_entity_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80); // mat4(64) + uint(4) + float(4) + uint(4) + pad(4) = 80
    let set_layouts = [set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_entity_pipeline_layout: {e:?}"))
}

/// Create an entity graphics pipeline with configurable depth settings.
///
/// `depth_compare_op` and `depth_write_enable` control how the entity
/// interacts with the depth buffer:
/// - World entities: `LESS_OR_EQUAL`, write off (on top of chunks)
/// - Held items: `ALWAYS`, write off (always visible, never clips through walls)
pub fn create_entity_pipeline_with_depth(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    depth_compare_op: vk::CompareOp,
    depth_write_enable: bool,
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
    .map_err(|e| anyhow!("entity vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("entity frag shader: {e:?}"))?;

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

    // Same vertex layout as chunks (32 bytes: pos + uv + light + tile + light_color).
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

    // No backface culling (billboards face camera regardless).
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(depth_write_enable)
        .depth_compare_op(depth_compare_op)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Alpha blending for transparent entity textures.
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
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_entity_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

/// Create the standard entity pipeline (LESS_OR_EQUAL depth, no write).
pub fn create_entity_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    create_entity_pipeline_with_depth(
        device,
        render_pass,
        layout,
        vs_spirv,
        fs_spirv,
        vk::CompareOp::LESS_OR_EQUAL,
        false,
        msaa_samples,
    )
}

/// Create the held-item entity pipeline (ALWAYS depth compare, no write).
/// Held items always render on top of world geometry to prevent clipping
/// through walls when the player is close to a surface.
pub fn create_held_item_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_spirv: &[u8],
    fs_spirv: &[u8],
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    create_entity_pipeline_with_depth(
        device,
        render_pass,
        layout,
        vs_spirv,
        fs_spirv,
        vk::CompareOp::ALWAYS,
        false,
        msaa_samples,
    )
}

/// Build a unit quad (2 triangles, 6 vertices) for billboard/cube rendering.
/// Each vertex has pos in [-0.5, 0.5], uv in [0, 1], light = 1.0, tile = 0.
pub fn unit_quad_vertices() -> Vec<crate::Vertex> {
    let tile = 0u32;
    let light = 1.0f32;
    let white = 0xFFFFFFFFu32; // white light color (no tint)
    vec![
        crate::Vertex { pos: [-0.5, -0.5, 0.0], uv: [0.0, 1.0], light, tile, light_color: white },
        crate::Vertex { pos: [ 0.5, -0.5, 0.0], uv: [1.0, 1.0], light, tile, light_color: white },
        crate::Vertex { pos: [ 0.5,  0.5, 0.0], uv: [1.0, 0.0], light, tile, light_color: white },
        crate::Vertex { pos: [-0.5, -0.5, 0.0], uv: [0.0, 1.0], light, tile, light_color: white },
        crate::Vertex { pos: [ 0.5,  0.5, 0.0], uv: [1.0, 0.0], light, tile, light_color: white },
        crate::Vertex { pos: [-0.5,  0.5, 0.0], uv: [0.0, 0.0], light, tile, light_color: white },
    ]
}

/// Create the pipeline layout for skinned entity rendering.
/// Uses two descriptor set layouts:
///   Set 0: camera UBO + atlas + fog + shadow (shared with chunk pipeline)
///   Set 1: joint matrices UBO
pub fn create_skinned_entity_pipeline_layout(
    device: &ash::Device,
    set_layout_0: vk::DescriptorSetLayout,
    set_layout_1: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80);
    let set_layouts = [set_layout_0, set_layout_1];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_skinned_entity_pipeline_layout: {e:?}"))
}

/// Create the skinned entity graphics pipeline.
/// Uses two vertex buffer bindings:
///   Binding 0: Vertex (28 bytes: pos + uv + light + tile)
///   Binding 1: SkinnedVertex (32 bytes: joint_indices + joint_weights)
pub fn create_skinned_entity_pipeline(
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
    .map_err(|e| anyhow!("skinned entity vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("skinned entity frag shader: {e:?}"))?;

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

    // Two vertex bindings:
    // Binding 0: Vertex (32 bytes)
    // Binding 1: SkinnedVertex (32 bytes)
    let vertex_bindings = [
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<crate::Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX),
        vk::VertexInputBindingDescription::default()
            .binding(1)
            .stride(std::mem::size_of::<crate::SkinnedVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX),
    ];

    // Attributes from binding 0 (same as regular entity)
    let vertex_attrs = [
        // location 0: pos (vec3)
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        // location 1: uv (vec2)
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
        // location 2: light (float)
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
        // location 3: tile (uint)
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_UINT)
            .offset(24),
        // location 4: light_color (vec4 packed as RGBA8) from binding 0
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(4)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(28),
        // location 5: joint_indices (uvec4) from binding 1
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(5)
            .format(vk::Format::R32G32B32A32_UINT)
            .offset(0),
        // location 6: joint_weights (vec4) from binding 1
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(6)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16),
    ];

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
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
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
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_skinned_entity_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

/// Create the descriptor set layout for the joint matrices UBO.
/// Set 1, binding 0: JointMatrices (64 × mat4 = 4096 bytes).
pub fn create_joint_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_joint_descriptor_set_layout: {e:?}"))
}

/// Create the descriptor set layout for the tile remap UBO.
/// Set 1, binding 0: TileRemap (256 × u32 = 1024 bytes).
pub fn create_tile_remap_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_tile_remap_descriptor_set_layout: {e:?}"))
}
