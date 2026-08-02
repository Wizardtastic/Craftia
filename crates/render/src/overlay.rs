//! 3D wireframe overlay pipeline for brush previews and selection boxes.
//!
//! Renders colored line segments in world space, overlaid on top of
//! existing geometry. Uses depth test ON but depth write OFF so the
//! overlay is always visible but respects depth ordering.

use anyhow::{anyhow, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::renderer::spirv_to_u32;

/// Overlay vertex: 16 bytes. World-space position + per-vertex RGBA color.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct OverlayVertex {
    pub pos: [f32; 3],
    pub color: [u8; 4],
}

/// Push constants for the overlay pipeline (64 bytes = mat4 view_proj).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OverlayPushConstants {
    pub view_proj: [[f32; 4]; 4],
}

/// Overlay line data collected per-frame.
#[derive(Clone, Debug, Default)]
pub struct OverlayLine {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [u8; 4],
}

/// All overlay data for the current frame.
#[derive(Clone, Debug, Default)]
pub struct OverlayData {
    pub lines: Vec<OverlayLine>,
}

impl OverlayData {
    /// Convert lines to vertices (2 vertices per line).
    pub fn to_vertices(&self) -> Vec<OverlayVertex> {
        let mut verts = Vec::with_capacity(self.lines.len() * 2);
        for line in &self.lines {
            verts.push(OverlayVertex {
                pos: line.a,
                color: line.color,
            });
            verts.push(OverlayVertex {
                pos: line.b,
                color: line.color,
            });
        }
        verts
    }
}

/// Create the overlay pipeline layout.
pub fn create_overlay_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64); // mat4 view_proj
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_overlay_pipeline_layout: {e:?}"))
}

/// Create the overlay graphics pipeline (LINE_LIST topology).
pub fn create_overlay_pipeline(
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
    .map_err(|e| anyhow!("overlay vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("overlay frag shader: {e:?}"))?;

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

    // Vertex layout: pos(12) + color(4) = 16 bytes.
    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<OverlayVertex>() as u32)
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
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(12),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::LINE_LIST)
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

    // Depth test ON (so lines behind walls are hidden), depth write OFF
    // (so overlay doesn't occlude other passes).
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Alpha blending.
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
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_overlay_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}
