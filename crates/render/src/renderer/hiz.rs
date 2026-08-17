//! Phase 3 — Hi-Z depth pyramid for GPU occlusion culling.
//!
//! A compute shader builds a min-depth pyramid from the previous frame's
//! resolved single-sample scene depth. `chunk_cull.comp` samples the pyramid
//! (a 1-frame-old representation of the scene) to reject chunks fully hidden
//! behind nearer geometry, which the GPU-driven path previously could not do
//! (its only cull was frustum-based).

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::alloc::Alloc;

/// Maximum pyramid mip levels (supports framebuffers up to 4096px tall).
pub const MAX_MIPS: u32 = 12;

pub struct HiZ {
    image: vk::Image,
    allocation: Option<Allocation>,
    /// STORAGE image views, one per mip level, for the builder's `imageStore`.
    mip_views: Vec<vk::ImageView>,
    /// SAMPLED view of the whole mip chain (used by the builder's `textureLod`
    /// reads and by the cull shader).
    sampled_view: vk::ImageView,
    sampler: vk::Sampler,
    mip_count: u32,
    extent: vk::Extent2D,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// One descriptor set per mip level (each binds a different destination
    /// storage-image view), so no descriptor is mutated while in use.
    sets: Vec<vk::DescriptorSet>,
}

impl HiZ {
    /// Compute the number of mip levels for a given framebuffer extent.
    /// The pyramid image is half-res at mip 0 (a 2x downsample of the
    /// full-res scene depth), so the chain runs from `extent / 2` down to 1x1.
    pub fn mip_count_for(extent: vk::Extent2D) -> u32 {
        let m = (extent.width / 2).min(extent.height / 2).max(1);
        (32 - m.leading_zeros()).min(MAX_MIPS).max(1)
    }

    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: &ash::Device,
        alloc: &Alloc,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        extent: vk::Extent2D,
        scene_depth_view: vk::ImageView,
        scene_depth_sampler: vk::Sampler,
    ) -> Result<Self> {
        use crate::texture::{begin_one_time, end_and_submit, transition_image_layout};
        let mip_count = Self::mip_count_for(extent);
        let image = create_pyramid_image(device, extent, mip_count)?;
        let allocation = {
            let reqs = unsafe { device.get_image_memory_requirements(image) };
            let a = alloc.allocate(&AllocationCreateDesc {
                name: "hiz_pyramid",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe { device.bind_image_memory(image, a.memory(), a.offset()) }
                .map_err(|e| anyhow!("hiz bind_image_memory: {e:?}"))?;
            Some(a)
        };
        // Initial layout so the first build's SHADER_READ_ONLY -> GENERAL
        // transition (and the first frame's cull sample) is well-defined.
        {
            let cmd = begin_one_time(device, command_pool)?;
            transition_image_layout(
                device,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                mip_count,
                1,
            );
            end_and_submit(device, command_pool, graphics_queue, cmd)?;
        }

        // Per-mip STORAGE views.
        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R32_SFLOAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: mip,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
            }
            .map_err(|e| anyhow!("hiz mip view {mip}: {e:?}"))?;
            mip_views.push(view);
        }
        let sampled_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: mip_count,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .map_err(|e| anyhow!("hiz sampled view: {e:?}"))?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(mip_count as f32),
                None,
            )
        }
        .map_err(|e| anyhow!("hiz sampler: {e:?}"))?;

        // Transition mip 0..N into GENERAL so the first build can write them.
        let set_layout = create_set_layout(device)?;
        let (pipeline, pipeline_layout) = create_pipeline(device, set_layout)?;
        let pool = create_descriptor_pool(device, mip_count)?;
        let mut sets = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let set = allocate_set(device, pool, set_layout)?;
            update_set(
                device,
                set,
                scene_depth_view,
                scene_depth_sampler,
                sampled_view,
                sampler,
                mip_views[mip as usize],
            );
            sets.push(set);
        }

        Ok(Self {
            image,
            allocation,
            mip_views,
            sampled_view,
            sampler,
            mip_count,
            extent,
            pipeline,
            pipeline_layout,
            set_layout,
            pool,
            sets,
        })
    }

    /// The SAMPLED view + sampler the cull shader reads.
    pub fn sampled_view(&self) -> vk::ImageView {
        self.sampled_view
    }
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Cull-shader parameters: `[mip0_w, mip0_h, mip_count, depth_bias]`.
    pub fn params(&self) -> [f32; 4] {
        let w = (self.extent.width / 2).max(1) as f32;
        let h = (self.extent.height / 2).max(1) as f32;
        [w, h, self.mip_count as f32, 0.005]
    }

    /// Record the pyramid build. The scene depth must already be in
    /// `SHADER_READ_ONLY_OPTIMAL` (it is, after `record_scene_opaque_copy`).
    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let sub = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: self.mip_count,
            base_array_layer: 0,
            layer_count: 1,
        };
        // WAR guard: the cull shader read this pyramid earlier in the frame
        // (previous build). Wait for those reads before writing, and move the
        // image into GENERAL for storage access.
        unsafe {
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.image)
                .subresource_range(sub)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        }
        // mip 0 = half-res downsample of scene depth; mip L = half of L-1.
        let mut dst_w = (self.extent.width / 2).max(1);
        let mut dst_h = (self.extent.height / 2).max(1);
        for mip in 0..self.mip_count {
            let pc = [dst_w, dst_h, mip, u32::from(mip == 0)];
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.sets[mip as usize]],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&pc),
                );
                device.cmd_dispatch(cmd, dst_w.div_ceil(8), dst_h.div_ceil(8), 1);
            }
            // Order mip write -> next mip's reads.
            if mip + 1 < self.mip_count {
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(self.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: self.mip_count,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            }
            dst_w = (dst_w / 2).max(1);
            dst_h = (dst_h / 2).max(1);
        }
        // Back to sampled layout for the next frame's cull.
        unsafe {
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.image)
                .subresource_range(sub)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            for v in self.mip_views.drain(..) {
                device.destroy_image_view(v, None);
            }
            device.destroy_image_view(self.sampled_view, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image(self.image, None);
        }
        self.pipeline = vk::Pipeline::null();
        self.pipeline_layout = vk::PipelineLayout::null();
        self.set_layout = vk::DescriptorSetLayout::null();
        self.pool = vk::DescriptorPool::null();
        self.sampled_view = vk::ImageView::null();
        self.sampler = vk::Sampler::null();
        self.image = vk::Image::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }
}

fn create_pyramid_image(
    device: &ash::Device,
    extent: vk::Extent2D,
    mip_count: u32,
) -> Result<vk::Image> {
    // The pyramid is half-res at mip 0, so the image's base extent is the
    // framebuffer extent divided by two. This keeps image mip level L equal
    // to logical pyramid level L (no offset).
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R32_SFLOAT)
        .extent(vk::Extent3D {
            width: (extent.width / 2).max(1),
            height: (extent.height / 2).max(1),
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    unsafe { device.create_image(&info, None) }.map_err(|e| anyhow!("hiz create_image: {e:?}"))
}

fn create_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        // scene depth (sampled)
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // pyramid (sampled, full mip chain)
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // destination mip storage image (re-bound per dispatch)
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| anyhow!("hiz set layout: {e:?}"))
}

fn create_pipeline(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(16); // ivec2 + 2 u32
    let set_layouts = [set_layout];
    let pl_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push));
    let pl = unsafe { device.create_pipeline_layout(&pl_info, None) }
        .map_err(|e| anyhow!("hiz pipeline layout: {e:?}"))?;
    let spv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/depth_pyramid.comp.spv")).to_vec();
    let code = super::spirv_to_u32(&spv);
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
    }
    .map_err(|e| anyhow!("hiz module: {e:?}"))?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pl);
    let result =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe { device.destroy_shader_module(module, None) };
    let pipelines = result.map_err(|(_p, e)| anyhow!("hiz compute pipeline: {e:?}"))?;
    Ok((pipelines.into_iter().next().expect("hiz pipeline"), pl))
}

fn create_descriptor_pool(device: &ash::Device, mip_count: u32) -> Result<vk::DescriptorPool> {
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2 * mip_count,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: mip_count,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&sizes)
        .max_sets(mip_count);
    unsafe { device.create_descriptor_pool(&info, None) }.map_err(|e| anyhow!("hiz pool: {e:?}"))
}

fn allocate_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    let layouts = [layout];
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|e| anyhow!("hiz set alloc: {e:?}"))?;
    Ok(sets[0])
}

fn update_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    scene_depth_view: vk::ImageView,
    scene_depth_sampler: vk::Sampler,
    pyramid_view: vk::ImageView,
    pyramid_sampler: vk::Sampler,
    mip0_view: vk::ImageView,
) {
    let scene_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(scene_depth_view)
        .sampler(scene_depth_sampler)];
    // The builder reads the pyramid via `textureLod` while the image is in
    // GENERAL layout (it is being written by the same build), so the sampled
    // binding must declare GENERAL — the cull shader has its own binding that
    // declares SHADER_READ_ONLY for after the build.
    let pyramid_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(pyramid_view)
        .sampler(pyramid_sampler)];
    let mip_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(mip0_view)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&scene_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&pyramid_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&mip_info),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}
