//! GPU voxel grid for ray-traced lighting.
//!
//! A single `R8_UINT` 3D texture holds the block ID of every voxel in a
//! world-aligned region around the player. The chunk fragment shader ray
//! marches it (Amanatides & Woo DDA) to evaluate sun/moon shadows, ambient
//! occlusion, sky GI, reflections and emissive block light - all per pixel,
//! with nothing baked into the meshes.
//!
//! Storage is 1 byte per voxel and the grid is uploaded incrementally: each
//! 16³ chunk is a `vkCmdCopyBufferToImage` from a host-visible staging buffer.
//! When the player travels far enough the grid is re-centred and repopulated.

use anyhow::{anyhow, Result};
use ash::vk;
use glam::IVec3;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::alloc::Alloc;
use crate::texture::{begin_one_time, end_and_submit, transition_image_layout};

/// Grid extent in blocks. X/Z are centred on the player's chunk; Y covers the
/// full world height so vertical rays never leave the volume.
pub const EXTENT: [u32; 3] = [192, 256, 192];

/// Bytes in one 16³ chunk of block IDs.
pub const CHUNK_BYTES: usize = 16 * 16 * 16;

pub struct VoxelGrid {
    image: vk::Image,
    allocation: Option<Allocation>,
    view: vk::ImageView,
    sampler: vk::Sampler,
    origin: IVec3,
}

impl VoxelGrid {
    pub fn new(
        device: &ash::Device,
        alloc: &Alloc,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Result<Self> {
        let image_create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(vk::Format::R8_UINT)
            .extent(vk::Extent3D {
                width: EXTENT[0],
                height: EXTENT[1],
                depth: EXTENT[2],
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&image_create, None) }
            .map_err(|e| anyhow!("create voxel grid image: {e:?}"))?;

        let allocation = {
            let reqs = unsafe { device.get_image_memory_requirements(image) };
            let a = alloc.allocate(&AllocationCreateDesc {
                name: "voxel_grid",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe { device.bind_image_memory(image, a.memory(), a.offset()) }
                .map_err(|e| anyhow!("bind voxel grid memory: {e:?}"))?;
            Some(a)
        };

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_3D)
                    .format(vk::Format::R8_UINT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .map_err(|e| anyhow!("voxel grid view: {e:?}"))?;

        let sampler = unsafe {
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
        .map_err(|e| anyhow!("voxel grid sampler: {e:?}"))?;

        // Keep the grid in GENERAL for its whole lifetime: it is written by
        // transfer operations and read by fragment shaders, and GENERAL makes
        // both legal without per-frame layout transitions.
        {
            let cmd = begin_one_time(device, command_pool)?;
            transition_image_layout(
                device,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            // Clear once so reads before the first chunk upload see air.
            clear_image(device, cmd, image);
            end_and_submit(device, command_pool, graphics_queue, cmd)?;
        }

        Ok(Self {
            image,
            allocation,
            view,
            sampler,
            origin: IVec3::new(-(EXTENT[0] as i32) / 2, 0, -(EXTENT[2] as i32) / 2),
        })
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub fn origin(&self) -> IVec3 {
        self.origin
    }

    pub fn set_origin(&mut self, origin: IVec3) {
        self.origin = origin;
    }

    /// True when a 16³ chunk at `world_origin` lies entirely inside the grid.
    pub fn contains_chunk(&self, world_origin: IVec3) -> bool {
        let rel = world_origin - self.origin;
        rel.x >= 0
            && rel.y >= 0
            && rel.z >= 0
            && rel.x + 16 <= EXTENT[0] as i32
            && rel.y + 16 <= EXTENT[1] as i32
            && rel.z + 16 <= EXTENT[2] as i32
    }

    /// Blocking barrier: previous shader reads -> transfer writes.
    pub fn begin_writes(device: &ash::Device, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Barrier: transfer writes -> fragment shader reads.
    pub fn end_writes(device: &ash::Device, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn record_clear(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        clear_image(device, cmd, self.image);
    }

    /// Copy one 16³ chunk of block IDs from `staging` into the grid. The
    /// caller must have already called [`Self::begin_writes`] and must call
    /// [`Self::end_writes`] after all copies.
    pub fn record_upload(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        staging: vk::Buffer,
        staging_offset: u64,
        world_origin: IVec3,
    ) {
        if !self.contains_chunk(world_origin) {
            return;
        }
        let offset = world_origin - self.origin;
        let region = vk::BufferImageCopy::default()
            .buffer_offset(staging_offset)
            .buffer_row_length(16)
            .buffer_image_height(16)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D {
                x: offset.x,
                y: offset.y,
                z: offset.z,
            })
            .image_extent(vk::Extent3D {
                width: 16,
                height: 16,
                depth: 16,
            });
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging,
                self.image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        self.sampler = vk::Sampler::null();
        self.view = vk::ImageView::null();
        self.image = vk::Image::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }
}

fn clear_image(device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image) {
    let clear = vk::ClearColorValue {
        uint32: [0, 0, 0, 0],
    };
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    unsafe {
        device.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &clear, &[range]);
    }
}
