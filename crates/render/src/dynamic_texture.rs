//! Dynamic atlas texture: a fixed-size GPU texture that can be re-uploaded
//! without destroying/recreating the image, view, or sampler.
//!
//! Used for the minimap framebuffer — the CPU-side RGBA buffer is rebuilt
//! periodically and uploaded to the GPU via a staging buffer copy.

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::alloc::Alloc;
use crate::buffer::GpuBuffer;
use crate::texture::{begin_one_time, end_and_submit, transition_image_layout};

/// A GPU texture that supports repeated pixel-data uploads at a fixed resolution.
pub struct DynamicAtlasTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    staging: GpuBuffer,
    width: u32,
    height: u32,
}

impl DynamicAtlasTexture {
    /// Create a new dynamic texture with the given dimensions (no initial data).
    /// The image is created in `SHADER_READ_ONLY_OPTIMAL` layout.
    pub fn new(
        device: &ash::Device,
        alloc: &Alloc,
        pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let format = vk::Format::R8G8B8A8_UNORM;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("dynamic_tex create_image: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "dynamic_atlas",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("dynamic_atlas bind: {e:?}"))?;
        }

        // Transition to SHADER_READ_ONLY_OPTIMAL (initial empty state).
        let cmd = begin_one_time(device, pool)?;
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );
        end_and_submit(device, pool, queue, cmd)?;

        let view = crate::buffer::create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .map_err(|e| anyhow!("dynamic_atlas sampler: {e:?}"))?;

        // Pre-allocate staging buffer for the full image.
        let staging_size = (width * height * 4) as vk::DeviceSize;
        let staging = GpuBuffer::host_visible(
            device,
            alloc,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "dynamic_atlas_staging",
        )?;

        Ok(Self {
            image,
            view,
            sampler,
            allocation: Some(allocation),
            staging,
            width,
            height,
        })
    }

    /// Upload new RGBA pixel data to the GPU texture.
    /// `data.len()` must equal `width * height * 4`.
    /// Transitions: SHADER_READ → TRANSFER_DST → copy → SHADER_READ.
    pub fn upload(
        &mut self,
        data: &[u8],
        device: &ash::Device,
        pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<()> {
        assert!(
            data.len() == (self.width * self.height * 4) as usize,
            "data size mismatch: expected {} got {}",
            self.width * self.height * 4,
            data.len()
        );

        self.staging.upload(device, data)?;

        let cmd = begin_one_time(device, pool)?;

        // Transition to TRANSFER_DST.
        transition_image_layout(
            device,
            cmd,
            self.image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );

        let extent = vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: 1,
        };
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(self.width)
            .buffer_image_height(self.height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(extent);
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                self.staging.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // Transition back to SHADER_READ.
        transition_image_layout(
            device,
            cmd,
            self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );

        end_and_submit(device, pool, queue, cmd)?;
        Ok(())
    }

    /// Destroy GPU resources (takes ownership).
    pub fn destroy(mut self, device: &ash::Device, alloc: &Alloc) {
        self.staging.destroy(device, alloc);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }

    /// Destroy in place (for use in Drop contexts where the value can't be moved).
    pub fn destroy_in_place(&mut self, device: &ash::Device, alloc: &Alloc) {
        self.staging.destroy_in_place(device, alloc);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            self.sampler = vk::Sampler::null();
            self.view = vk::ImageView::null();
            self.image = vk::Image::null();
        }
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }
}
