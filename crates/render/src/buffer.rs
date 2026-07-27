//! Buffer and image creation helpers backed by the shared allocator.

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::alloc::Alloc;

/// An owned buffer + its memory allocation. Drops free both.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Create a device-local buffer with the given usage flags (no host access).
    pub fn device_local(
        device: &ash::Device,
        alloc: &Alloc,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&create_info, None) }
            .map_err(|e| anyhow!("create_buffer failed: {e:?}"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_buffer_memory failed: {e:?}"))?;
        }
        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Create a host-visible (CPU-mapped) buffer for staging / uniform data.
    pub fn host_visible(
        device: &ash::Device,
        alloc: &Alloc,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&create_info, None) }
            .map_err(|e| anyhow!("create_buffer failed: {e:?}"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_buffer_memory failed: {e:?}"))?;
        }
        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Write bytes into a host-visible buffer's mapped memory.
    pub fn upload(&mut self, data: &[u8]) -> Result<()> {
        let allocation = self
            .allocation
            .as_mut()
            .ok_or_else(|| anyhow!("buffer has no allocation"))?;
        let slice = allocation
            .mapped_slice_mut()
            .ok_or_else(|| anyhow!("buffer memory is not host-visible"))?;
        if data.len() > slice.len() {
            return Err(anyhow!(
                "upload overflows buffer: {} > {}",
                data.len(),
                slice.len()
            ));
        }
        slice[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Map the buffer as a mutable byte slice (for uniform updates).
    pub fn mapped_slice_mut(&mut self) -> Result<&mut [u8]> {
        let allocation = self
            .allocation
            .as_mut()
            .ok_or_else(|| anyhow!("buffer has no allocation"))?;
        allocation
            .mapped_slice_mut()
            .ok_or_else(|| anyhow!("buffer memory is not host-visible"))
    }
}

impl GpuBuffer {
    /// Destroy the Vulkan buffer and free its allocation. Call before drop.
    pub fn destroy(mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        self.buffer = vk::Buffer::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }

    /// Destroy in place (for fields that can't be moved out, e.g. in `Drop`).
    /// Leaves `self` hollow (null buffer, no allocation).
    pub fn destroy_in_place(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        self.buffer = vk::Buffer::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            log::warn!("GpuBuffer dropped without calling destroy_in_place — GPU resource leaked");
        }
    }
}

/// An owned image + its memory allocation.
pub struct GpuImage {
    pub image: vk::Image,
    pub allocation: Option<Allocation>,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
}

impl GpuImage {
    pub fn destroy(mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        self.image = vk::Image::null();
        self.view = vk::ImageView::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }

    /// Destroy in place (for fields that can't be moved out, e.g. in `Drop`).
    pub fn destroy_in_place(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        self.image = vk::Image::null();
        self.view = vk::ImageView::null();
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }

    /// Create a depth image suitable for the render pass's depth attachment.
    ///
    /// The single-sample depth image doubles as the post-pass SSAO source and
    /// as the transfer source of the per-frame `scene_opaque_depth` copy used
    /// by the water/glass SSR ray-march, so it needs `SAMPLED | TRANSFER_SRC`
    /// in addition to `DEPTH_STENCIL_ATTACHMENT`. When MSAA is active the main
    /// render pass resolves its multisampled depth into this image via
    /// `VkSubpassDescriptionDepthStencilResolve` (see pipeline.rs).
    pub fn depth(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self> {
        Self::depth_with_usage(
            device,
            alloc,
            extent,
            format,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
    }

    /// Create a depth image with an explicit sample count (for MSAA).
    pub fn depth_msaa(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        Self::depth_with_usage(
            device,
            alloc,
            extent,
            format,
            samples,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    /// Shared depth-image constructor with caller-chosen usage flags.
    fn depth_with_usage(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("create_image failed: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name: "depth",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_image_memory failed: {e:?}"))?;
        }
        let view = create_image_view(device, image, format, vk::ImageAspectFlags::DEPTH)?;
        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            format,
            extent: extent3d,
        })
    }

    pub fn color_attachment(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        name: &str,
    ) -> Result<Self> {
        Self::color_attachment_msaa(device, alloc, extent, format, vk::SampleCountFlags::TYPE_1, name)
    }

    /// Create a color attachment image with an explicit sample count (for MSAA).
    /// Unlike the resolve target, MSAA color attachments are transient (no SAMPLED/TRANSFER_SRC).
    pub fn color_attachment_msaa(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        name: &str,
    ) -> Result<Self> {
        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let usage = if samples == vk::SampleCountFlags::TYPE_1 {
            // Resolve target: needs SAMPLED + TRANSFER_SRC for post pass + capture.
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            // MSAA transient: only COLOR_ATTACHMENT (never sampled directly).
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("create_image failed: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_image_memory failed: {e:?}"))?;
        }
        let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            format,
            extent: extent3d,
        })
    }    pub fn depth_array(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        array_layers: u32,
        name: &str,
    ) -> Result<Self> {
        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("create_image failed: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_image_memory failed: {e:?}"))?;
        }
        let view = create_image_view_array(
            device,
            image,
            format,
            vk::ImageAspectFlags::DEPTH,
            array_layers,
        )?;

        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            format,
            extent: extent3d,
        })
    }

    /// Create a `scene_opaque_color` image — a single-sample colour image used
    /// as the destination of a `vkCmdCopyImage` (`TRANSFER_DST`) and the source
    /// of chunk-fragment-shader absorption/refraction sampling (`SAMPLED`).
    ///
    /// Sized to the offscreen render target so a full-resolution copy fits
    /// without downscaling. NOT a render-target attachment (so no
    /// `vkCmdPipelineBarrier` to `COLOR_ATTACHMENT_OPTIMAL` is needed — just
    /// `SHADER_READ_ONLY_OPTIMAL`).
    pub fn scene_opaque(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        name: &str,
    ) -> Result<Self> {
        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("create_image (scene_opaque) failed: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_image_memory failed: {e:?}"))?;
        }
        let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            format,
            extent: extent3d,
        })

        // (delete the orphan with-statement comment marker so future readers
        //  don't try to "fix" a missing closing brace on this scope).
    }

    /// Create a `scene_opaque_depth` image — the single-sample depth companion
    /// to `scene_opaque_color`. Once per frame (right after the main render
    /// pass, alongside the color copy) the resolved single-sample scene depth
    /// is `vkCmdCopyImage`'d into this image, which the chunk fragment shader
    /// then samples (binding 7) for the water/glass SSR ray-march and the
    /// water-column Beer absorption depth. Same extent and depth format as the
    /// main pass depth attachment so the copy is a plain texel-for-texel blit.
    pub fn scene_opaque_depth(
        device: &ash::Device,
        alloc: &Alloc,
        extent: vk::Extent2D,
        format: vk::Format,
        name: &str,
    ) -> Result<Self> {
        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("create_image (scene_opaque_depth) failed: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("bind_image_memory failed: {e:?}"))?;
        }
        let view = create_image_view(device, image, format, vk::ImageAspectFlags::DEPTH)?;
        Ok(Self {
            image,
            allocation: Some(allocation),
            view,
            format,
            extent: extent3d,
        })
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        if self.image != vk::Image::null() {
            log::warn!("GpuImage dropped without calling destroy — GPU resource leaked");
        }
    }
}

/// Create a simple 2D image view.
pub fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { device.create_image_view(&create_info, None) }
        .map_err(|e| anyhow!("create_image_view failed: {e:?}"))
}

pub fn create_image_view_array(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    layer_count: u32,
) -> Result<vk::ImageView> {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count,
        });
    unsafe { device.create_image_view(&create_info, None) }
        .map_err(|e| anyhow!("create_image_view failed: {e:?}"))
}
