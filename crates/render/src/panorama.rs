//! Panorama cubemap loader: loads 6 PNG face images into a Vulkan cubemap.
//!
//! Face ordering matches Minecraft's convention:
//!   0 = +X (right), 1 = -X (left), 2 = +Y (top), 3 = -Y (bottom),
//!   4 = +Z (front), 5 = -Z (back).

use ash::vk;
use gpu_allocator::MemoryLocation;
use std::path::Path;

use crate::alloc::Alloc;
use crate::buffer::GpuBuffer;
use crate::texture::{begin_one_time, end_and_submit, transition_image_layout};

/// Panorama cubemap: image + cubemap view + sampler. `None` fields mean the
/// panorama textures were not found and the sky gradient should be used instead.
pub struct Panorama {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub loaded: bool,
}

impl Panorama {
    /// Attempt to load 6 PNG faces from `dir`. Returns `Self { loaded: false, .. }`
    /// if the directory or any file is missing — the renderer falls back to the
    /// sky gradient gracefully.
    pub fn load(
        device: &ash::Device,
        alloc: &Alloc,
        pool: vk::CommandPool,
        queue: vk::Queue,
        dir: &Path,
    ) -> Self {
        let names = [
            "panorama_0.png",
            "panorama_1.png",
            "panorama_2.png",
            "panorama_3.png",
            "panorama_4.png",
            "panorama_5.png",
        ];

        // Try to load all 6 faces.
        let mut faces: Vec<Vec<u8>> = Vec::new();
        let mut face_width = 0u32;
        let mut face_height = 0u32;

        for name in &names {
            let path = dir.join(name);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    match image::load_from_memory(&bytes) {
                        Ok(img) => {
                            let rgba = img.to_rgba8();
                            let w = rgba.width();
                            let h = rgba.height();
                            if face_width == 0 {
                                face_width = w;
                                face_height = h;
                            } else if w != face_width || h != face_height {
                                log::warn!(
                                "panorama: {} has size {}x{}, expected {}x{} — skipping panorama",
                                name, w, h, face_width, face_height
                            );
                                return Self::empty(device);
                            }
                            faces.push(rgba.into_raw());
                        }
                        Err(e) => {
                            log::warn!("panorama: failed to decode {name}: {e}");
                            return Self::empty(device);
                        }
                    }
                }
                Err(_) => {
                    log::debug!("panorama: {path:?} not found, skipping panorama");
                    return Self::empty(device);
                }
            }
        }

        if faces.len() != 6 {
            return Self::empty(device);
        }

        // Create cubemap image.
        let format = vk::Format::R8G8B8A8_UNORM;
        let extent = vk::Extent3D {
            width: face_width,
            height: face_height,
            depth: 1,
        };

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(6)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);
        let image = match unsafe { device.create_image(&create_info, None) } {
            Ok(i) => i,
            Err(e) => {
                log::warn!("panorama: create_image failed: {e:?}");
                return Self::empty(device);
            }
        };

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "panorama_cubemap",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                log::warn!("panorama: alloc failed: {e}");
                unsafe { device.destroy_image(image, None) };
                return Self::empty(device);
            }
        };
        unsafe {
            if let Err(e) =
                device.bind_image_memory(image, allocation.memory(), allocation.offset())
            {
                log::warn!("panorama: bind failed: {e}");
                alloc.free(allocation);
                device.destroy_image(image, None);
                return Self::empty(device);
            }
        }

        // Upload all 6 faces via staging buffer.
        let face_bytes = (face_width * face_height * 4) as vk::DeviceSize;
        let total_bytes = face_bytes * 6;
        let mut staging = match GpuBuffer::host_visible(
            device,
            alloc,
            total_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "panorama_staging",
        ) {
            Ok(b) => b,
            Err(e) => {
                log::warn!("panorama: staging alloc failed: {e}");
                alloc.free(allocation);
                unsafe { device.destroy_image(image, None) };
                return Self::empty(device);
            }
        };

        // Copy all face data into the staging buffer.
        let mut all_data = Vec::with_capacity(total_bytes as usize);
        for face in &faces {
            all_data.extend_from_slice(face);
        }
        if let Err(e) = staging.upload(device, &all_data) {
            log::warn!("panorama: staging upload failed: {e}");
            staging.destroy(device, alloc);
            alloc.free(allocation);
            unsafe { device.destroy_image(image, None) };
            return Self::empty(device);
        }

        // Record copy commands.
        let cmd = match begin_one_time(device, pool) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("panorama: begin_one_time failed: {e}");
                staging.destroy(device, alloc);
                alloc.free(allocation);
                unsafe { device.destroy_image(image, None) };
                return Self::empty(device);
            }
        };

        // Transition all layers to TRANSFER_DST.
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            6,
        );

        // Copy each face.
        for face_idx in 0..6u32 {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(face_idx as vk::DeviceSize * face_bytes)
                .buffer_row_length(face_width)
                .buffer_image_height(face_height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: face_idx,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(extent);
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
        }

        // Transition to SHADER_READ.
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            6,
        );

        if let Err(e) = end_and_submit(device, pool, queue, cmd) {
            log::warn!("panorama: submit failed: {e}");
            staging.destroy(device, alloc);
            alloc.free(allocation);
            unsafe { device.destroy_image(image, None) };
            return Self::empty(device);
        }

        staging.destroy(device, alloc);

        // Create cubemap image view.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });
        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(v) => v,
            Err(e) => {
                log::warn!("panorama: create_image_view failed: {e}");
                alloc.free(allocation);
                unsafe { device.destroy_image(image, None) };
                return Self::empty(device);
            }
        };

        // Create sampler (linear filtering, clamp to edge).
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .unnormalized_coordinates(false);
        let sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
            Ok(s) => s,
            Err(e) => {
                log::warn!("panorama: create_sampler failed: {e}");
                unsafe { device.destroy_image_view(view, None) };
                alloc.free(allocation);
                unsafe { device.destroy_image(image, None) };
                return Self::empty(device);
            }
        };

        log::info!(
            "panorama: loaded {}x{} cubemap from {}",
            face_width,
            face_height,
            dir.display()
        );

        Self {
            image,
            view,
            sampler,
            allocation: Some(allocation),
            loaded: true,
        }
    }

    /// Create an empty (unloaded) panorama placeholder.
    fn empty(_device: &ash::Device) -> Self {
        Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            allocation: None,
            loaded: false,
        }
    }

    /// Destroy GPU resources.
    pub fn destroy(&mut self, device: &ash::Device, alloc: &Alloc) {
        if self.loaded {
            unsafe {
                device.destroy_sampler(self.sampler, None);
                device.destroy_image_view(self.view, None);
                device.destroy_image(self.image, None);
            }
            if let Some(a) = self.allocation.take() {
                alloc.free(a);
            }
            self.loaded = false;
            self.image = vk::Image::null();
            self.view = vk::ImageView::null();
            self.sampler = vk::Sampler::null();
        }
    }
}
