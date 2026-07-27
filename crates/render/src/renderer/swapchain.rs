use ash::vk;
use anyhow::{anyhow, Result};

pub(super) fn create_swapchain(
    _device: &ash::Device,
    swapchain_device: &ash::khr::swapchain::Device,
    surface: &ash::khr::surface::Instance,
    pdev: vk::PhysicalDevice,
    actual_surface: vk::SurfaceKHR,
    vsync: bool,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
    let caps = unsafe { surface.get_physical_device_surface_capabilities(pdev, actual_surface) }
        .map_err(|e| anyhow!("surface capabilities: {e:?}"))?;
    let formats = unsafe { surface.get_physical_device_surface_formats(pdev, actual_surface) }
        .map_err(|e| anyhow!("surface formats: {e:?}"))?;
    let present_modes =
        unsafe { surface.get_physical_device_surface_present_modes(pdev, actual_surface) }
            .map_err(|e| anyhow!("surface present modes: {e:?}"))?;

    let format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .ok_or_else(|| anyhow!("no surface formats"))?;

    let present_mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        present_modes
            .iter()
            .copied()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: 1280,
            height: 720,
        }
    };
    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 && image_count > caps.max_image_count {
        image_count = caps.max_image_count;
    }

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(actual_surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    let swapchain = unsafe { swapchain_device.create_swapchain(&create_info, None) }
        .map_err(|e| anyhow!("create_swapchain: {e:?}"))?;
    let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }
        .map_err(|e| anyhow!("get_swapchain_images: {e:?}"))?;
    Ok((swapchain, images, format.format, extent))
}

pub(super) fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let mut views = Vec::with_capacity(images.len());
    for &img in images {
        views.push(crate::buffer::create_image_view(
            device,
            img,
            format,
            vk::ImageAspectFlags::COLOR,
        )?);
    }
    Ok(views)
}

pub(super) fn create_framebuffer_with(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    attachments: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<vk::Framebuffer> {
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(attachments)
        .width(extent.width)
        .height(extent.height)
        .layers(1);
    unsafe { device.create_framebuffer(&create_info, None) }
        .map_err(|e| anyhow!("create_framebuffer_with: {e:?}"))
}
