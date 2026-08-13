use std::ffi::{c_char, CStr};

use anyhow::{anyhow, Result};
use ash::vk;
use ash::{Entry, Instance as AshInstance};
use raw_window_handle::RawDisplayHandle;

#[derive(Clone, Copy, Debug)]
pub(super) struct QueueFamilies {
    pub(super) graphics: u32,
    pub(super) present: u32,
}

pub(super) fn create_instance(
    entry: &Entry,
    display: RawDisplayHandle,
    validation: bool,
) -> Result<AshInstance> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"voxel")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"voxel-engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let surface_exts = ash_window::enumerate_required_extensions(display)
        .map_err(|e| anyhow!("enumerate_required_extensions: {e:?}"))?;
    let mut extension_names: Vec<*const c_char> = surface_exts.to_vec();
    if validation {
        extension_names.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
    }

    let layers: Vec<&CStr> = if validation {
        vec![c"VK_LAYER_KHRONOS_validation"]
    } else {
        vec![]
    };
    let layer_ptrs: Vec<*const c_char> = layers.iter().map(|l| l.as_ptr()).collect();

    let mut create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    create_info = create_info
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_ptrs);

    unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| anyhow!("create_instance: {e:?}"))
}

unsafe extern "system" fn debug_callback(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !data.is_null() {
        let data = &*data;
        let msg = unsafe { CStr::from_ptr(data.p_message) };
        log::warn!("[validation] {}", msg.to_string_lossy());
    }
    vk::FALSE
}

pub(super) fn create_debug_messenger(
    entry: &Entry,
    instance: &AshInstance,
) -> Result<vk::DebugUtilsMessengerEXT> {
    let du = ash::ext::debug_utils::Instance::new(entry, instance);
    let severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let ty = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(severity)
        .message_type(ty)
        .pfn_user_callback(Some(debug_callback));
    unsafe { du.create_debug_utils_messenger(&create_info, None) }
        .map_err(|e| anyhow!("create_debug_utils_messenger: {e:?}"))
}

pub(super) fn pick_physical_device(
    instance: &AshInstance,
    surface: &ash::khr::surface::Instance,
    actual_surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, QueueFamilies)> {
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| anyhow!("enumerate_physical_devices: {e:?}"))?;
    for pdev in physicals {
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        if props.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        if let Some(q) = find_queue_families(instance, surface, pdev, actual_surface) {
            return Ok((pdev, q));
        }
    }
    Err(anyhow!("no suitable Vulkan physical device found"))
}

pub(super) fn find_queue_families(
    instance: &AshInstance,
    surface: &ash::khr::surface::Instance,
    pdev: vk::PhysicalDevice,
    actual_surface: vk::SurfaceKHR,
) -> Option<QueueFamilies> {
    let props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
    let mut graphics = None;
    let mut present = None;
    for (i, q) in props.iter().enumerate() {
        if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
            graphics = Some(i as u32);
        }
        let supports =
            unsafe { surface.get_physical_device_surface_support(pdev, i as u32, actual_surface) }
                .unwrap_or(false);
        if supports && present.is_none() {
            present = Some(i as u32);
        }
    }
    Some(QueueFamilies {
        graphics: graphics?,
        present: present?,
    })
}

pub(super) fn create_logical_device(
    instance: &AshInstance,
    pdev: vk::PhysicalDevice,
    queues: QueueFamilies,
    _surface: &ash::khr::surface::Instance,
    _actual_surface: vk::SurfaceKHR,
) -> Result<(ash::Device, vk::Queue, vk::Queue)> {
    let mut unique = vec![queues.graphics, queues.present];
    unique.sort();
    unique.dedup();
    let priorities = [1.0f32];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo> = unique
        .iter()
        .map(|&q| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(q)
                .queue_priorities(&priorities)
        })
        .collect();

    let extension_names = [vk::KHR_SWAPCHAIN_NAME.as_ptr()];
    // Enable the 1.0 feature bits the GPU-driven indirect path needs:
    //   - multi_draw_indirect:        one vkCmdDrawIndexedIndirect issues N draws.
    //   - draw_indirect_first_instance: allows per-draw firstInstance != 0, which
    //                                  the indirect path uses as a chunk slot
    //                                  index into the bindless origins SSBO.
    let features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(false)
        .multi_draw_indirect(true)
        .draw_indirect_first_instance(true)
        // Validation requires these for the chunks/SSR/wireframe/AA paths:
        //   - fill_mode_non_solid: the wireframe overlay pipeline uses
        //     VK_POLYGON_MODE_LINE (chunk wireframe preview + debug grid).
        //   - sample_rate_shading: post.frag declares a SPIR-V `SampleRateShading`
        //     capability to drive per-sample MSAA shading on the SSR sample ray;
        //     without the feature bit, vkCreateShaderModule fails at startup.
        .fill_mode_non_solid(true)
        .sample_rate_shading(true);
    // Enable Vulkan 1.2 host_query_reset so vkResetQueryPool can be called
    // from host code (e.g., the init-time pool resets and the per-frame
    // timestamp readback reset in renderer/mod.rs). Without this feature
    // the calls return VK_ERROR_FEATURE_NOT_PRESENT and the queries stay
    // in the "uninitialized" state, tripping
    // VUID-vkGetQueryPoolResults-None-09401.
    let mut vulkan12_features =
        vk::PhysicalDeviceVulkan12Features::default().host_query_reset(true);
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extension_names)
        .enabled_features(&features)
        .push_next(&mut vulkan12_features);

    let device = unsafe { instance.create_device(pdev, &create_info, None) }
        .map_err(|e| anyhow!("create_device: {e:?}"))?;
    let graphics_queue = unsafe { device.get_device_queue(queues.graphics, 0) };
    let present_queue = unsafe { device.get_device_queue(queues.present, 0) };
    Ok((device, graphics_queue, present_queue))
}

pub(super) fn find_depth_format(instance: &AshInstance, pdev: vk::PhysicalDevice) -> vk::Format {
    for &f in &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ] {
        let props = unsafe { instance.get_physical_device_format_properties(pdev, f) };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return f;
        }
    }
    vk::Format::D32_SFLOAT
}
