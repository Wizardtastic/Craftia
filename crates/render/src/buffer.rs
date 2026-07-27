//! Buffer and image creation helpers backed by the shared allocator.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::alloc::Alloc;

/// Process-wide cap on the FIRST-N-leaks-warn behaviour in `GpuBuffer` /
/// `GpuImage` `Drop` impls. The first `LEAK_WARN_BUDGET` drops each emit
/// a `WARN` (with a captured `std::backtrace::Backtrace` on the very first
/// occurrence if `RUST_BACKTRACE=1` is set), then a single `WARN`
/// announces that the budget is exhausted. After that, the per-drop
/// `WARN` goes silent and a post-budget leak detector takes over: every
/// drop is checked against a deduplicated set of allocation sites, and
/// the FIRST drop from a previously-unseen allocation site surfaces a
/// `WARN` carrying the source location and the live Vulkan handle, so
/// slow leaks (one new buffer every N frames) stay observable instead of
/// exceeding-the-budget log spam.
const LEAK_WARN_BUDGET: usize = 8;
static GPUBUFFER_LEAK_WARN_COUNT: AtomicUsize = AtomicUsize::new(0);
static GPUIMAGE_LEAK_WARN_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Post-budget canary: deduplicated set of allocation sites that have
/// already produced a leak-detector `WARN`. Indexed purely on
/// `&'static Location<'static>` — keyed that way rather than on
/// `vk::Buffer` because the same source site leaking across frames
/// produces FRESH `vk::Buffer` handles every time, so handle-based
/// deduplication would silently grow unbounded for slow leaks.
///
/// The set is capped at `FINGERPRINT_CAP` entries per kind (buffer and
/// image separately). When exceeded, the set is cleared with a single
/// `INFO` log explaining the rotation, so a long-running session never
/// OOMs the post-budget detector. After rotation the detector
/// restarts fingerprinting from zero, so a leak path that was already
/// logged may re-fire once if it persists past the rotation boundary
/// — that's the trade for keeping the memory footprint bounded. We
/// emit at most one rotation message per kind per session.
static BUFFER_LEAK_LOCATIONS: LazyLock<Mutex<HashSet<&'static std::panic::Location<'static>>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));
static IMAGE_LEAK_LOCATIONS: LazyLock<Mutex<HashSet<&'static std::panic::Location<'static>>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));
/// Per-kind cap on the post-budget fingerprint set. Sized so a long
/// session with thousands of distinct buffer-creation source lines
/// (frame-sized upload buffers, transient per-pass UBOs, ...)
/// doesn't OOM the log directory; the rotation message that fires
/// when this cap is hit tells the user the surface has been exceeded.
/// In tests we use a much smaller cap so the rotation path is
/// exercisable without hand-writing hundreds of drop statements.
///
/// Exactly one of the two `const`s below is compiled per build — the
/// production binary ignores the `cfg(test)` arm and vice versa. If a
/// future contributor adds a third arm under neither/both cfgs they
/// will see a compile error pointing here, which is the intended
/// voice-of-experience to keep the duplication contained.
#[cfg(not(test))]
const FINGERPRINT_CAP: usize = 256;
#[cfg(test)]
const FINGERPRINT_CAP: usize = 4;
static BUFFER_FINGERPRINT_ROTATED: AtomicUsize = AtomicUsize::new(0);
static IMAGE_FINGERPRINT_ROTATED: AtomicUsize = AtomicUsize::new(0);

/// An owned buffer + its memory allocation. Drops free both.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
    /// Source location where this buffer was allocated; used by the
    /// post-budget leak detector to identify which call site is leaking.
    /// Captured by every constructor via `#[track_caller]` +
    /// `Location::caller()` so the value is `&'static` and costs nothing
    /// at runtime.
    pub allocated_at: &'static std::panic::Location<'static>,
}

impl GpuBuffer {
    /// Create a device-local buffer with the given usage flags (no host access).
    #[track_caller]
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
            allocated_at: std::panic::Location::caller(),
        })
    }

    /// Create a host-visible (CPU-mapped) buffer for staging / uniform data.
    #[track_caller]
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
            allocated_at: std::panic::Location::caller(),
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
            let n = GPUBUFFER_LEAK_WARN_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < LEAK_WARN_BUDGET {
                // First-N: capture a real Rust backtrace on the very first
                // leak so the user can find the drop site directly. We
                // deliberately only capture once because Backtrace::capture
                // walks every stack frame, which is non-trivial cost in a
                // tight loop. Additional occurrences stay cheap (no extra
                // capture) but still get the counter so the user can see
                // whether the leak is happening once-per-init (early-return
                // path) or every frame (real bug).
                let trace = if n == 0 && std::env::var_os("RUST_BACKTRACE").is_some() {
                    Some(std::backtrace::Backtrace::capture())
                } else {
                    None
                };
                if let Some(t) = trace {
                    log::warn!(
                        "GpuBuffer dropped without calling destroy_in_place — GPU resource leaked (occurrence 1/{}, real backtrace follows):",
                        LEAK_WARN_BUDGET
                    );
                    for line in t.to_string().lines() {
                        log::warn!("    {}", line);
                    }
                } else {
                    log::warn!(
                        "GpuBuffer dropped without calling destroy_in_place — GPU resource leaked (occurrence {}/{}; set RUST_BACKTRACE=1 to capture a stack trace on the first leak)",
                        n + 1,
                        LEAK_WARN_BUDGET
                    );
                }
            } else if n == LEAK_WARN_BUDGET {
                log::warn!(
                    "further GpuBuffer leak warnings suppressed (saw {}+ occurrences — entering post-budget leak-detector mode, which logs one WARN per newly-leaked allocation site)",
                    LEAK_WARN_BUDGET
                );
            } else {
                // Post-budget canary: deduplicated by allocation site so a
                // long-running slow leak (one new leaked buffer per N frames)
                // surfaces once per *new* offending site, instead of every
                // frame. Site is captured at construction via `#[track_caller]`
                // + `Location::caller()`, so the diagnostic pins the line in
                // user code that created the buffer. Ignored Mutex poison
                // (the GPU-hang path that brought us here might also be
                // unwinding through poisoned-lock territory).
                if let Ok(mut seen) = BUFFER_LEAK_LOCATIONS.lock() {
                    if seen.len() >= FINGERPRINT_CAP {
                        seen.clear();
                        let rotated = BUFFER_FINGERPRINT_ROTATED
                            .fetch_add(1, Ordering::Relaxed);
                        if rotated == 0 {
                            log::warn!(
                                "post-budget GpuBuffer fingerprint store reached {} entries; rotating. \
                                 Long-running sessions with many distinct leaking sites may re-fire \
                                 fingerprints once after each rotation.",
                                FINGERPRINT_CAP,
                            );
                        }
                    }
                    if seen.insert(self.allocated_at) {
                        log::warn!(
                            "GpuBuffer leaked from a previously-unseen allocation site: {} (vk::Buffer handle: {:?}, post-budget occurrence {})",
                            self.allocated_at,
                            self.buffer,
                            n + 1,
                        );
                    }
                }
            }
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
    /// Source location where this image was allocated; captured by every
    /// constructor via `#[track_caller]` + `Location::caller()` and used
    /// by the post-budget leak detector. `pub` for symmetry with
    /// `GpuBuffer::allocated_at` and to make ad-hoc diagnostics easier
    /// from test code; never mutated.
    pub allocated_at: &'static std::panic::Location<'static>,
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
    ///
    /// `#[track_caller]` is on every layer of the forwarding chain
    /// (`depth` → `depth_with_usage`) so `Location::caller()` inside
    /// `depth_with_usage` reflects the actual call site in user code,
    /// not the line in `depth` that forwards the call. Without it the
    /// post-budget leak detector would collapse every `depth()` leak onto
    /// a single fingerprint and lose all diagnostic value.
    #[track_caller]
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
                | vk::ImageUsageFlags::TRANSFER_SRC
                // INPUT_ATTACHMENT is required because the particle subpass
                // (subpass 1 of the main render pass) reads the depth
                // attachment as an input attachment via `subpassLoad`, and the
                // slice-2 transparent render pass also reads it as a sampled
                // image in the water/glass SSR ray-march. Without this flag,
                // `vkCreateFramebuffer` fails VUID-00879 and
                // `vkUpdateDescriptorSets` fails VUID-00338 every frame.
                | vk::ImageUsageFlags::INPUT_ATTACHMENT,
        )
    }

    /// Create a depth image with an explicit sample count (for MSAA).
    #[track_caller]
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
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                // INPUT_ATTACHMENT for the particle subpass's subpassLoad
                // input attachment read on the resolved MSAA depth.
                | vk::ImageUsageFlags::INPUT_ATTACHMENT,
        )
    }

    /// Shared depth-image constructor with caller-chosen usage flags.
    #[track_caller]
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
            allocated_at: std::panic::Location::caller(),
        })
    }

    /// `#[track_caller]` propagates the caller location through
    /// `color_attachment_msaa` so the post-budget leak detector sees the
    /// real user-side call site.
    #[track_caller]
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
    #[track_caller]
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
            allocated_at: std::panic::Location::caller(),
        })
    }    #[track_caller]
    pub fn depth_array(
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
            allocated_at: std::panic::Location::caller(),
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
    #[track_caller]
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
            allocated_at: std::panic::Location::caller(),
        })
    }

    /// Create a `scene_opaque_depth` image — the single-sample depth companion
    /// to `scene_opaque_color`. Once per frame (right after the main render
    /// pass, alongside the color copy) the resolved single-sample scene depth
    /// is `vkCmdCopyImage`'d into this image, which the chunk fragment shader
    /// then samples (binding 7) for the water/glass SSR ray-march and the
    /// water-column Beer absorption depth. Same extent and depth format as the
    /// main pass depth attachment so the copy is a plain texel-for-texel blit.
    #[track_caller]
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
            allocated_at: std::panic::Location::caller(),
        })
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        if self.image != vk::Image::null() {
            let n = GPUIMAGE_LEAK_WARN_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < LEAK_WARN_BUDGET {
                // Same first-N policy as GpuBuffer: capture a real backtrace
                // on the very first leak (only if RUST_BACKTRACE=1), warn
                // cheap-and-contextual for occurrences 2..=BUDGET.
                let trace = if n == 0 && std::env::var_os("RUST_BACKTRACE").is_some() {
                    Some(std::backtrace::Backtrace::capture())
                } else {
                    None
                };
                if let Some(t) = trace {
                    log::warn!(
                        "GpuImage dropped without calling destroy — GPU resource leaked (occurrence 1/{}, real backtrace follows):",
                        LEAK_WARN_BUDGET
                    );
                    for line in t.to_string().lines() {
                        log::warn!("    {}", line);
                    }
                } else {
                    log::warn!(
                        "GpuImage dropped without calling destroy — GPU resource leaked (occurrence {}/{}; set RUST_BACKTRACE=1 to capture a stack trace on the first leak)",
                        n + 1,
                        LEAK_WARN_BUDGET
                    );
                }
            } else if n == LEAK_WARN_BUDGET {
                log::warn!(
                    "further GpuImage leak warnings suppressed (saw {}+ occurrences — entering post-budget leak-detector mode, which logs one WARN per newly-leaked allocation site)",
                    LEAK_WARN_BUDGET
                );
            } else {
                // Post-budget canary, mirroring GpuBuffer. Site-keyed
                // fingerprint means a long-running slow image leak (one
                // new image per several frames) is observable instead of
                // blanketing the log.
                if let Ok(mut seen) = IMAGE_LEAK_LOCATIONS.lock() {
                    if seen.len() >= FINGERPRINT_CAP {
                        seen.clear();
                        let rotated = IMAGE_FINGERPRINT_ROTATED
                            .fetch_add(1, Ordering::Relaxed);
                        if rotated == 0 {
                            log::warn!(
                                "post-budget GpuImage fingerprint store reached {} entries; rotating. \
                                 Long-running sessions with many distinct leaking sites may re-fire \
                                 fingerprints once after each rotation.",
                                FINGERPRINT_CAP,
                            );
                        }
                    }
                    if seen.insert(self.allocated_at) {
                        log::warn!(
                            "GpuImage leaked from a previously-unseen allocation site: {} (vk::Image handle: {:?}, vk::ImageView handle: {:?}, post-budget occurrence {})",
                            self.allocated_at,
                            self.image,
                            self.view,
                            n + 1,
                        );
                    }
                }
            }
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

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) fn reset_leak_detectors_for_test() {
    GPUBUFFER_LEAK_WARN_COUNT.store(0, Ordering::Relaxed);
    GPUIMAGE_LEAK_WARN_COUNT.store(0, Ordering::Relaxed);
    if let Ok(mut s) = BUFFER_LEAK_LOCATIONS.lock() {
        s.clear();
    }
    if let Ok(mut s) = IMAGE_LEAK_LOCATIONS.lock() {
        s.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    /// Build a `GpuBuffer` directly without going through the Vulkan
    /// device — we just want a drop that has a non-null `vk::Buffer`
    /// (which is what triggers the leak path in `Drop`) and a stable
    /// `allocated_at`. `vk::Buffer::from_raw` is total in modern ash,
    /// so no `unsafe` block here.
    fn mock_buffer_with_allocated_at(
        raw: u64,
        allocated_at: &'static std::panic::Location<'static>,
    ) -> GpuBuffer {
        GpuBuffer {
            buffer: vk::Buffer::from_raw(raw),
            allocation: None,
            size: 1024,
            allocated_at,
        }
    }

    /// Helper that drops a fresh `GpuBuffer` whose `allocated_at`
    /// reflects the caller's source line. Without `#[track_caller]`,
    /// `Location::caller()` inside would point at this helper's own
    /// body and N call sites would all collapse to the same Location;
    /// with `#[track_caller]`, each call site produces a distinct
    /// Location so the post-budget detector records each as a unique
    /// fingerprint. Used to drive FINGERPRINT_CAP+1 distinct drops in
    /// the rotation test.
    ///
    /// ⚠️ Do not remove the `#[track_caller]` annotation — without it
    /// the rotation test will silently pass with broken invariants (all
    /// 5 helper calls collapse to one Location so the rotation never
    /// fires). The annotation is the load-bearing piece of the test.
    #[track_caller]
    fn drop_at_caller_line(raw: u64) {
        let loc = std::panic::Location::caller();
        let buf = mock_buffer_with_allocated_at(raw, loc);
        drop(buf);
    }

    /// Verify the post-budget detector:
    ///   (1) the warning counter reaches the post-budget branch
    ///       (`counter > LEAK_WARN_BUDGET`),
    ///   (2) drops at the same allocation site dedup to a single
    ///       seen-set entry,
    ///   (3) drops at different sites are recorded as distinct entries.
    ///
    /// Step 1 uses a tight loop intentionally. Each loop iteration's
    /// `Location::caller()` returns the *same* source location (same
    /// line, same column), so those drops share an allocation site —
    /// but they all run through the FIRST-N branch (`n < BUDGET`) and
    /// the suppression line (`n == BUDGET`), neither of which touches
    /// `BUFFER_LEAK_LOCATIONS`. The result is: counter pushes to
    /// `LEAK_WARN_BUDGET + 1` while seen-set stays empty. Step 2+
    /// then drives the dedup-vs-unique distinction explicitly, with
    /// every drop going through the post-budget branch.
    #[test]
    fn post_budget_detector_dedup_by_location() {
        use std::sync::atomic::Ordering;

        reset_leak_detectors_for_test();

        // Step 1 — drive counter past the budget via a tight loop.
        // All 9 drops run through first-N + suppression branches only;
        // `seen-set` is not modified here. Counter ends at 9.
        for _ in 0..(LEAK_WARN_BUDGET + 1) {
            drop(mock_buffer_with_allocated_at(
                1,
                std::panic::Location::caller(),
            ));
        }

        let counter_after_step1 =
            GPUBUFFER_LEAK_WARN_COUNT.load(Ordering::Relaxed);
        assert!(
            counter_after_step1 > LEAK_WARN_BUDGET,
            "post-budget branch not exercised: counter={counter_after_step1}",
        );
        assert_eq!(
            counter_after_step1,
            LEAK_WARN_BUDGET + 1,
            "step-1 counter should reflect every tight-loop drop",
        );

        // Verify pre-condition: step 1 must NOT have polluted the
        // seen-set (it only ran first-N + suppression branches, both
        // of which bypass the post-budget fingerprint store). This
        // sanity-check catches a regression where someone accidentally
        // adds the post-budget branch earlier in the Drop chain.
        assert!(
            BUFFER_LEAK_LOCATIONS
                .lock()
                .expect("BUFFER_LEAK_LOCATIONS poisoned in test")
                .is_empty(),
            "step 1 (tight loop) must not touch BUFFER_LEAK_LOCATIONS",
        );

        // Step 2 — 2 drops at the SAME captured site. The captured
        // `stable_loc` is bound once and reused, so both buffers carry
        // the same `allocated_at` — post-budget detector must dedup to
        // exactly 1 entry.
        let stable_loc = std::panic::Location::caller();
        drop(mock_buffer_with_allocated_at(2, stable_loc));
        drop(mock_buffer_with_allocated_at(3, stable_loc));

        // Step 3 — 1 drop at a DIFFERENT site (different source line in
        // this test, matches `stable_loc != last_loc`).
        let last_loc = std::panic::Location::caller();
        drop(mock_buffer_with_allocated_at(4, last_loc));

        // Total drops = 9 + 2 + 1 = 12. Counter should match.
        let total_drops = (LEAK_WARN_BUDGET + 1) + 2 + 1;
        let counter_final =
            GPUBUFFER_LEAK_WARN_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            counter_final, total_drops,
            "warning counter should equal the total number of leaked buffers seen",
        );

        let seen = BUFFER_LEAK_LOCATIONS
            .lock()
            .expect("BUFFER_LEAK_LOCATIONS poisoned in test");

        // Load-bearing assertion: DEDUP INVARIANT. 2 drops at
        // stable_loc produce exactly 1 entry.
        let stable_count = seen.iter().filter(|l| *l == &stable_loc).count();
        assert_eq!(
            stable_count, 1,
            "stable_loc must appear exactly once in the seen-set across \
             2 drops that share the allocation site (post-budget detector dedups)",
        );

        // DISTINCTNESS INVARIANT: last_loc must be present and distinct.
        assert!(
            seen.contains(&last_loc),
            "last_loc must be present in the seen-set",
        );
        assert_ne!(
            stable_loc, last_loc,
            "sanity: stable_loc and last_loc call sites must differ",
        );
    }

    /// When the post-budget fingerprint set fills past `FINGERPRINT_CAP`
    /// entries, the detector rotates (clears the seen-set, bumps the
    /// per-kind rotation marker). In tests `FINGERPRINT_CAP = 4`, so
    /// the boundary is crossed after 4 distinct-site inserts and the
    /// 5th drop triggers the clear.
    #[test]
    fn post_budget_detector_rotates_when_full() {
        use std::sync::atomic::Ordering;

        reset_leak_detectors_for_test();

        // Step 1 — drive counter past the budget via a tight loop.
        // Counter goes to 9; seen-set stays empty (first-N + suppression
        // branches only; see the dedup test's comment for the same
        // reasoning).
        for _ in 0..(LEAK_WARN_BUDGET + 1) {
            drop(mock_buffer_with_allocated_at(
                1,
                std::panic::Location::caller(),
            ));
        }

        // Regression guard mirrored from the dedup test: step 1 must
        // not have polluted BUFFER_LEAK_LOCATIONS (only first-N +
        // suppression branches run during the tight loop). If a future
        // refactor accidentally moves post-budget detection earlier in
        // the Drop chain, this assertion fails with a clear message.
        assert!(
            BUFFER_LEAK_LOCATIONS
                .lock()
                .expect("BUFFER_LEAK_LOCATIONS poisoned in test")
                .is_empty(),
            "step 1 (tight loop) must not touch BUFFER_LEAK_LOCATIONS",
        );

        // Step 2 — 5 distinct-source-line drops. Each call to
        // `drop_at_caller_line` lives on its own source line and the
        // helper is `#[track_caller]`-marked, so each invocation
        // produces a distinct Location fed into `BUFFER_LEAK_LOCATIONS`.
        // Trace (FINGERPRINT_CAP = 4 in tests):
        //   drop 1: seen.len()=0 < 4, insert loc_A           -> len=1
        //   drop 2: seen.len()=1 < 4, insert loc_B           -> len=2
        //   drop 3: seen.len()=2 < 4, insert loc_C           -> len=3
        //   drop 4: seen.len()=3 < 4, insert loc_D           -> len=4
        //   drop 5: seen.len()=4 >= 4, ROTATE clear+log+counter
        //            seen={}, then insert loc_E              -> len=1
        drop_at_caller_line(10);
        drop_at_caller_line(11);
        drop_at_caller_line(12);
        drop_at_caller_line(13);
        drop_at_caller_line(14);

        // Rotation marker must have fired exactly once: drop 5
        // triggered the clear (its `seen.len() >= CAP` branch).
        assert_eq!(
            BUFFER_FINGERPRINT_ROTATED.load(Ordering::Relaxed),
            1,
            "rotation should have fired exactly once when seen-set reached \
             {} entries (drop 5 is the boundary)",
            FINGERPRINT_CAP,
        );

        // After rotation, only the post-rotation insert (loc_E from
        // drop 5) survived — seen.len() should be exactly 1.
        let seen = BUFFER_LEAK_LOCATIONS
            .lock()
            .expect("BUFFER_LEAK_LOCATIONS poisoned in test");
        assert_eq!(
            seen.len(),
            1,
            "post-rotation seen-set should hold exactly the post-rotation \
             insert (loc_E from drop 5); rotation cleared everything else",
        );
    }
}
