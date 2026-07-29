//! Phase 1 — GPU-driven rendering pipeline.
//!
//! Replaces the per-chunk `vkCmdDrawIndexed` loop with GPU-driven primitives:
//! mega geometry buffers, a compute-shader frustum cull writing into an indirect
//! command buffer, and a bindless origins SSBO. The legacy per-chunk path
//! remains the default; this subsystem is enabled via
//! [`crate::renderer::RendererConfig::gpu_driven`].

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use voxel_core::{math::chunk_origin, ChunkPos, CHUNK_SIZE};

use crate::alloc::Alloc;
use crate::buffer::GpuBuffer;
use crate::texture::{begin_one_time, end_and_submit};
use crate::MeshPass;

use super::pipeline::{create_graphics_pipeline, spirv_to_u32};

const VERTEX_STRIDE: vk::DeviceSize = std::mem::size_of::<crate::Vertex>() as vk::DeviceSize;
const INDEX_STRIDE: vk::DeviceSize = std::mem::size_of::<u32>() as vk::DeviceSize;
const MEGA_VBO_INITIAL: vk::DeviceSize = 4 * 1024 * 1024;
const MEGA_IBO_INITIAL: vk::DeviceSize = 1 * 1024 * 1024;
const MAX_DRAW_SLOTS: usize = 8192;
const TILE_REMAP_UBO_SIZE: vk::DeviceSize = 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ChunkAabb {
    pub min: [f32; 4],
    pub max: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CullUbo {
    pub view_proj: [f32; 16],
    pub cam_pos: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ChunkOrigin {
    pub origin: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct MegaSlot {
    vbo_offset: vk::DeviceSize,
    ibo_offset: vk::DeviceSize,
    vertex_count: u32,
    index_count: u32,
}

#[derive(Default)]
struct GpuChunk {
    opaque: Option<MegaSlot>,
    transparent: Option<MegaSlot>,
}

pub struct GpuDriven {
    mega_vbo: GpuBuffer,
    mega_ibo: GpuBuffer,
    mega_vbo_capacity: vk::DeviceSize,
    mega_ibo_capacity: vk::DeviceSize,
    mega_vbo_used: vk::DeviceSize,
    mega_ibo_used: vk::DeviceSize,
    chunks: HashMap<ChunkPos, GpuChunk>,
    order: Vec<ChunkPos>,
    dirty: bool,
    opaque_slots: usize,
    transparent_slots: usize,
    indirect_cmd_buf: GpuBuffer,
    aabb_buf: GpuBuffer,
    origins_buf: GpuBuffer,
    cull_ubo: GpuBuffer,
    dummy_tile_remap_ubo: GpuBuffer,
    cull_pipeline: vk::Pipeline,
    cull_pipeline_layout: vk::PipelineLayout,
    cull_set_layout: vk::DescriptorSetLayout,
    cull_descriptor_pool: vk::DescriptorPool,
    cull_descriptor_set: vk::DescriptorSet,
    opaque_pipeline: vk::Pipeline,
    transparent_pipeline: vk::Pipeline,
    indirect_pipeline_layout: vk::PipelineLayout,
    indirect_set_layout: vk::DescriptorSetLayout,
    indirect_descriptor_pool: vk::DescriptorPool,
    indirect_descriptor_set: vk::DescriptorSet,
    /// Phase-2 GPU compute mesher. `Some` when `config.gpu_meshing` is enabled.
    gpu_mesher: Option<GpuMesher>,
}

impl GpuDriven {
    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: &ash::Device,
        alloc: &Alloc,
        render_pass: vk::RenderPass,
        chunk_set_layout: vk::DescriptorSetLayout,
        msaa_samples: vk::SampleCountFlags,
        gpu_meshing: bool,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Result<Self> {
        let mega_vbo = GpuBuffer::device_local(device, alloc, MEGA_VBO_INITIAL,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, "mega_vbo")?;
        let mega_ibo = GpuBuffer::device_local(device, alloc, MEGA_IBO_INITIAL,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, "mega_ibo")?;
        let cmd_sz = (MAX_DRAW_SLOTS * std::mem::size_of::<DrawIndexedIndirectCommand>()) as vk::DeviceSize;
        let aabb_sz = (MAX_DRAW_SLOTS * std::mem::size_of::<ChunkAabb>()) as vk::DeviceSize;
        let orig_sz = (MAX_DRAW_SLOTS * std::mem::size_of::<ChunkOrigin>()) as vk::DeviceSize;
        let indirect_cmd_buf = GpuBuffer::host_visible(device, alloc, cmd_sz,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER, "indirect_cmd_buf")?;
        let aabb_buf = GpuBuffer::host_visible(device, alloc, aabb_sz,
            vk::BufferUsageFlags::STORAGE_BUFFER, "aabb_buf")?;
        let origins_buf = GpuBuffer::host_visible(device, alloc, orig_sz,
            vk::BufferUsageFlags::STORAGE_BUFFER, "origins_buf")?;
        let cull_ubo = GpuBuffer::host_visible(device, alloc,
            std::mem::size_of::<CullUbo>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "cull_ubo")?;
        let mut dummy = GpuBuffer::host_visible(device, alloc, TILE_REMAP_UBO_SIZE,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "dummy_tile_remap")?;
        dummy.mapped_slice_mut()?.fill(0);
        // Flush zero-init so the device sees a deterministic UBO rather
        // than garbage if the GPU ever reads it before another write.
        // HOST_COHERENT -> no-op; non-coherent -> required.
        if let Err(e) = dummy.flush_whole(device) {
            log::warn!("dummy_tile_remap flush failed: {e}");
        }
        let cull_spv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_cull.comp.spv")).to_vec();
        let vert_spv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_indirect.vert.spv")).to_vec();
        let frag_spv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.frag.spv")).to_vec();
        let cull_set_layout = create_cull_set_layout(device)?;
        let cull_pl = create_cull_pipeline_layout(device, cull_set_layout)?;
        let cull_pipeline = create_compute_pipeline(device, cull_pl, &cull_spv)?;
        let cull_pool = create_cull_descriptor_pool(device)?;
        let cull_set = allocate_set(device, cull_pool, cull_set_layout)?;
        update_cull_set(device, cull_set, cull_ubo.buffer, indirect_cmd_buf.buffer, aabb_buf.buffer);
        let ind_set_layout = create_indirect_set_layout(device)?;
        let ind_pl = create_indirect_pipeline_layout(device, chunk_set_layout, ind_set_layout)?;
        let opaque_pipeline = create_graphics_pipeline(device, render_pass, ind_pl,
            vk::PolygonMode::FILL, vk::CullModeFlags::BACK, &vert_spv, &frag_spv, msaa_samples, true)?;
        let transparent_pipeline = create_graphics_pipeline(device, render_pass, ind_pl,
            vk::PolygonMode::FILL, vk::CullModeFlags::NONE, &vert_spv, &frag_spv, msaa_samples, false)?;
        let ind_pool = create_indirect_descriptor_pool(device)?;
        let ind_set = allocate_set(device, ind_pool, ind_set_layout)?;
        update_indirect_set(device, ind_set, dummy.buffer, origins_buf.buffer);
        let gpu_mesher = if gpu_meshing {
            match GpuMesher::new(device, alloc, command_pool, graphics_queue) {
                Ok(m) => Some(m),
                Err(e) => { log::error!("GPU mesher init failed: {e}"); None }
            }
        } else { None };
        log::info!("GPU-driven chunk pipeline ready (VBO {} KiB, IBO {} KiB, max {} slots)",
            MEGA_VBO_INITIAL / 1024, MEGA_IBO_INITIAL / 1024, MAX_DRAW_SLOTS);
        Ok(Self {
            mega_vbo, mega_ibo, mega_vbo_capacity: MEGA_VBO_INITIAL, mega_ibo_capacity: MEGA_IBO_INITIAL,
            mega_vbo_used: 0, mega_ibo_used: 0, chunks: HashMap::new(), order: Vec::new(),
            dirty: true, opaque_slots: 0, transparent_slots: 0,
            indirect_cmd_buf, aabb_buf, origins_buf, cull_ubo, dummy_tile_remap_ubo: dummy,
            cull_pipeline, cull_pipeline_layout: cull_pl, cull_set_layout,
            cull_descriptor_pool: cull_pool, cull_descriptor_set: cull_set,
            opaque_pipeline, transparent_pipeline, indirect_pipeline_layout: ind_pl,
            indirect_set_layout: ind_set_layout, indirect_descriptor_pool: ind_pool, indirect_descriptor_set: ind_set,
            gpu_mesher,
        })
    }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }

    /// Update the block properties table used by the GPU compute mesher.
    pub fn set_block_properties(&mut self, device: &ash::Device, alloc: &Alloc, props: &[crate::BlockPropertiesGpu]) {
        if let Some(m) = self.gpu_mesher.as_mut() {
            m.set_block_properties(device, alloc, props);
        }
    }

    /// GPU-mesh a chunk: dispatch the compute mesher and insert the result into
    /// the mega VBO/IBO. Returns true on success.
    pub fn upload_chunk_gpu_mesh(
        &mut self, device: &ash::Device, alloc: &Alloc,
        command_pool: vk::CommandPool, graphics_queue: vk::Queue,
        pos: ChunkPos, pass: MeshPass, voxels: &[u16],
    ) -> bool {
        if self.gpu_mesher.is_none() { return false; }
        let pass_mode = match pass { MeshPass::Opaque => 0u32, MeshPass::Transparent => 1u32 };
        let est_vbo = (16 * 16 * 16) as vk::DeviceSize * 6 * 4 * 32;
        let est_ibo = (16 * 16 * 16) as vk::DeviceSize * 6 * 6 * 4;
        if self.mega_vbo_used + est_vbo > self.mega_vbo_capacity
            || self.mega_ibo_used + est_ibo > self.mega_ibo_capacity
        {
            if let Err(e) = self.grow_and_compact(device, alloc, command_pool, graphics_queue, est_vbo, est_ibo) {
                log::error!("gpu-mesh grow_and_compact: {e}"); return false;
            }
        }
        let vbo_off = self.mega_vbo_used;
        let ibo_off = self.mega_ibo_used;
        let mega_vbo = self.mega_vbo.buffer;
        let mega_ibo = self.mega_ibo.buffer;
        let result = self.gpu_mesher.as_mut().unwrap().mesh_chunk(
            device, alloc, command_pool, graphics_queue, voxels, pass_mode,
            mega_vbo, mega_ibo, vbo_off, ibo_off,
        );
        let Some((vert_count, idx_count)) = result else { return false; };
        if vert_count == 0 || idx_count == 0 {
            let entry = self.chunks.entry(pos).or_default();
            let slot = match pass {
                MeshPass::Opaque => &mut entry.opaque,
                MeshPass::Transparent => &mut entry.transparent,
            };
            *slot = None;
            return true;
        }
        let v_bytes = vert_count as vk::DeviceSize * 32;
        let i_bytes = idx_count as vk::DeviceSize * 4;
        let slot = MegaSlot { vbo_offset: vbo_off, ibo_offset: ibo_off, vertex_count: vert_count, index_count: idx_count };
        let entry = self.chunks.entry(pos).or_default();
        *match pass { MeshPass::Opaque => &mut entry.opaque, MeshPass::Transparent => &mut entry.transparent } = Some(slot);
        self.mega_vbo_used += v_bytes;
        self.mega_ibo_used += i_bytes;
        self.dirty = true;
        true
    }
}

fn create_cull_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default().binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default().binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default().binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }.map_err(|e| anyhow!("cull_set_layout: {e:?}"))
}
fn create_cull_pipeline_layout(device: &ash::Device, set_layout: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout> {
    let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0).size(std::mem::size_of::<u32>() as u32);
    let set_layouts = [set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push));
    unsafe { device.create_pipeline_layout(&info, None) }.map_err(|e| anyhow!("cull_pl: {e:?}"))
}
fn create_compute_pipeline(device: &ash::Device, layout: vk::PipelineLayout, spirv: &[u8]) -> Result<vk::Pipeline> {
    let code = spirv_to_u32(spirv);
    let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }
        .map_err(|e| anyhow!("cull module: {e:?}"))?;
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
    let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
    let result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe { device.destroy_shader_module(module, None) };
    let pipelines = result.map_err(|(_p, e)| anyhow!("compute_pipelines: {e:?}"))?;
    Ok(pipelines.into_iter().next().expect("compute pipeline"))
}
fn create_cull_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 },
    ];
    let info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe { device.create_descriptor_pool(&info, None) }.map_err(|e| anyhow!("cull_pool: {e:?}"))
}
fn update_cull_set(device: &ash::Device, set: vk::DescriptorSet, ubo: vk::Buffer, cmd_buf: vk::Buffer, aabb_buf: vk::Buffer) {
    let ubo_info = [vk::DescriptorBufferInfo::default().buffer(ubo).offset(0).range(std::mem::size_of::<CullUbo>() as u64)];
    let cmd_info = [vk::DescriptorBufferInfo::default().buffer(cmd_buf).offset(0).range(vk::WHOLE_SIZE)];
    let aabb_info = [vk::DescriptorBufferInfo::default().buffer(aabb_buf).offset(0).range(vk::WHOLE_SIZE)];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&cmd_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&aabb_info),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn create_indirect_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::VERTEX),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }.map_err(|e| anyhow!("ind_set_layout: {e:?}"))
}
fn create_indirect_pipeline_layout(device: &ash::Device, chunk: vk::DescriptorSetLayout, ind: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout> {
    let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::VERTEX).offset(0).size(80);
    let set_layouts = [chunk, ind];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(std::slice::from_ref(&push));
    unsafe { device.create_pipeline_layout(&info, None) }.map_err(|e| anyhow!("ind_pl: {e:?}"))
}
fn create_indirect_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
    ];
    let info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe { device.create_descriptor_pool(&info, None) }.map_err(|e| anyhow!("ind_pool: {e:?}"))
}
fn update_indirect_set(device: &ash::Device, set: vk::DescriptorSet, ubo: vk::Buffer, origins: vk::Buffer) {
    let ubo_info = [vk::DescriptorBufferInfo::default().buffer(ubo).offset(0).range(TILE_REMAP_UBO_SIZE)];
    let orig_info = [vk::DescriptorBufferInfo::default().buffer(origins).offset(0).range(vk::WHOLE_SIZE)];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&orig_info),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}
fn allocate_set(device: &ash::Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
    let layouts = [layout];
    let info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&info) }.map_err(|e| anyhow!("alloc_set: {e:?}"))?;
    Ok(sets[0])
}

impl GpuDriven {
    pub fn upload(&mut self, device: &ash::Device, alloc: &Alloc,
        command_pool: vk::CommandPool, graphics_queue: vk::Queue, uploads: Vec<crate::ChunkUpload>) {
        let uploads: Vec<_> = uploads.into_iter()
            .filter(|u| !u.vertices.is_empty() && !u.indices.is_empty()).collect();
        if uploads.is_empty() { return; }
        let total_vbo: vk::DeviceSize = uploads.iter().map(|u| u.vertices.len() as vk::DeviceSize).sum();
        let total_ibo: vk::DeviceSize = uploads.iter().map(|u| u.indices.len() as vk::DeviceSize).sum();
        if self.mega_vbo_used + total_vbo > self.mega_vbo_capacity
            || self.mega_ibo_used + total_ibo > self.mega_ibo_capacity
        {
            if let Err(e) = self.grow_and_compact(device, alloc, command_pool, graphics_queue, total_vbo, total_ibo) {
                log::error!("gpu-driven grow_and_compact failed: {e}"); return;
            }
        }
        let staging_size = total_vbo + total_ibo;
        let mut staging = match GpuBuffer::host_visible(device, alloc, staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC, "gpu_staging") {
            Ok(b) => b, Err(e) => { log::error!("gpu staging alloc: {e}"); return; }
        };
        let staging_buf = staging.buffer;
        {
            let slice = match staging.mapped_slice_mut() {
                Ok(s) => s, Err(e) => { log::error!("gpu staging map: {e}"); staging.destroy(device, alloc); return; }
            };
            let mut off = 0usize;
            for u in &uploads {
                let v = u.vertices.len(); let i = u.indices.len();
                slice[off..off + v].copy_from_slice(&u.vertices);
                slice[off + v..off + v + i].copy_from_slice(&u.indices);
                off += v + i;
            }
        }
        // Flush the staging buffer so the GPU's `cmd_copy_buffer` reads
        // the latest bytes, not stale ones. Required on non-coherent
        // memory (which is what produced the "flashing white chunks"
        // symptom); no-op on HOST_COHERENT.
        if let Err(e) = staging.flush_whole(device) {
            log::error!("gpu staging flush failed: {e}");
        }
        let cmd = match begin_one_time(device, command_pool) {
            Ok(c) => c, Err(e) => { log::error!("gpu begin_one_time: {e}"); staging.destroy(device, alloc); return; }
        };
        let mut vc = self.mega_vbo_used; let mut ic = self.mega_ibo_used;
        let mut soff: vk::DeviceSize = 0;
        for u in &uploads {
            let vs = u.vertices.len() as vk::DeviceSize;
            let is = u.indices.len() as vk::DeviceSize;
            unsafe {
                device.cmd_copy_buffer(cmd, staging_buf, self.mega_vbo.buffer,
                    &[vk::BufferCopy::default().src_offset(soff).dst_offset(vc).size(vs)]);
                device.cmd_copy_buffer(cmd, staging_buf, self.mega_ibo.buffer,
                    &[vk::BufferCopy::default().src_offset(soff + vs).dst_offset(ic).size(is)]);
            }
            soff += vs + is; vc += vs; ic += is;
        }
        if let Err(e) = end_and_submit(device, command_pool, graphics_queue, cmd) {
            log::error!("gpu upload submit: {e}"); staging.destroy(device, alloc); return;
        }
        let mut vc = self.mega_vbo_used; let mut ic = self.mega_ibo_used;
        for u in uploads {
            let vs = u.vertices.len() as vk::DeviceSize;
            let is = u.indices.len() as vk::DeviceSize;
            let slot = MegaSlot { vbo_offset: vc, ibo_offset: ic,
                vertex_count: (vs / VERTEX_STRIDE) as u32, index_count: u.index_count };
            let entry = self.chunks.entry(u.pos).or_default();
            *match u.pass { MeshPass::Opaque => &mut entry.opaque, MeshPass::Transparent => &mut entry.transparent } = Some(slot);
            vc += vs; ic += is;
        }
        self.mega_vbo_used = vc; self.mega_ibo_used = ic; self.dirty = true;
        staging.destroy(device, alloc);
    }

    fn grow_and_compact(&mut self, device: &ash::Device, alloc: &Alloc,
        command_pool: vk::CommandPool, graphics_queue: vk::Queue,
        needed_vbo: vk::DeviceSize, needed_ibo: vk::DeviceSize) -> Result<()> {
        let new_vc = std::cmp::max(self.mega_vbo_capacity * 2, self.mega_vbo_used + needed_vbo);
        let new_ic = std::cmp::max(self.mega_ibo_capacity * 2, self.mega_ibo_used + needed_ibo);
        let new_vbo = GpuBuffer::device_local(device, alloc, new_vc,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC, "mega_vbo_new")?;
        let new_ibo = GpuBuffer::device_local(device, alloc, new_ic,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC, "mega_ibo_new")?;
        let mut live: Vec<(ChunkPos, MeshPass, MegaSlot)> = Vec::new();
        for (&pos, chunk) in &self.chunks {
            if let Some(s) = chunk.opaque { live.push((pos, MeshPass::Opaque, s)); }
            if let Some(s) = chunk.transparent { live.push((pos, MeshPass::Transparent, s)); }
        }
        live.sort_by_key(|(_, _, s)| s.vbo_offset);
        let cmd = begin_one_time(device, command_pool)?;
        let mut nvc: vk::DeviceSize = 0; let mut nic: vk::DeviceSize = 0;
        let mut new_slots: Vec<(ChunkPos, MeshPass, MegaSlot)> = Vec::with_capacity(live.len());
        for (pos, pass, old) in &live {
            let vb = (old.vertex_count as vk::DeviceSize) * VERTEX_STRIDE;
            let ib = (old.index_count as vk::DeviceSize) * INDEX_STRIDE;
            unsafe {
                if old.vertex_count > 0 {
                    device.cmd_copy_buffer(cmd, self.mega_vbo.buffer, new_vbo.buffer,
                        &[vk::BufferCopy::default().src_offset(old.vbo_offset).dst_offset(nvc).size(vb)]);
                }
                device.cmd_copy_buffer(cmd, self.mega_ibo.buffer, new_ibo.buffer,
                    &[vk::BufferCopy::default().src_offset(old.ibo_offset).dst_offset(nic).size(ib)]);
            }
            new_slots.push((*pos, *pass, MegaSlot { vbo_offset: nvc, ibo_offset: nic,
                vertex_count: old.vertex_count, index_count: old.index_count }));
            nvc += vb; nic += ib;
        }
        end_and_submit(device, command_pool, graphics_queue, cmd)?;
        self.mega_vbo.destroy_in_place(device, alloc); self.mega_ibo.destroy_in_place(device, alloc);
        self.mega_vbo = new_vbo; self.mega_ibo = new_ibo;
        self.mega_vbo_capacity = new_vc; self.mega_ibo_capacity = new_ic;
        self.mega_vbo_used = nvc; self.mega_ibo_used = nic;
        for (pos, pass, slot) in new_slots {
            let ch = self.chunks.get_mut(&pos).expect("live chunk must exist");
            match pass { MeshPass::Opaque => ch.opaque = Some(slot), MeshPass::Transparent => ch.transparent = Some(slot) }
        }
        self.dirty = true; Ok(())
    }

    pub fn remove(&mut self, pos: ChunkPos) {
        if self.chunks.remove(&pos).is_some() { self.dirty = true; }
    }

    fn rebuild(&mut self, device: &ash::Device) {
        let mut positions: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        positions.sort_by_key(|p| (p.0.x, p.0.y, p.0.z));
        self.order = positions.clone();
        let (mut oc, mut tc) = (0usize, 0usize);
        for pos in &self.order {
            let ch = &self.chunks[pos];
            if ch.opaque.is_some() { oc += 1; }
            if ch.transparent.is_some() { tc += 1; }
        }
        self.opaque_slots = oc;
        self.transparent_slots = tc;
        let total = oc + tc;
        let mut cmds = vec![DrawIndexedIndirectCommand::default(); total];
        let mut aabbs = vec![ChunkAabb::default(); total];
        let mut origins = vec![ChunkOrigin::default(); total];
        let (mut oi, mut ti) = (0usize, oc);
        for pos in &self.order {
            let ch = &self.chunks[pos];
            let org = chunk_origin(*pos);
            let mn = Vec3::new(org.x as f32, org.y as f32, org.z as f32);
            let mx = mn + Vec3::splat(CHUNK_SIZE as f32);
            let o4 = [org.x as f32, org.y as f32, org.z as f32, 0.0];
            let ab = ChunkAabb { min: [mn.x, mn.y, mn.z, 0.0], max: [mx.x, mx.y, mx.z, 0.0] };
            if let Some(s) = ch.opaque {
                cmds[oi] = DrawIndexedIndirectCommand { index_count: s.index_count, instance_count: 0,
                    first_index: (s.ibo_offset / INDEX_STRIDE) as u32,
                    vertex_offset: (s.vbo_offset / VERTEX_STRIDE) as i32, first_instance: oi as u32 };
                aabbs[oi] = ab; origins[oi] = ChunkOrigin { origin: o4 }; oi += 1;
            }
            if let Some(s) = ch.transparent {
                cmds[ti] = DrawIndexedIndirectCommand { index_count: s.index_count, instance_count: 0,
                    first_index: (s.ibo_offset / INDEX_STRIDE) as u32,
                    vertex_offset: (s.vbo_offset / VERTEX_STRIDE) as i32, first_instance: ti as u32 };
                aabbs[ti] = ab; origins[ti] = ChunkOrigin { origin: o4 }; ti += 1;
            }
        }
        if let Ok(slice) = self.indirect_cmd_buf.mapped_slice_mut() {
            let n = std::mem::size_of::<DrawIndexedIndirectCommand>() * total;
            slice[..n].copy_from_slice(bytemuck::cast_slice(&cmds));
            if let Err(e) = self.indirect_cmd_buf.flush_whole(device) {
                log::warn!("indirect_cmd_buf flush failed: {e}");
            }
        }
        if let Ok(slice) = self.aabb_buf.mapped_slice_mut() {
            let n = std::mem::size_of::<ChunkAabb>() * total;
            slice[..n].copy_from_slice(bytemuck::cast_slice(&aabbs));
            if let Err(e) = self.aabb_buf.flush_whole(device) {
                log::warn!("aabb_buf flush failed: {e}");
            }
        }
        if let Ok(slice) = self.origins_buf.mapped_slice_mut() {
            let n = std::mem::size_of::<ChunkOrigin>() * total;
            slice[..n].copy_from_slice(bytemuck::cast_slice(&origins));
            if let Err(e) = self.origins_buf.flush_whole(device) {
                log::warn!("origins_buf flush failed: {e}");
            }
        }
        self.dirty = false;
    }

    /// Record the GPU-driven chunk passes: dispatch the frustum-cull compute
    /// shader, then issue one `vkCmdDrawIndexedIndirect` per pass (opaque +
    /// transparent). Replaces the legacy per-chunk draw loop.
    pub fn record(&mut self, device: &ash::Device, cmd: vk::CommandBuffer,
        chunk_descriptor_set: vk::DescriptorSet, vp_cols: &[f32], game_time: f32,
        cam_pos: Vec3, query_pool: vk::QueryPool,
        opaque_end_ts: Option<u32>, transparent_end_ts: Option<u32>) {
        if self.dirty { self.rebuild(device); }
        let total = self.opaque_slots + self.transparent_slots;
        if total == 0 { return; }
        if let Ok(slice) = self.cull_ubo.mapped_slice_mut() {
            let mut a = [0f32; 16]; a.copy_from_slice(vp_cols);
            let ubo = CullUbo { view_proj: a, cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0] };
            slice[..std::mem::size_of::<CullUbo>()].copy_from_slice(bytemuck::bytes_of(&ubo));
            if let Err(e) = self.cull_ubo.flush_whole(device) {
                log::warn!("cull_ubo flush failed: {e}");
            }
        }
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                self.cull_pipeline_layout, 0, &[self.cull_descriptor_set], &[]);
            let dc = total as u32;
            device.cmd_push_constants(cmd, self.cull_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&dc));
            device.cmd_dispatch(cmd, (dc + 63) / 64, 1, 1);
        }
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(self.indirect_cmd_buf.buffer).offset(0).size(vk::WHOLE_SIZE)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        unsafe {
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT, vk::DependencyFlags::empty(),
                &[], &[barrier], &[]);
        }
        let mut push = [0f32; 20];
        push[0..16].copy_from_slice(vp_cols);
        push[16] = game_time;
        let stride = std::mem::size_of::<DrawIndexedIndirectCommand>() as u32;
        let sets = [chunk_descriptor_set, self.indirect_descriptor_set];
        let vbo = [self.mega_vbo.buffer];
        if self.opaque_slots > 0 {
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.opaque_pipeline);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                    self.indirect_pipeline_layout, 0, &sets, &[]);
                device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                device.cmd_bind_index_buffer(cmd, self.mega_ibo.buffer, 0, vk::IndexType::UINT32);
                device.cmd_push_constants(cmd, self.indirect_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX, 0, bytemuck::bytes_of(&push));
                device.cmd_draw_indexed_indirect(cmd, self.indirect_cmd_buf.buffer,
                    0, self.opaque_slots as u32, stride);
                if let Some(q) = opaque_end_ts {
                    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, query_pool, q);
                }
            }
        }
        if self.transparent_slots > 0 {
            let off = (self.opaque_slots * std::mem::size_of::<DrawIndexedIndirectCommand>()) as vk::DeviceSize;
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.transparent_pipeline);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                    self.indirect_pipeline_layout, 0, &sets, &[]);
                device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                device.cmd_bind_index_buffer(cmd, self.mega_ibo.buffer, 0, vk::IndexType::UINT32);
                device.cmd_push_constants(cmd, self.indirect_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX, 0, bytemuck::bytes_of(&push));
                device.cmd_draw_indexed_indirect(cmd, self.indirect_cmd_buf.buffer,
                    off, self.transparent_slots as u32, stride);
                if let Some(q) = transparent_end_ts {
                    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, query_pool, q);
                }
            }
        }
    }
}

impl GpuDriven {
    /// Destroy all Vulkan resources. Call before drop (mirrors
    /// [`crate::buffer::GpuBuffer::destroy`]).
    pub fn destroy(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_pipeline(self.opaque_pipeline, None);
            self.opaque_pipeline = vk::Pipeline::null();
            device.destroy_pipeline(self.transparent_pipeline, None);
            self.transparent_pipeline = vk::Pipeline::null();
            device.destroy_pipeline_layout(self.indirect_pipeline_layout, None);
            self.indirect_pipeline_layout = vk::PipelineLayout::null();
            device.destroy_descriptor_set_layout(self.indirect_set_layout, None);
            self.indirect_set_layout = vk::DescriptorSetLayout::null();
            device.destroy_descriptor_pool(self.indirect_descriptor_pool, None);
            device.destroy_pipeline(self.cull_pipeline, None);
            self.cull_pipeline = vk::Pipeline::null();
            device.destroy_pipeline_layout(self.cull_pipeline_layout, None);
            self.cull_pipeline_layout = vk::PipelineLayout::null();
            device.destroy_descriptor_set_layout(self.cull_set_layout, None);
            self.cull_set_layout = vk::DescriptorSetLayout::null();
            device.destroy_descriptor_pool(self.cull_descriptor_pool, None);
        }
        self.mega_vbo.destroy_in_place(device, alloc);
        self.mega_ibo.destroy_in_place(device, alloc);
        self.indirect_cmd_buf.destroy_in_place(device, alloc);
        self.aabb_buf.destroy_in_place(device, alloc);
        self.origins_buf.destroy_in_place(device, alloc);
        self.cull_ubo.destroy_in_place(device, alloc);
        self.dummy_tile_remap_ubo.destroy_in_place(device, alloc);
        if let Some(mut m) = self.gpu_mesher.take() { m.destroy(device, alloc); }
    }
}

// ── Phase 2: GPU compute mesher ───────────────────────────────────────────

const VOXEL_TEX_SIZE: u32 = 18;
const MESH_MAX_VERTS: vk::DeviceSize = (16 * 16 * 16) as vk::DeviceSize * 6 * 4;
const MESH_MAX_IDXS: vk::DeviceSize = (16 * 16 * 16) as vk::DeviceSize * 6 * 6;
const MESH_VERT_UINTS: vk::DeviceSize = 8;

pub struct GpuMesher {
    voxel_image: vk::Image,
    voxel_image_mem: Option<gpu_allocator::vulkan::Allocation>,
    voxel_view: vk::ImageView,
    voxel_sampler: vk::Sampler,
    voxel_staging: GpuBuffer,
    block_props: GpuBuffer,
    out_verts: GpuBuffer,
    out_idxs: GpuBuffer,
    counter: GpuBuffer,
    mesh_pipeline: vk::Pipeline,
    mesh_pipeline_layout: vk::PipelineLayout,
    mesh_set_layout: vk::DescriptorSetLayout,
    mesh_descriptor_pool: vk::DescriptorPool,
    mesh_descriptor_set: vk::DescriptorSet,
    block_count: u32,
}

/// Partially-constructed `GpuMesher`. Holds every Vulkan resource it has
/// successfully created, and runs the matching teardown on `Drop` if
/// construction fails partway through (e.g. shader compile error in
/// `create_compute_pipeline`). Without this guard, the 14 resources the
/// `GpuMesher::new` body allocates before that step would all be leaked
/// (the `GpuBuffer` / `GpuImage` `Drop` impls only WARN — they don't
/// actually free the GPU handle).
///
/// Idiom:
///   1. Allocate each resource in source order, assigning into the
///      corresponding `Option<...>` field on success. After that point,
///      an early `?` return triggers our cleanup.
///   2. After every step succeeds, call `.finalize()` to consume the
///      builder into a real `GpuMesher`. Step 11 in `GpuMesher::new` does
///      this; the no-fail path uses `Ok(b.finalize())`.
///
/// Notes on subtle invariants:
///   - `descriptor_set` is not explicitly destroyed in `Drop` because
///     `create_mesh_descriptor_pool` allocates the pool with
///     `FREE_DESCRIPTOR_SET` — destroying the pool auto-frees every set
///     that was allocated from it. Don't add explicit set cleanup here
///     without also restructuring the pool-creation flags or you'll
///     double-free.
///   - `finalize` uses `std::mem::take(&mut self.X)` rather than
///     `self.X.expect(...)`. Rust forbids destructuring/moving fields out
///     of any type that implements `Drop`, even via the `Option` enum's
///     `expect`, because the borrow checker can't verify the move leaves
///     `self` in a state matchable by the `Drop` impl. `mem::take` swaps
///     the field to `None` and returns the original, leaving `self` in a
///     valid (all-`None`) state for `Drop`.
struct PartialGpuMesher<'a> {
    image: Option<(vk::Image, gpu_allocator::vulkan::Allocation)>,
    view: Option<vk::ImageView>,
    sampler: Option<vk::Sampler>,
    staging: Option<GpuBuffer>,
    block_props: Option<GpuBuffer>,
    out_verts: Option<GpuBuffer>,
    out_idxs: Option<GpuBuffer>,
    counter: Option<GpuBuffer>,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set: Option<vk::DescriptorSet>,
    device: &'a ash::Device,
    alloc: &'a Alloc,
}

impl<'a> PartialGpuMesher<'a> {
    fn new(device: &'a ash::Device, alloc: &'a Alloc) -> Self {
        Self {
            image: None,
            view: None,
            sampler: None,
            staging: None,
            block_props: None,
            out_verts: None,
            out_idxs: None,
            counter: None,
            pipeline: None,
            pipeline_layout: None,
            set_layout: None,
            descriptor_pool: None,
            descriptor_set: None,
            device,
            alloc,
        }
    }

    /// Consume the partial builder and produce a `GpuMesher`. Only valid
    /// to call once every field has been populated by a successful step.
    /// Uses `std::mem::take` rather than direct move-out, because Rust
    /// forbids destructuring fields out of any type that implements
    /// `Drop`. After this returns, `self` is left in an all-`None` state
    /// so its `Drop` becomes a no-op.
    fn finalize(mut self) -> GpuMesher {
        // Pull each field out via mem::take (which replaces with `None`
        // and returns the original Option<T>) so the Drop impl on
        // `PartialGpuMesher` still typechecks. The replacement-None
        // values are then no-ops in `Drop::drop`.
        let (voxel_image, voxel_image_mem) = std::mem::take(&mut self.image)
            .expect("PartialGpuMesher::finalize without image: commit missing");
        let voxel_view = std::mem::take(&mut self.view)
            .expect("PartialGpuMesher::finalize without view: commit missing");
        let voxel_sampler = std::mem::take(&mut self.sampler)
            .expect("PartialGpuMesher::finalize without sampler: commit missing");
        let voxel_staging = std::mem::take(&mut self.staging)
            .expect("PartialGpuMesher::finalize without staging: commit missing");
        let block_props = std::mem::take(&mut self.block_props)
            .expect("PartialGpuMesher::finalize without block_props: commit missing");
        let out_verts = std::mem::take(&mut self.out_verts)
            .expect("PartialGpuMesher::finalize without out_verts: commit missing");
        let out_idxs = std::mem::take(&mut self.out_idxs)
            .expect("PartialGpuMesher::finalize without out_idxs: commit missing");
        let counter = std::mem::take(&mut self.counter)
            .expect("PartialGpuMesher::finalize without counter: commit missing");
        let mesh_pipeline = std::mem::take(&mut self.pipeline)
            .expect("PartialGpuMesher::finalize without pipeline: commit missing");
        let mesh_pipeline_layout = std::mem::take(&mut self.pipeline_layout)
            .expect("PartialGpuMesher::finalize without pipeline_layout: commit missing");
        let mesh_set_layout = std::mem::take(&mut self.set_layout)
            .expect("PartialGpuMesher::finalize without set_layout: commit missing");
        let mesh_descriptor_pool = std::mem::take(&mut self.descriptor_pool)
            .expect("PartialGpuMesher::finalize without descriptor_pool: commit missing");
        let mesh_descriptor_set = std::mem::take(&mut self.descriptor_set)
            .expect("PartialGpuMesher::finalize without descriptor_set: commit missing");
        GpuMesher {
            voxel_image,
            voxel_image_mem: Some(voxel_image_mem),
            voxel_view,
            voxel_sampler,
            voxel_staging,
            block_props,
            out_verts,
            out_idxs,
            counter,
            mesh_pipeline,
            mesh_pipeline_layout,
            mesh_set_layout,
            mesh_descriptor_pool,
            mesh_descriptor_set,
            block_count: 0,
        }
    }
}

impl<'a> Drop for PartialGpuMesher<'a> {
    fn drop(&mut self) {
        // Reverse-construction-order teardown. Anything still allocated at
        // the site of the `?`-induced early return gets destroyed here.
        // We idempotently no-op on `Some(null)` and tolerate null handles
        // for the same reason — using `destroy_*` on a destroy-handle that
        // was already freed is undefined, so we must only call it when
        // the handle genuinely points at something we created.
        unsafe {
            if let Some(p) = self.descriptor_pool {
                self.device.destroy_descriptor_pool(p, None);
            }
            if let Some(p) = self.pipeline {
                self.device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.pipeline_layout {
                self.device.destroy_pipeline_layout(p, None);
            }
            if let Some(s) = self.set_layout {
                self.device.destroy_descriptor_set_layout(s, None);
            }
        }
        // GpuBuffer-owned resources need both device + alloc to free memory.
        // `destroy_in_place` is idempotent on an already-empty buffer thanks
        // to the `buffer != null` check in its body, so these `take()`s are
        // safe even if some intermediate step already cleared them.
        if let Some(mut b) = self.counter.take() { b.destroy_in_place(self.device, self.alloc); }
        if let Some(mut b) = self.out_idxs.take() { b.destroy_in_place(self.device, self.alloc); }
        if let Some(mut b) = self.out_verts.take() { b.destroy_in_place(self.device, self.alloc); }
        if let Some(mut b) = self.block_props.take() { b.destroy_in_place(self.device, self.alloc); }
        if let Some(mut b) = self.staging.take() { b.destroy_in_place(self.device, self.alloc); }
        unsafe {
            if let Some(s) = self.sampler { self.device.destroy_sampler(s, None); }
            if let Some(v) = self.view { self.device.destroy_image_view(v, None); }
            if let Some((img, alloc)) = self.image.take() {
                self.device.destroy_image(img, None);
                self.alloc.free(alloc);
            }
        }
    }
}

impl GpuMesher {
    pub fn new(device: &ash::Device, alloc: &Alloc, command_pool: vk::CommandPool, graphics_queue: vk::Queue) -> Result<Self> {
        use crate::texture::{begin_one_time, end_and_submit, transition_image_layout};
        use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
        use gpu_allocator::MemoryLocation;

        let mut b = PartialGpuMesher::new(device, alloc);

        // 1. Voxel 3D image + device-local memory.
        //    `vk::Image` is Copy, so we can keep using `voxel_image` even after
        //    stashing a copy of it into `b.image`.
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D).format(vk::Format::R16_UINT)
            .extent(vk::Extent3D { width: VOXEL_TEX_SIZE, height: VOXEL_TEX_SIZE, depth: VOXEL_TEX_SIZE })
            .mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let voxel_image = unsafe { device.create_image(&img_info, None) }
            .map_err(|e| anyhow!("mesher create_image: {e:?}"))?;
        let reqs = unsafe { device.get_image_memory_requirements(voxel_image) };
        let voxel_image_mem = alloc.allocate(&AllocationCreateDesc {
            name: "voxel_tex", requirements: reqs,
            location: MemoryLocation::GpuOnly, linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(voxel_image, voxel_image_mem.memory(), voxel_image_mem.offset())
            .map_err(|e| anyhow!("mesher bind_image: {e:?}"))?; }
        b.image = Some((voxel_image, voxel_image_mem));

        // 2. Image view.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(voxel_image).view_type(vk::ImageViewType::TYPE_3D).format(vk::Format::R16_UINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0,
                level_count: 1, base_array_layer: 0, layer_count: 1,
            });
        let voxel_view = unsafe { device.create_image_view(&view_info, None) }
            .map_err(|e| anyhow!("mesher image_view: {e:?}"))?;
        b.view = Some(voxel_view);

        // 3. First-image layout transition (UNDEFINED -> SHADER_READ_ONLY).
        {
            let cmd = begin_one_time(device, command_pool)?;
            transition_image_layout(device, cmd, voxel_image,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR, 1, 1);
            end_and_submit(device, command_pool, graphics_queue, cmd)?;
        }

        // 4. Sampler.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let voxel_sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .map_err(|e| anyhow!("mesher sampler: {e:?}"))?;
        b.sampler = Some(voxel_sampler);

        // 5. Mesh staging + block props + output buffers + atomic counter.
        //    Each GpuBuffer is committed to `b.X` immediately after
        //    creation. The two-step `host_visible -> mapped_slice_mut` for
        //    `block_props` and `counter` would otherwise leak a buffer
        //    allocation if `mapped_slice_mut()` returned Err between the
        //    `host_visible` success point and the eventual commit step.
        let voxel_staging = GpuBuffer::host_visible(device, alloc,
            (VOXEL_TEX_SIZE * VOXEL_TEX_SIZE * VOXEL_TEX_SIZE * 2) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC, "voxel_staging")?;
        b.staging = Some(voxel_staging);
        let block_props = GpuBuffer::host_visible(device, alloc,
            256 * std::mem::size_of::<crate::BlockPropertiesGpu>() as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, "block_props")?;
        b.block_props = Some(block_props);
        let out_verts = GpuBuffer::device_local(device, alloc,
            MESH_MAX_VERTS * MESH_VERT_UINTS * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, "mesh_out_verts")?;
        b.out_verts = Some(out_verts);
        let out_idxs = GpuBuffer::device_local(device, alloc,
            MESH_MAX_IDXS * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, "mesh_out_idxs")?;
        b.out_idxs = Some(out_idxs);
        let counter = GpuBuffer::host_visible(device, alloc, 8,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, "mesh_counter")?;
        b.counter = Some(counter);
        // mapped_slice_mut on a freshly-created host_visible buffer is
        // effectively infallible (CpuToGpu allocations always have a
        // mapped slice from gpu_allocator), but the ? keeps the API
        // symmetric with the rest of the file. If it does fail, `b.counter`
        // is already Some so PartialGpuMesher::drop will free it cleanly.
        b.counter.as_mut()
            .ok_or_else(|| anyhow!("counter stashed but immediately missing"))?
            .mapped_slice_mut()?
            .fill(0);

        // 6. Mesh compute pipeline + descriptor plumbing. Each step commits
        //    to `b` immediately after success so a `?`-induced early return
        //    triggers `PartialGpuMesher::drop` to tear down whatever's in
        //    `b` so far. `vk::*` handle types are Copy, so re-reading from
        //    `b` later via `expect(...)` is a borrow, not a move.
        let mesh_spv: Vec<u8> = include_bytes!(concat!(env!("OUT_DIR"), "/chunk_mesh.comp.spv")).to_vec();
        let mesh_set_layout = create_mesh_set_layout(device)?;
        b.set_layout = Some(mesh_set_layout);
        let mesh_pipeline_layout = create_mesh_pipeline_layout(device, mesh_set_layout)?;
        b.pipeline_layout = Some(mesh_pipeline_layout);
        let mesh_pipeline = create_compute_pipeline(device, mesh_pipeline_layout, &mesh_spv)?;
        b.pipeline = Some(mesh_pipeline);
        let mesh_descriptor_pool = create_mesh_descriptor_pool(device)?;
        b.descriptor_pool = Some(mesh_descriptor_pool);
        let mesh_descriptor_set = allocate_set(device, mesh_descriptor_pool, mesh_set_layout)?;
        b.descriptor_set = Some(mesh_descriptor_set);
        update_mesh_set(
            device, mesh_descriptor_set,
            b.view.expect("view committed in step 2"),
            b.sampler.expect("sampler committed in step 4"),
            b.block_props.as_ref().expect("block_props committed in step 5").buffer,
            b.out_verts.as_ref().expect("out_verts committed in step 5").buffer,
            b.out_idxs.as_ref().expect("out_idxs committed in step 5").buffer,
            b.counter.as_ref().expect("counter committed in step 5").buffer,
        );
        log::info!("GPU compute mesher ready (voxel tex {}³, max {} verts)", VOXEL_TEX_SIZE, MESH_MAX_VERTS);
        Ok(b.finalize())
    }

    pub fn set_block_properties(&mut self, device: &ash::Device, alloc: &Alloc, props: &[crate::BlockPropertiesGpu]) {
        let needed = props.len() * std::mem::size_of::<crate::BlockPropertiesGpu>();
        if needed as vk::DeviceSize > self.block_props.size {
            self.block_props.destroy_in_place(device, alloc);
            self.block_props = match GpuBuffer::host_visible(device, alloc, needed as vk::DeviceSize,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, "block_props") {
                Ok(b) => b, Err(e) => { log::error!("block_props realloc: {e}"); return; }
            };
            update_mesh_set(device, self.mesh_descriptor_set, self.voxel_view, self.voxel_sampler,
                self.block_props.buffer, self.out_verts.buffer, self.out_idxs.buffer, self.counter.buffer);
        }
        if let Ok(slice) = self.block_props.mapped_slice_mut() {
            let n = needed.min(slice.len());
            slice[..n].copy_from_slice(bytemuck::cast_slice(props));
            if let Err(e) = self.block_props.flush_whole(device) {
                log::warn!("block_props flush failed: {e}");
            }
        }
        self.block_count = props.len() as u32;
    }

    pub fn mesh_chunk(
        &mut self, device: &ash::Device, _alloc: &Alloc,
        command_pool: vk::CommandPool, graphics_queue: vk::Queue,
        voxels: &[u16], pass_mode: u32,
        mega_vbo: vk::Buffer, mega_ibo: vk::Buffer,
        vbo_offset: vk::DeviceSize, ibo_offset: vk::DeviceSize,
    ) -> Option<(u32, u32)> {
        use crate::texture::{begin_one_time, end_and_submit};
        {
            let slice = self.voxel_staging.mapped_slice_mut().ok()?;
            let n = (voxels.len() * 2).min(slice.len());
            slice[..n].copy_from_slice(bytemuck::cast_slice(voxels));
            // GPU reads this via `cmd_copy_buffer_to_image` immediately
            // below — flush so the device sees the latest voxel bytes.
            if let Err(e) = self.voxel_staging.flush_whole(device) {
                log::warn!("voxel_staging flush failed: {e}");
            }
        }
        let cmd = begin_one_time(device, command_pool).ok()?;
        unsafe {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0).buffer_row_length(VOXEL_TEX_SIZE).buffer_image_height(VOXEL_TEX_SIZE)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width: VOXEL_TEX_SIZE, height: VOXEL_TEX_SIZE, depth: VOXEL_TEX_SIZE });
            device.cmd_copy_buffer_to_image(cmd, self.voxel_staging.buffer, self.voxel_image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, &[region]);
            device.cmd_fill_buffer(cmd, self.counter.buffer, 0, 8, 0);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.mesh_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                self.mesh_pipeline_layout, 0, &[self.mesh_descriptor_set], &[]);
            device.cmd_push_constants(cmd, self.mesh_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pass_mode));
            device.cmd_dispatch(cmd, 2, 2, 2);
            let barriers = [
                vk::BufferMemoryBarrier::default().buffer(self.out_verts.buffer).offset(0).size(vk::WHOLE_SIZE)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                vk::BufferMemoryBarrier::default().buffer(self.out_idxs.buffer).offset(0).size(vk::WHOLE_SIZE)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                vk::BufferMemoryBarrier::default().buffer(self.counter.buffer).offset(0).size(8)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
            ];
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[], &barriers, &[]);
        }
        end_and_submit(device, command_pool, graphics_queue, cmd).ok()?;
        let (vert_count, idx_count) = {
            let slice = self.counter.mapped_slice_mut().ok()?;
            let counts: &[u32] = bytemuck::cast_slice(&slice[..8]);
            (counts[0], counts[1])
        };
        if vert_count == 0 || idx_count == 0 { return Some((0, 0)); }
        let cmd = begin_one_time(device, command_pool).ok()?;
        unsafe {
            let v_bytes = vert_count as vk::DeviceSize * 32;
            let i_bytes = idx_count as vk::DeviceSize * 4;
            device.cmd_copy_buffer(cmd, self.out_verts.buffer, mega_vbo,
                &[vk::BufferCopy::default().src_offset(0).dst_offset(vbo_offset).size(v_bytes)]);
            device.cmd_copy_buffer(cmd, self.out_idxs.buffer, mega_ibo,
                &[vk::BufferCopy::default().src_offset(0).dst_offset(ibo_offset).size(i_bytes)]);
        }
        end_and_submit(device, command_pool, graphics_queue, cmd).ok()?;
        Some((vert_count, idx_count))
    }

    pub fn destroy(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_sampler(self.voxel_sampler, None);
            device.destroy_image_view(self.voxel_view, None);
            device.destroy_image(self.voxel_image, None);
            self.voxel_image = vk::Image::null();
            self.voxel_view = vk::ImageView::null();
            device.destroy_pipeline(self.mesh_pipeline, None);
            self.mesh_pipeline = vk::Pipeline::null();
            device.destroy_pipeline_layout(self.mesh_pipeline_layout, None);
            self.mesh_pipeline_layout = vk::PipelineLayout::null();
            device.destroy_descriptor_set_layout(self.mesh_set_layout, None);
            self.mesh_set_layout = vk::DescriptorSetLayout::null();
            device.destroy_descriptor_pool(self.mesh_descriptor_pool, None);
        }
        if let Some(m) = self.voxel_image_mem.take() { alloc.free(m); }
        self.voxel_staging.destroy_in_place(device, alloc);
        self.block_props.destroy_in_place(device, alloc);
        self.out_verts.destroy_in_place(device, alloc);
        self.out_idxs.destroy_in_place(device, alloc);
        self.counter.destroy_in_place(device, alloc);
    }
}

fn create_mesh_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default().binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default().binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default().binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default().binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default().binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| anyhow!("mesh_set_layout: {e:?}"))
}

fn create_mesh_pipeline_layout(device: &ash::Device, set_layout: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout> {
    let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0).size(std::mem::size_of::<u32>() as u32);
    let set_layouts = [set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push));
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| anyhow!("mesh_pl: {e:?}"))
}

fn create_mesh_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 },
    ];
    let info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| anyhow!("mesh_pool: {e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn update_mesh_set(
    device: &ash::Device, set: vk::DescriptorSet,
    voxel_view: vk::ImageView, voxel_sampler: vk::Sampler,
    props_buf: vk::Buffer, vert_buf: vk::Buffer, idx_buf: vk::Buffer, ctr_buf: vk::Buffer,
) {
    let img_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(voxel_view).sampler(voxel_sampler)];
    let props_info = [vk::DescriptorBufferInfo::default().buffer(props_buf).offset(0).range(vk::WHOLE_SIZE)];
    let vert_info = [vk::DescriptorBufferInfo::default().buffer(vert_buf).offset(0).range(vk::WHOLE_SIZE)];
    let idx_info = [vk::DescriptorBufferInfo::default().buffer(idx_buf).offset(0).range(vk::WHOLE_SIZE)];
    let ctr_info = [vk::DescriptorBufferInfo::default().buffer(ctr_buf).offset(0).range(8)];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&props_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&vert_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&idx_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ctr_info),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}