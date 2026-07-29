//! glTF model loading and GPU mesh storage.
//!
//! Loads `.glb` (binary) and `.gltf` (JSON) files via the `gltf` crate,
//! extracts vertex/index data, and uploads to device-local GPU buffers.
//! Supports node hierarchy and skeletal animation (skins, joints, weights).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ash::vk;
use glam::Mat4;

use crate::alloc::Alloc;
use crate::buffer::GpuBuffer;

/// Loaded GPU mesh data for a single glTF primitive.
pub struct ModelMesh {
    pub vbo: GpuBuffer,
    pub ibo: GpuBuffer,
    pub index_count: u32,
    pub vertex_count: u32,
    pub material_index: u32,
    /// Optional skin data for this mesh.
    pub skin: Option<MeshSkinData>,
}

impl ModelMesh {
    pub fn destroy(self, device: &ash::Device, alloc: &Alloc) {
        self.vbo.destroy(device, alloc);
        self.ibo.destroy(device, alloc);
        if let Some(skin) = self.skin {
            if let Some(skin_vbo) = skin.skin_vbo {
                skin_vbo.destroy(device, alloc);
            }
        }
    }
}

/// Per-mesh skin data: joint indices and weights for each vertex.
pub struct MeshSkinData {
    /// Which skin this mesh uses (index into Model.skins).
    pub skin_index: u32,
    /// CPU copy of joint indices per vertex: [u32; 4] per vertex.
    pub joint_indices: Vec<[u32; 4]>,
    /// CPU copy of joint weights per vertex: [f32; 4] per vertex.
    pub joint_weights: Vec<[f32; 4]>,
    /// GPU buffer for skin data (joint_indices + joint_weights interleaved).
    pub skin_vbo: Option<GpuBuffer>,
}

/// A skin definition from glTF: joints and their inverse bind matrices.
#[derive(Clone, Debug)]
pub struct Skin {
    /// Node indices that are joints.
    pub joints: Vec<usize>,
    /// Inverse bind matrices for each joint.
    pub inverse_bind_matrices: Vec<Mat4>,
    /// Root joint node index.
    pub skeleton: Option<usize>,
}

/// All meshes from one glTF file, grouped by node.
pub struct Model {
    pub meshes: Vec<ModelMesh>,
    pub nodes: Vec<ModelNode>,
    pub skins: Vec<Skin>,
    pub node_parents: Vec<Option<usize>>,
}

impl Model {
    pub fn destroy(self, device: &ash::Device, alloc: &Alloc) {
        for mesh in self.meshes {
            mesh.destroy(device, alloc);
        }
    }
}

/// A node in the glTF scene tree. References a mesh by index and
/// stores local transform + children.
pub struct ModelNode {
    pub mesh_index: Option<u32>,
    pub children: Vec<usize>,
    pub local_transform: glam::Mat4,
    pub name: Option<String>,
    /// If this node is a joint, which skin it belongs to.
    pub joint_skin_index: Option<u32>,
}

/// In-memory vertex data parsed from glTF accessors.
struct ModelVertexData {
    positions: Vec<[f32; 3]>,
    #[allow(dead_code)]
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    material_index: u32,
    /// Optional joint indices per vertex (for skinning).
    joint_indices: Option<Vec<[u32; 4]>>,
    /// Optional joint weights per vertex (for skinning).
    joint_weights: Option<Vec<[f32; 4]>>,
}

/// Load a glTF model from disk and upload to GPU.
///
/// `device`, `alloc`, `pool`, `queue` are used for GPU buffer creation
/// and staging copy (same pattern as chunk mesh upload).
pub fn load_model(
    device: &ash::Device,
    alloc: &Alloc,
    pool: vk::CommandPool,
    queue: vk::Queue,
    path: &Path,
) -> Result<Model> {
    let (document, buffers, _images) = gltf::import(path)
        .with_context(|| format!("load_model: failed to load {}", path.display()))?;

    let mut meshes = Vec::new();
    let mut nodes = Vec::new();

    // Build joint-to-skin mapping: for each node, which skin it belongs to.
    let mut node_to_skin: Vec<Option<u32>> = vec![None; document.nodes().count()];
    for (skin_idx, skin) in document.skins().enumerate() {
        for joint in skin.joints() {
            node_to_skin[joint.index()] = Some(skin_idx as u32);
        }
    }

    // Parse all meshes.
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let material_index = primitive.material().index().unwrap_or(0) as u32;

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| anyhow!("mesh primitive missing POSITION accessor"))?
                .collect();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            // Read joint indices (JOINTS_0) and weights (WEIGHTS_0) for skinning.
            let joint_indices: Option<Vec<[u32; 4]>> = reader
                .read_joints(0)
                .map(|iter| {
                    iter.into_u16()
                        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                        .collect()
                });

            let joint_weights: Option<Vec<[f32; 4]>> = reader
                .read_weights(0)
                .map(|iter| iter.into_f32().collect());

            let vertex_data = ModelVertexData {
                positions,
                normals,
                uvs,
                indices,
                material_index,
                joint_indices,
                joint_weights,
            };

            // Determine which skin this primitive belongs to.
            let skin_index = mesh
                .primitives()
                .next()
                .and_then(|_p| {
                    // Find the node that references this mesh.
                    document.nodes().find(|n| {
                        n.mesh().map(|m| m.index()) == Some(mesh.index())
                    })
                })
                .and_then(|n| node_to_skin[n.index()]);

            let model_mesh = upload_mesh(device, alloc, pool, queue, &vertex_data, skin_index)?;
            meshes.push(model_mesh);
        }
    }

    // Parse node hierarchy.
    for node in document.nodes() {
        let mesh_index = node.mesh().map(|m| m.index() as u32);
        let children: Vec<usize> = node.children().map(|c| c.index()).collect();
        let local_transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        let name = node.name().map(String::from);
        let joint_skin_index = node_to_skin[node.index()];

        nodes.push(ModelNode {
            mesh_index,
            children,
            local_transform,
            name,
            joint_skin_index,
        });
    }

    // Parse skins.
    let mut skins = Vec::new();
    for skin in document.skins() {
        let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();

        // Read inverse bind matrices.
        let inverse_bind_matrices: Vec<Mat4> = if let Some(accessor) = skin.inverse_bind_matrices() {
            let view = accessor.view().unwrap();
            let mut data = vec![0u8; accessor.count() * accessor.size()];
            let offset = view.offset() + accessor.offset();
            let src = &buffers[view.buffer().index()];
            let src_slice = &src[offset..offset + data.len()];
            data.copy_from_slice(src_slice);

            data.chunks_exact(64)
                .map(|chunk| {
                    let arr: [f32; 16] = bytemuck::cast_slice(chunk).try_into().unwrap_or([0.0; 16]);
                    Mat4::from_cols_array(&arr)
                })
                .collect()
        } else {
            vec![Mat4::IDENTITY; joints.len()]
        };

        let skeleton = skin.skeleton().map(|s| s.index());

        skins.push(Skin {
            joints,
            inverse_bind_matrices,
            skeleton,
        });
    }

    // Build parent map.
    let mut node_parents = vec![None; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        for &child in &node.children {
            if child < node_parents.len() {
                node_parents[child] = Some(i);
            }
        }
    }

    Ok(Model {
        meshes,
        nodes,
        skins,
        node_parents,
    })
}

/// Upload a single mesh primitive to GPU buffers.
fn upload_mesh(
    device: &ash::Device,
    alloc: &Alloc,
    pool: vk::CommandPool,
    queue: vk::Queue,
    data: &ModelVertexData,
    skin_index: Option<u32>,
) -> Result<ModelMesh> {
    // Interleave vertex data into the engine's Vertex format (28 bytes).
    // pos(12) + uv(8) + light(4) + tile(4) = 28 bytes per vertex.
    let vertex_count = data.positions.len();
    let mut vertex_bytes = Vec::with_capacity(vertex_count * 28);
    for i in 0..vertex_count {
        let pos = data.positions[i];
        let uv = data.uvs[i];
        // Pack as: [f32;3] pos, [f32;2] uv, f32 light, u32 tile
        vertex_bytes.extend_from_slice(&pos[0].to_le_bytes());
        vertex_bytes.extend_from_slice(&pos[1].to_le_bytes());
        vertex_bytes.extend_from_slice(&pos[2].to_le_bytes());
        vertex_bytes.extend_from_slice(&uv[0].to_le_bytes());
        vertex_bytes.extend_from_slice(&uv[1].to_le_bytes());
        vertex_bytes.extend_from_slice(&1.0f32.to_le_bytes()); // light = 1.0
        vertex_bytes.extend_from_slice(&0u32.to_le_bytes()); // tile = 0
    }

    let index_count = data.indices.len();
    let index_bytes: Vec<u8> = data
        .indices
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();

    // Create device-local buffers.
    let vbo = GpuBuffer::device_local(
        device,
        alloc,
        vertex_bytes.len() as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        "model_vbo",
    )?;
    let ibo = GpuBuffer::device_local(
        device,
        alloc,
        index_bytes.len() as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        "model_ibo",
    )?;

    // Build skin data if present.
    let skin_data = if let (Some(joints), Some(weights), Some(skin_idx)) =
        (&data.joint_indices, &data.joint_weights, skin_index)
    {
        // Interleave joint indices (4x u32 = 16 bytes) + weights (4x f32 = 16 bytes) = 32 bytes per vertex.
        let mut skin_bytes = Vec::with_capacity(vertex_count * 32);
        for i in 0..vertex_count {
            let ji = joints[i];
            let jw = weights[i];
            skin_bytes.extend_from_slice(&ji[0].to_le_bytes());
            skin_bytes.extend_from_slice(&ji[1].to_le_bytes());
            skin_bytes.extend_from_slice(&ji[2].to_le_bytes());
            skin_bytes.extend_from_slice(&ji[3].to_le_bytes());
            skin_bytes.extend_from_slice(&jw[0].to_le_bytes());
            skin_bytes.extend_from_slice(&jw[1].to_le_bytes());
            skin_bytes.extend_from_slice(&jw[2].to_le_bytes());
            skin_bytes.extend_from_slice(&jw[3].to_le_bytes());
        }

        Some((skin_bytes, skin_idx, joints.clone(), weights.clone()))
    } else {
        None
    };

    // Staging buffer (host-visible) for the copy.
    let skin_byte_count = skin_data.as_ref().map(|(sb, _, _, _)| sb.len()).unwrap_or(0);
    let staging_size = (vertex_bytes.len() + index_bytes.len() + skin_byte_count) as vk::DeviceSize;
    let mut staging = GpuBuffer::host_visible(
        device,
        alloc,
        staging_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        "model_staging",
    )?;

    // Upload to staging.
    {
        let slice = staging.mapped_slice_mut()?;
        let mut offset = 0;
        slice[offset..offset + vertex_bytes.len()].copy_from_slice(&vertex_bytes);
        offset += vertex_bytes.len();
        slice[offset..offset + index_bytes.len()].copy_from_slice(&index_bytes);
        offset += index_bytes.len();
        if let Some((ref skin_bytes, _, _, _)) = skin_data {
            slice[offset..offset + skin_bytes.len()].copy_from_slice(skin_bytes);
        }
        // GPU `vkCmdCopyBuffer` follows below — flush the writes so the
        // device sees the latest bytes. On HOST_COHERENT memory this is a
        // no-op; on non-coherent (the case that produced the
        // "flashing white chunks" symptom) it's required.
        if let Err(e) = staging.flush_whole(device) {
            log::warn!("model staging flush failed: {e}");
        }
    }

    // Copy from staging to device-local via one-time command buffer.
    let cmd = crate::texture::begin_one_time(device, pool)?;

    let mut buffer_offset = 0u64;

    let v_region = vk::BufferCopy::default()
        .src_offset(buffer_offset)
        .dst_offset(0)
        .size(vertex_bytes.len() as vk::DeviceSize);
    unsafe {
        device.cmd_copy_buffer(cmd, staging.buffer, vbo.buffer, &[v_region]);
    }
    buffer_offset += vertex_bytes.len() as u64;

    let i_region = vk::BufferCopy::default()
        .src_offset(buffer_offset)
        .dst_offset(0)
        .size(index_bytes.len() as vk::DeviceSize);
    unsafe {
        device.cmd_copy_buffer(cmd, staging.buffer, ibo.buffer, &[i_region]);
    }
    buffer_offset += index_bytes.len() as u64;

    // Create and upload skin VBO if present.
    let skin_vbo = if let Some((ref skin_bytes, _, _, _)) = skin_data {
        let buf = GpuBuffer::device_local(
            device,
            alloc,
            skin_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            "model_skin_vbo",
        )?;
        let s_region = vk::BufferCopy::default()
            .src_offset(buffer_offset)
            .dst_offset(0)
            .size(skin_bytes.len() as vk::DeviceSize);
        unsafe {
            device.cmd_copy_buffer(cmd, staging.buffer, buf.buffer, &[s_region]);
        }
        Some(buf)
    } else {
        None
    };

    crate::texture::end_and_submit(device, pool, queue, cmd)?;
    staging.destroy(device, alloc);

    let skin = skin_data.map(|(_, skin_idx, joint_indices, joint_weights)| MeshSkinData {
        skin_index: skin_idx,
        joint_indices,
        joint_weights,
        skin_vbo,
    });

    Ok(ModelMesh {
        vbo,
        ibo,
        index_count: index_count as u32,
        vertex_count: vertex_count as u32,
        material_index: data.material_index,
        skin,
    })
}
