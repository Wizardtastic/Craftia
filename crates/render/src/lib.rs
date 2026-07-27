//! `voxel-render` — Vulkan rendering backend.
//!
//! See crate root docs. Public surface:
//! - [`Renderer`] — the high-level facade owned by the engine.
//! - [`atlas`] — name-based texture atlas loaded from PNG files.
//! - [`Vertex`] — the GPU vertex layout (28 bytes), matching the chunk mesher.
//!
//! The renderer is intentionally decoupled from `voxel-world`: it accepts chunk
//! meshes as raw bytes (`&[u8]` vertices + `&[u32]` indices) plus a chunk world
//! origin, so worldgen/meshing can evolve without touching the renderer.

pub mod alloc;
pub mod animation;
pub mod atlas;
pub mod buffer;
pub mod dynamic_texture;
pub mod entity;
pub mod hot_reload;
pub mod model;
pub mod overlay;
pub mod panorama;
pub mod particle;
pub mod renderer;
pub mod texture;
pub mod ui;

pub use atlas::{build_atlas_with_textures, Atlas};
pub use hot_reload::{compile_shader, FileWatcher, HotReloadEvent};
pub use renderer::{ChunkUpload, GpuTimings, MeshPass, Renderer, RendererConfig};
pub use texture::AtlasTexture;
pub use ui::{FontAtlas, UiDrawData, UiVertex};

use bytemuck::{Pod, Zeroable};


/// GPU vertex layout (32 bytes), matching `voxel_world::mesh::ChunkVertex`.
/// Kept here as the rendering-side contract; the mesher produces the same layout.
///
/// The `tile` field carries the per-vertex atlas tile index (`u32`, location 3).
/// It is passed `flat` from the vertex shader to the fragment shader and used
/// to pick the correct sub-tile from the 2D atlas via the shader-side
/// `(tile_origin + fract(uv)) / 16` calculation.
///
/// The `light_color` field carries a packed RGBA color for tinted lighting.
/// It is passed `flat` from the vertex shader to the fragment shader.
/// Mirror changes here in `crates/render/src/renderer.rs` `create_graphics_pipeline`
/// + `create_shadow_pipeline` `vertex_attributes` arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub light: f32,
    pub tile: u32,
    pub light_color: u32,
}

/// GPU vertex layout for skinned meshes (60 bytes).
/// Extends the base Vertex with joint indices and weights for skeletal animation.
/// Used as a second vertex buffer binding (binding 1) alongside the base Vertex (binding 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SkinnedVertex {
    /// Joint indices (4 bones per vertex).
    pub joint_indices: [u32; 4],
    /// Joint weights (4 weights, should sum to 1.0).
    pub joint_weights: [f32; 4],
}

/// Maximum number of joints per model for GPU skinning.
pub const MAX_JOINTS: usize = 64;

/// Joint matrices uploaded to GPU UBO for skinning.
/// Each matrix transforms from joint local space to model space.
/// Stored as flat array of floats for GPU compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct JointMatrices {
    /// Flat storage: 64 matrices × 16 floats each = 1024 floats.
    pub data: [f32; MAX_JOINTS * 16],
}

// SAFETY: JointMatrices is a plain f32 array, safe for Pod/Zeroable.
unsafe impl Pod for JointMatrices {}
unsafe impl Zeroable for JointMatrices {}

impl Default for JointMatrices {
    fn default() -> Self {
        Self {
            data: [0.0; MAX_JOINTS * 16],
        }
    }
}

/// GPU-side mirror of `voxel_world::registry::BlockMaterial`, packed to
/// 16 bytes (one `vec4` worth) so each tile entry in the chunk material UBO
/// sits on an std430 base alignment of 4. Layout matches the GLSL
/// declaration in `shaders/chunk.frag` exactly:
///
/// ```glsl
/// struct BlockMaterialGpu {
///     uint flags_roughness_emissive_pad; // bits 0..7 flags, 8..15 roughness, 16..23 emissive
///     uint sss_tint;                     // RGBA8
///     uint wet_tint;                     // RGBA8
///     uint absorption_pad;               // RGB8 absorption + 8-bit pad
/// };
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BlockMaterialGpu {
    pub flags_roughness_emissive_pad: u32,
    pub sss_tint: u32,
    pub wet_tint: u32,
    pub absorption_pad: u32,
}

impl BlockMaterialGpu {
    /// Pack a CPU-side `voxel_world::registry::BlockMaterial` (exposed as the
    /// [`BlockMaterialFields`] trait-input shape so render doesn't depend on
    /// the world crate's full type graph) into our std430-friendly 16-byte
    /// representation. Unused bits are zeroed so future uses don't fight
    /// with random memory.
    pub fn pack(
        flags: u8,
        roughness: u8,
        emissive: u8,
        sss_tint: [u8; 4],
        wet_tint: [u8; 4],
        absorption: [u8; 3],
    ) -> Self {
        let fre = (flags as u32)
            | ((roughness as u32) << 8)
            | ((emissive as u32) << 16);
        let sss = u32::from_le_bytes(sss_tint);
        let wet = u32::from_le_bytes(wet_tint);
        let abs = (absorption[0] as u32)
            | ((absorption[1] as u32) << 8)
            | ((absorption[2] as u32) << 16);
        Self {
            flags_roughness_emissive_pad: fre,
            sss_tint: sss,
            wet_tint: wet,
            absorption_pad: abs,
        }
    }
}

/// GPU-side block render properties for the Phase-2 compute mesher.
///
/// 16 bytes per block, matching the compute shader's `uvec4 props[]` layout:
/// - `.x` = tiles[0] | (tiles[1] << 16)   (NegX, PosX)
/// - `.y` = tiles[2] | (tiles[3] << 16)   (NegY, PosY)
/// - `.z` = tiles[4] | (tiles[5] << 16)   (NegZ, PosZ)
/// - `.w` = flags: bit0 = opaque, bit1 = liquid, bit2 = render (not air)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BlockPropertiesGpu {
    pub tiles01: u32,
    pub tiles23: u32,
    pub tiles45: u32,
    pub flags: u32,
}

impl BlockPropertiesGpu {
    /// Flags bit constants (must match the compute shader).
    pub const FLAG_OPAQUE: u32 = 1;
    pub const FLAG_LIQUID: u32 = 2;
    pub const FLAG_RENDER: u32 = 4;

    /// Build from per-face tile indices and block properties.
    pub fn pack(tiles: [u16; 6], opaque: bool, liquid: bool, not_air: bool) -> Self {
        let mut flags = 0u32;
        if opaque { flags |= Self::FLAG_OPAQUE; }
        if liquid { flags |= Self::FLAG_LIQUID; }
        if not_air { flags |= Self::FLAG_RENDER; }
        Self {
            tiles01: tiles[0] as u32 | ((tiles[1] as u32) << 16),
            tiles23: tiles[2] as u32 | ((tiles[3] as u32) << 16),
            tiles45: tiles[4] as u32 | ((tiles[5] as u32) << 16),
            flags,
        }
    }
}

/// Fixed number of atlas tiles the renderer can address in the MaterialTable
/// UBO. The texture atlas is `16×16 = 256` tiles so 256 is a hard upper
/// bound; bump this constant if the atlas layout grows.
pub const TILE_MATERIAL_TABLE_LEN: usize = 256;

/// Layout of the chunk material UBO at descriptor binding 5:
///
/// ```text
/// +0..256*16            : BlockMaterialGpu materials[256]  (tile-indexed lookup)
/// +256*16..256*16+16    : vec4 world_params
///                            .x = water surface Y (world units)
///                            .y = wet_edge strength
///                            .z = caustics strength
///                            .w = leaves SSS strength
/// ```
///
/// Total size = 256 × 16 + 16 = 4112 bytes. Updated by the engine once per
/// frame from the registry + configuration. `world_params` is appended to
/// the array because std430 arrays of u32 structs align to 4 and a
/// trailing vec4 also aligns to 16 — no padding shenanigans.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialTable {
    pub materials: [BlockMaterialGpu; TILE_MATERIAL_TABLE_LEN],
    pub world_params: [f32; 4],
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self {
            materials: [BlockMaterialGpu::default(); TILE_MATERIAL_TABLE_LEN],
            world_params: [0.0; 4],
        }
    }
}

impl MaterialTable {
    pub const SIZE_BYTES: usize = std::mem::size_of::<Self>();

    /// Build a default table covering every tile slot. Tile 0 (air) and any
    /// not-yet-defined tiles are zeroed — fine because air doesn't render and
    /// unknown tiles fall back to vanilla lighting.
    pub fn empty() -> Self {
        Self::default()
    }
}

impl JointMatrices {
    /// Set a joint matrix at the given index.
    pub fn set(&mut self, index: usize, mat: glam::Mat4) {
        if index < MAX_JOINTS {
            let arr = mat.to_cols_array();
            let base = index * 16;
            self.data[base..base + 16].copy_from_slice(&arr);
        }
    }

    /// Get the byte slice for uploading to GPU.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}

/// Tile remap table for animated block textures.
/// Maps canonical tile indices to current frame tile indices.
/// Size: 256 entries × u32 = 1024 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileRemap {
    pub map: [u32; 256],
}

impl Default for TileRemap {
    fn default() -> Self {
        // Identity mapping: each tile maps to itself.
        let mut map = [0u32; 256];
        for i in 0..256 {
            map[i] = i as u32;
        }
        Self { map }
    }
}

impl TileRemap {
    /// Set a remap entry: canonical tile -> current frame tile.
    pub fn set(&mut self, canonical: u32, current: u32) {
        if (canonical as usize) < 256 {
            self.map[canonical as usize] = current;
        }
    }

    /// Get the byte slice for uploading to GPU.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.map)
    }
}
