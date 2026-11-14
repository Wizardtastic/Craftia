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
pub mod ui_atlas;

pub use atlas::{build_atlas_with_textures, Atlas};
pub use hot_reload::{compile_shader, FileWatcher, HotReloadEvent};
pub use renderer::{
    ChunkUpload, FrameInput, GpuTimings, MeshPass, Renderer, RendererConfig,
    WATER_SURFACE_GRID_LAYERS, WATER_SURFACE_GRID_NONE, WATER_SURFACE_GRID_SIZE,
};
pub use texture::AtlasTexture;
pub use ui::{FontAtlas, GraphStyle, UiDrawData, UiVertex};
pub use ui_atlas::{
    ui_atlas, ui_tile_uv, NSLICE, TILE_ARMOR_BG, TILE_ARMOR_FULL, TILE_ARMOR_HALF, TILE_ARROW_DOWN,
    TILE_ARROW_UP, TILE_BTN, TILE_BTN_DISABLED, TILE_BTN_HOVER, TILE_BTN_PRESSED, TILE_BUBBLE_BG,
    TILE_BUBBLE_FULL, TILE_CROSSHAIR, TILE_HEART_BG, TILE_HEART_FULL, TILE_HEART_HALF,
    TILE_HUD_FRAME, TILE_HUNGER_BG, TILE_HUNGER_FULL, TILE_HUNGER_HALF, TILE_PANEL,
    TILE_PANEL_INSET, TILE_SCROLL_THUMB, TILE_SCROLL_TRACK, TILE_SEL_FRAME, TILE_SLOT,
    TILE_SLOT_HIGHLIGHT, TILE_TOOLTIP, UI_ATLAS_COLS, UI_ATLAS_ROWS, UI_ATLAS_TILES,
};

use bytemuck::{Pod, Zeroable};

/// GPU vertex layout (28 bytes), matching `voxel_world::mesh::ChunkVertex`.
///
/// Lighting is no longer carried in the vertex stream. The chunk fragment
/// shader ray traces the voxel world per pixel, so the per-vertex data is
/// geometry, tile UVs, the atlas tile index and packed water animation.
///
/// Mirror changes here in the vertex attribute arrays (chunk/entity pipelines)
/// and in `shaders/chunk.vert` / `shaders/chunk_indirect.vert`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// Atlas tile index; passed flat to the fragment shader.
    pub tile: u32,
    /// Packed water animation (see `voxel_world::water::WaterVertexInfo`).
    pub water: u32,
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
        let fre = (flags as u32) | ((roughness as u32) << 8) | ((emissive as u32) << 16);
        let sss = u32::from_le_bytes(sss_tint);
        let wet = u32::from_le_bytes(wet_tint);
        let abs =
            (absorption[0] as u32) | ((absorption[1] as u32) << 8) | ((absorption[2] as u32) << 16);
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
        if opaque {
            flags |= Self::FLAG_OPAQUE;
        }
        if liquid {
            flags |= Self::FLAG_LIQUID;
        }
        if not_air {
            flags |= Self::FLAG_RENDER;
        }
        Self {
            tiles01: tiles[0] as u32 | ((tiles[1] as u32) << 16),
            tiles23: tiles[2] as u32 | ((tiles[3] as u32) << 16),
            tiles45: tiles[4] as u32 | ((tiles[5] as u32) << 16),
            flags,
        }
    }
}

/// Per-block ray-tracing material, indexed by block ID in the GPU voxel grid.
///
/// 32 bytes, matching `BlockLightGpu` in `shaders/chunk.frag`. The padding
/// keeps the std140 array stride at 32 bytes (20 bytes would round up to 32
/// on the GPU and desynchronise every entry after the first).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BlockLightGpu {
    /// 0 = air/pass-through, 1 = opaque, 2 = water, 3 = translucent,
    /// 4 = foliage (partial sunlight transmission).
    pub kind: u32,
    /// Emissive light level 0..15.
    pub emission_level: u32,
    /// Packed RGB8 emissive colour (low byte = R).
    pub emission_rgb: u32,
    /// 0 = mirror, 255 = matte.
    pub roughness: u32,
    /// Packed RGB8 average albedo (from the block's map colour), used for
    /// one-bounce diffuse GI.
    pub albedo_rgb: u32,
    pub _pad: [u32; 3],
}

impl BlockLightGpu {
    pub const KIND_AIR: u32 = 0;
    pub const KIND_OPAQUE: u32 = 1;
    pub const KIND_WATER: u32 = 2;
    pub const KIND_TRANSLUCENT: u32 = 3;
    pub const KIND_FOLIAGE: u32 = 4;

    pub fn new(
        kind: u32,
        emission_level: u8,
        emission: [u8; 3],
        roughness: u8,
        albedo: [u8; 3],
    ) -> Self {
        Self {
            kind,
            emission_level: emission_level as u32,
            emission_rgb: emission[0] as u32
                | ((emission[1] as u32) << 8)
                | ((emission[2] as u32) << 16),
            roughness: roughness as u32,
            albedo_rgb: albedo[0] as u32
                | ((albedo[1] as u32) << 8)
                | ((albedo[2] as u32) << 16),
            _pad: [0; 3],
        }
    }
}

/// Block-light lookup table uploaded once per frame (chunk binding 9).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockLightTable {
    pub blocks: [BlockLightGpu; TILE_MATERIAL_TABLE_LEN],
}

impl Default for BlockLightTable {
    fn default() -> Self {
        Self {
            blocks: [BlockLightGpu::default(); TILE_MATERIAL_TABLE_LEN],
        }
    }
}

impl BlockLightTable {
    pub const SIZE_BYTES: usize = std::mem::size_of::<Self>();
}

/// Maximum number of dynamic emissive blocks uploaded to the shader per frame.
pub const MAX_EMISSIVE_LIGHTS: usize = 32;

/// One dynamic point light (torch, lava, glowstone) sourced from the loaded
/// voxel chunks. 32 bytes, matching `EmissiveLights` in `shaders/chunk.frag`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EmissiveLight {
    pub position: [f32; 3],
    /// Cutoff distance in blocks; intensity fades smoothly to zero here.
    pub range: f32,
    /// Linear RGB radiance colour.
    pub color: [f32; 3],
    /// Radiant intensity scale (already includes the emission level).
    pub intensity: f32,
}

/// Full ray-tracing lighting state for one frame, supplied by the engine's
/// astronomical day/night model.
#[derive(Clone, Copy, Debug)]
pub struct RaytraceLighting {
    /// Direction *towards* the sun (normalised).
    pub sun_dir: [f32; 3],
    /// Direct sun radiance in linear HDR units.
    pub sun_color: [f32; 3],
    /// Physical angular radius of the sun in radians (soft shadow penumbra).
    pub sun_angular_radius: f32,
    /// Direction *towards* the moon.
    pub moon_dir: [f32; 3],
    /// Moonlight radiance in linear HDR units.
    pub moon_color: [f32; 3],
    /// Moon angular radius in radians.
    pub moon_angular_radius: f32,
    /// Illuminated fraction of the moon (0 = new, 1 = full).
    pub moon_phase: f32,
    /// Sky radiance at the zenith / horizon (linear HDR).
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    /// 0 at day, 1 at night (stars, exposure lift).
    pub night: f32,
    /// Fog/horizon colour and density scale.
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    /// Ray-tracing quality 0..1 (affects ray counts and AO strength).
    pub quality: f32,
    /// Extra penumbra width in radians (0 = physical sun angular radius).
    pub shadow_softness: f32,
    /// Master reflection strength 0..1.
    pub reflection_strength: f32,
    /// Game time in seconds (animated water / caustics / dithering).
    pub time: f32,
    /// Camera is submerged.
    pub underwater: bool,
}

impl Default for RaytraceLighting {
    fn default() -> Self {
        Self {
            sun_dir: [0.0, 1.0, 0.0],
            sun_color: [3.3, 3.2, 3.0],
            sun_angular_radius: 0.0047,
            moon_dir: [0.0, -1.0, 0.0],
            moon_color: [0.0, 0.0, 0.0],
            moon_angular_radius: 0.0045,
            moon_phase: 0.5,
            sky_zenith: [0.145, 0.36, 0.86],
            sky_horizon: [0.61, 0.775, 0.98],
            night: 0.0,
            fog_color: [0.62, 0.78, 0.96],
            fog_density: 1.0,
            quality: 1.0,
            shadow_softness: 0.004,
            reflection_strength: 0.85,
            time: 0.0,
            underwater: false,
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
        Self {
            map: std::array::from_fn(|i| i as u32),
        }
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
