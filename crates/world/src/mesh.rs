//! Chunk meshing: greedy meshing with per-vertex AO and face culling.
//!
//! Two meshes are produced per chunk:
//! - `opaque`: stone, dirt, grass, wood, …
//! - `transparent`: water, leaves, glass
//!
//! Greedy meshing merges adjacent coplanar faces with the same block type,
//! light level, and AO pattern into larger quads. Water, foliage, and cactus
//! use per-block emission.
//!
//! Vertex layout (28 bytes):
//!   pos: [f32; 3]   — local block corner in [0, 16]
//!   uv:  [f32; 2]   — pre-baked LOCAL tile UV in [0, w] × [0, h]
//!                       (the fragment shader applies `fract()` per-tile
//!                       so multi-block merges repeat the same tile instead
//!                       of bleeding into neighbouring atlas tiles).
//!   light: f32       — baked face shading in [0, 1]
//!   tile: u32        — atlas tile index (carried `flat`; not interpolated)

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec2, Vec3 as GVec3};
use voxel_core::{math::chunk_origin, BlockId, CHUNK_SIZE};

use crate::chunk::Chunk;
use crate::registry::{BlockDef, BlockKind, BlockRegistry, Face};

// The chunk now stores torchlight_color directly as a packed RGBA8 u32
// (bit layout `(A<<24)|(B<<16)|(G<<8)|R`) which matches the shader's
// `R8G8B8A8_UNORM` attribute exactly. No more palette quantization at the
// mesher boundary — the chunk owns the colour and the mesher just forwards
// it to each vertex.

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct ChunkVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub light: f32,
    /// Atlas tile index used to select this quad's texture in the shader.
    /// The shader treats this as a `flat uint`; never interpolated.
    pub tile: u32,
    /// Packed RGBA light color. R,G,B in [0,255], A unused (255).
    /// Interpolated as `flat` in the shader for colored lighting.
    pub light_color: u32,
}

#[derive(Clone, Default, Debug)]
pub struct ChunkMesh {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
    pub fn vertex_bytes(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<ChunkVertex>()
    }
    pub fn index_bytes(&self) -> usize {
        self.indices.len() * std::mem::size_of::<u32>()
    }
}

#[derive(Clone, Default, Debug)]
pub struct ChunkMeshBundle {
    pub opaque: ChunkMesh,
    pub transparent: ChunkMesh,
}

impl ChunkMeshBundle {
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.transparent.is_empty()
    }
}

#[rustfmt::skip]
const FACE_GEO: [(GVec3, [Vec2; 4], f32); 6] = [
    (GVec3::new(0.0, 0.0, 0.0), [Vec2::new(0.0,1.0), Vec2::new(0.0,0.0), Vec2::new(1.0,0.0), Vec2::new(1.0,1.0)], 0.6),
    (GVec3::new(1.0, 0.0, 0.0), [Vec2::new(0.0,1.0), Vec2::new(1.0,1.0), Vec2::new(1.0,0.0), Vec2::new(0.0,0.0)], 0.6),
    (GVec3::new(0.0, 0.0, 0.0), [Vec2::new(0.0,0.0), Vec2::new(1.0,0.0), Vec2::new(1.0,1.0), Vec2::new(0.0,1.0)], 0.4),
    (GVec3::new(0.0, 1.0, 0.0), [Vec2::new(0.0,0.0), Vec2::new(0.0,1.0), Vec2::new(1.0,1.0), Vec2::new(1.0,0.0)], 1.0),
    (GVec3::new(0.0, 0.0, 0.0), [Vec2::new(1.0,1.0), Vec2::new(0.0,1.0), Vec2::new(0.0,0.0), Vec2::new(1.0,0.0)], 0.8),
    (GVec3::new(0.0, 0.0, 1.0), [Vec2::new(0.0,1.0), Vec2::new(1.0,1.0), Vec2::new(1.0,0.0), Vec2::new(0.0,0.0)], 0.8),
];

#[rustfmt::skip]
const FACE_CORNERS: [[GVec3; 4]; 6] = [
    [GVec3::new(0.,0.,1.), GVec3::new(0.,1.,1.), GVec3::new(0.,1.,0.), GVec3::new(0.,0.,0.)],
    [GVec3::new(0.,0.,0.), GVec3::new(0.,1.,0.), GVec3::new(0.,1.,1.), GVec3::new(0.,0.,1.)],
    [GVec3::new(0.,0.,0.), GVec3::new(1.,0.,0.), GVec3::new(1.,0.,1.), GVec3::new(0.,0.,1.)],
    [GVec3::new(0.,0.,0.), GVec3::new(0.,0.,1.), GVec3::new(1.,0.,1.), GVec3::new(1.,0.,0.)],
    [GVec3::new(1.,0.,0.), GVec3::new(0.,0.,0.), GVec3::new(0.,1.,0.), GVec3::new(1.,1.,0.)],
    [GVec3::new(0.,0.,0.), GVec3::new(1.,0.,0.), GVec3::new(1.,1.,0.), GVec3::new(0.,1.,0.)],
];

const S: usize = CHUNK_SIZE as usize;

#[derive(Clone, Copy)]
struct MaskCell {
    block_id: BlockId,
    ao: [f32; 4],
    block_light: u8,
    /// Packed RGBA8 light tint (same layout as `ChunkVertex.light_color`).
    light_color: u32,
}

const EMPTY_CELL: MaskCell = MaskCell {
    block_id: BlockId::AIR,
    ao: [1.0; 4],
    block_light: 0,
    light_color: 0,
};

pub struct ChunkMesher;

impl ChunkMesher {
    pub fn build(
        &self,
        chunk: &Chunk,
        reg: &BlockRegistry,
        sample: impl Fn(i32, i32, i32) -> BlockId,
        sample_water: impl Fn(i32, i32, i32) -> u8,
        sample_loaded: impl Fn(i32, i32, i32) -> bool,
        sun_dir: GVec3,
    ) -> ChunkMeshBundle {
        let mut bundle = ChunkMeshBundle::default();
        let nac = chunk.non_air_count();
        if nac == 0 {
            return bundle;
        }
        bundle.opaque.vertices.reserve(nac * 4);
        bundle.opaque.indices.reserve(nac * 6);
        bundle.transparent.vertices.reserve(nac);
        bundle.transparent.indices.reserve(nac * 2);
        let origin = chunk_origin(chunk.pos);

        for ly in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let id = chunk.get(lx, ly, lz);
                    if id.is_air() {
                        continue;
                    }
                    let def = reg.get(id);
                    if !def.is_rendered() {
                        continue;
                    }
                    match def.kind {
                        BlockKind::Foliage => {
                            emit_foliage_cross(&mut bundle, chunk, lx, ly, lz, def);
                        }
                        BlockKind::Liquid => {
                            let wl = chunk.get_water_level(lx, ly, lz);
                            if wl == 0 {
                                continue;
                            }
                            emit_water_block(
                                &mut bundle,
                                chunk,
                                reg,
                                &sample,
                                &sample_water,
                                &sample_loaded,
                                lx,
                                ly,
                                lz,
                                def,
                                wl,
                                origin,
                            );
                        }
                        _ if def.kind == BlockKind::Cactus => {
                            emit_cactus_block(
                                &mut bundle,
                                chunk,
                                reg,
                                &sample,
                                lx,
                                ly,
                                lz,
                                def,
                                origin,
                                sun_dir,
                            );
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut mask = [[EMPTY_CELL; S]; S];

        for fi in 0..6usize {
            let face = Face::ALL[fi];
            let n = face.normal();
            let fn_f = GVec3::new(n.x as f32, n.y as f32, n.z as f32);
            let diffuse = fn_f.dot(-sun_dir).max(0.0);
            let directional = (diffuse * 0.65 + 0.35).clamp(0.0, 1.0);

            for slice in 0..CHUNK_SIZE {
                for v in 0..S {
                    for u in 0..S {
                        mask[v][u] = build_mask_cell(
                            face,
                            slice,
                            u as i32,
                            v as i32,
                            chunk,
                            reg,
                            &sample,
                            &sample_loaded,
                            origin,
                        );
                    }
                }
                greedy_emit(
                    &mut bundle,
                    &mask,
                    face,
                    slice,
                    chunk,
                    reg,
                    directional,
                    origin,
                );
                for cell in mask.iter_mut().flat_map(|r| r.iter_mut()) {
                    *cell = EMPTY_CELL;
                }
            }
        }

        bundle
    }
}

fn mask_to_block(face: Face, slice: i32, u: i32, v: i32) -> (i32, i32, i32) {
    match face {
        Face::NegX | Face::PosX => (slice, v, u),
        Face::NegY | Face::PosY => (u, slice, v),
        Face::NegZ | Face::PosZ => (u, v, slice),
    }
}

fn build_mask_cell(
    face: Face,
    slice: i32,
    u: i32,
    v: i32,
    chunk: &Chunk,
    reg: &BlockRegistry,
    sample: &impl Fn(i32, i32, i32) -> BlockId,
    sample_loaded: &impl Fn(i32, i32, i32) -> bool,
    origin: glam::IVec3,
) -> MaskCell {
    let (lx, ly, lz) = mask_to_block(face, slice, u, v);
    let id = chunk.get(lx, ly, lz);
    if id.is_air() {
        return EMPTY_CELL;
    }
    let def = reg.get(id);
    if !def.is_rendered()
        || def.kind == BlockKind::Liquid
        || def.kind == BlockKind::Foliage
        || def.kind == BlockKind::Cactus
    {
        return EMPTY_CELL;
    }
    let wx = origin.x + lx;
    let wy = origin.y + ly;
    let wz = origin.z + lz;
    if !should_emit_face(face, lx, ly, lz, chunk, reg, sample, sample_loaded) {
        return EMPTY_CELL;
    }
    let n = face.normal();
    // For NegX / NegZ / NegY faces at chunk borders, should_emit_face
    // already verified the neighbour is air or not loaded — the
    // neighbour-sample + opaque check below would be a no-op. Skip it.
    let is_neg_border = (face == Face::NegX && lx == 0)
        || (face == Face::NegZ && lz == 0)
        || (face == Face::NegY && ly == 0 && chunk.pos.y() > 0);
    if !is_neg_border {
        let nlx = lx + n.x;
        let nly = ly + n.y;
        let nlz = lz + n.z;
        let neighbour = if nlx >= 0
            && nlx < CHUNK_SIZE
            && nly >= 0
            && nly < CHUNK_SIZE
            && nlz >= 0
            && nlz < CHUNK_SIZE
        {
            chunk.get(nlx, nly, nlz)
        } else {
            sample(wx + n.x, wy + n.y, wz + n.z)
        };
        let neighbour_def = reg.get(neighbour);
        if neighbour_def.opaque {
            return EMPTY_CELL;
        }
        if !neighbour.is_air() && neighbour_def.kind == def.kind {
            return EMPTY_CELL;
        }
    }
    let face_normal = IVec3::new(n.x, n.y, n.z);
    let corners = FACE_CORNERS[fi_from_face(face)];
    let chunk_sample = |x: i32, y: i32, z: i32| -> BlockId {
        let lx = x - origin.x;
        let ly = y - origin.y;
        let lz = z - origin.z;
        if lx >= 0 && lx < CHUNK_SIZE && ly >= 0 && ly < CHUNK_SIZE && lz >= 0 && lz < CHUNK_SIZE {
            chunk.get(lx, ly, lz)
        } else {
            sample(x, y, z)
        }
    };
    let ao = [
        crate::light::compute_vertex_ao(
            wx,
            wy,
            wz,
            face_normal,
            iv3(corners[0]),
            &chunk_sample,
            reg,
        ),
        crate::light::compute_vertex_ao(
            wx,
            wy,
            wz,
            face_normal,
            iv3(corners[1]),
            &chunk_sample,
            reg,
        ),
        crate::light::compute_vertex_ao(
            wx,
            wy,
            wz,
            face_normal,
            iv3(corners[2]),
            &chunk_sample,
            reg,
        ),
        crate::light::compute_vertex_ao(
            wx,
            wy,
            wz,
            face_normal,
            iv3(corners[3]),
            &chunk_sample,
            reg,
        ),
    ];
    let bl = chunk.get_combined_light(lx, ly, lz);
    let lc = chunk.get_combined_light_color(lx, ly, lz);
    MaskCell {
        block_id: id,
        ao,
        block_light: bl,
        light_color: lc,
    }
}

fn iv3(v: GVec3) -> IVec3 {
    IVec3::new(v.x as i32, v.y as i32, v.z as i32)
}

fn fi_from_face(face: Face) -> usize {
    match face {
        Face::NegX => 0,
        Face::PosX => 1,
        Face::NegY => 2,
        Face::PosY => 3,
        Face::NegZ => 4,
        Face::PosZ => 5,
    }
}

fn should_emit_face(
    face: Face,
    lx: i32,
    ly: i32,
    lz: i32,
    chunk: &Chunk,
    _reg: &BlockRegistry,
    sample: &impl Fn(i32, i32, i32) -> BlockId,
    sample_loaded: &impl Fn(i32, i32, i32) -> bool,
) -> bool {
    let origin = chunk_origin(chunk.pos);
    let wx = origin.x + lx;
    let wy = origin.y + ly;
    let wz = origin.z + lz;
    let n = face.normal();
    // Chunk-border face ownership convention: for each pair of adjacent
    // chunks, the chunk on the NEGATIVE side (lower coordinate) owns the
    // border face. This chunk is on the POSITIVE side, so its NegX / NegZ /
    // NegY faces must be suppressed whenever the owning neighbour is
    // loaded AND has a non-air block there — the neighbour's PosX face
    // handles rendering.
    //
    // When the neighbour block IS air the positive-side PosX face isn't
    // emitted (air blocks are skipped), so WE must emit the NegX face.
    //
    // The old code only suppressed when the neighbour was opaque, which
    // left both chunks emitting the same face for air/transparent
    // neighbours — causing GPU Z-fighting (constant flickering/flashing).
    match face {
        Face::NegX if lx == 0 => {
            if sample_loaded(wx - 1, wy, wz) {
                let nb = sample(wx + n.x, wy + n.y, wz + n.z);
                if !nb.is_air() {
                    return false; // neighbour has a real block — it owns the face
                }
                // neighbour is air — we must emit (positive side won't)
            }
        }
        Face::NegZ if lz == 0 => {
            if sample_loaded(wx, wy, wz - 1) {
                let nb = sample(wx + n.x, wy + n.y, wz + n.z);
                if !nb.is_air() {
                    return false;
                }
            }
        }
        Face::NegY if ly == 0 && chunk.pos.y() > 0 => {
            if sample_loaded(wx, wy - 1, wz) {
                let nb = sample(wx + n.x, wy + n.y, wz + n.z);
                if !nb.is_air() {
                    return false;
                }
            }
        }
        _ => {}
    }
    true
}

// Per-face AO merge pairs for horizontal merge (extending w along u axis).
// Each entry is [(a_ao_i, b_ao_i), (a_ao_j, b_ao_j)] where a is the rightmost
// cell and b is the new cell to the right.
const H_MERGE_PAIRS: [[(usize, usize); 2]; 6] = [
    [(0, 3), (1, 2)], // NegX
    [(3, 0), (2, 1)], // PosX
    [(1, 0), (2, 3)], // NegY
    [(3, 0), (2, 1)], // PosY
    [(0, 1), (3, 2)], // NegZ
    [(1, 0), (2, 3)], // PosZ
];

// Per-face AO merge pairs for vertical merge (extending h along v axis).
// Each entry is [(a_ao_i, b_ao_i), (a_ao_j, b_ao_j)] where a is the bottommost
// row cell and b is the new row above.
const V_MERGE_PAIRS: [[(usize, usize); 2]; 6] = [
    [(1, 0), (2, 3)], // NegX
    [(1, 0), (2, 3)], // PosX
    [(3, 0), (2, 1)], // NegY
    [(1, 0), (2, 3)], // PosY
    [(3, 0), (2, 1)], // NegZ
    [(3, 0), (2, 1)], // PosZ
];

// Per-face corner AO extraction map.
// For each merged quad corner (0-3), stores (ao_index, cell_corner).
// Cell corners: 0=BL(v,u), 1=BR(v,u+w-1), 2=TR(v+h-1,u+w-1), 3=TL(v+h-1,u).
const CORNER_AO_MAP: [[(usize, usize); 4]; 6] = [
    [(0, 1), (1, 2), (2, 3), (3, 0)], // NegX
    [(0, 0), (1, 3), (2, 2), (3, 1)], // PosX
    [(0, 0), (1, 1), (2, 2), (3, 3)], // NegY
    [(0, 0), (1, 3), (2, 2), (3, 1)], // PosY
    [(1, 0), (2, 3), (3, 2), (0, 1)], // NegZ
    [(1, 1), (2, 2), (3, 3), (0, 0)], // PosZ
];

fn can_merge(a: &MaskCell, b: &MaskCell, pairs: &[(usize, usize); 2]) -> bool {
    a.block_id == b.block_id
        && a.block_light == b.block_light
        && a.light_color == b.light_color
        && a.ao[pairs[0].0] == b.ao[pairs[0].1]
        && a.ao[pairs[1].0] == b.ao[pairs[1].1]
}

fn can_merge_down(
    top: &[MaskCell; S],
    bot: &[MaskCell; S],
    u: usize,
    w: usize,
    pairs: &[(usize, usize); 2],
) -> bool {
    for i in 0..w {
        let a = &top[u + i];
        let b = &bot[u + i];
        if a.block_id != b.block_id || a.block_light != b.block_light || a.light_color != b.light_color {
            return false;
        }
        if a.ao[pairs[0].0] != b.ao[pairs[0].1] || a.ao[pairs[1].0] != b.ao[pairs[1].1] {
            return false;
        }
    }
    true
}

fn cell_offset(corner: usize, w: usize, h: usize) -> (usize, usize) {
    match corner {
        0 => (0, 0),
        1 => (0, w - 1),
        2 => (h - 1, w - 1),
        3 => (h - 1, 0),
        _ => unreachable!(),
    }
}

fn merged_uvs(face: Face, w: u32, h: u32) -> [[f32; 2]; 4] {
    // LOCAL tile-repeat coordinates: `[0, w] × [0, h]` in a single tile.  The
    // fragment shader does `fract(uv)` and adds the per-tile atlas origin
    // (carried in the `tile` vertex attribute), so a `w × h` merged quad
    // repeats the same tile `w × h` times instead of bleeding into the
    // neighbouring tiles of the 2D atlas.  Corners must match the geometry
    // corner order produced by `merged_positions` so the U-axis follows u
    // and the V-axis follows v.
    //
    // V-axis convention: V=0 is image-top in atlas space (the atlas is
    // uploaded PNG-as-is, no Y-flip). For side faces (PosX/NegX/PosZ/NegZ)
    // the geometry's `v` axis maps directly to world Y, so we INVERT V so
    // that world Y=0 (block bottom) samples image-bottom and world Y=h
    // (block top) samples image-top. This is what makes directional art
    // like `grass_side.png` render right-side-up (grass at the top of the
    // block, dirt at the bottom). For top/bottom faces (PosY/NegY) the
    // `v` axis maps to world Z, not Y, so the V convention is unchanged.
    let wf = w as f32;
    let hf = h as f32;
    match face {
        // mask (u,v) → (X,Y,Z) = (u, slice, v); geometry order: (0,0), (0,h), (w,h), (w,0)
        Face::PosY => [[0.0, 0.0], [0.0, hf], [wf, hf], [wf, 0.0]],
        // mask (u,v) → (X,Y,Z) = (u, slice, v); geometry order: (0,0), (w,0), (w,h), (0,h)
        Face::NegY => [[0.0, 0.0], [wf, 0.0], [wf, hf], [0.0, hf]],
        // mask (u,v) → (X,Y,Z) = (slice, v, u); V INVERTED so y=0 → image bottom.
        Face::PosX => [[0.0, hf], [0.0, 0.0], [wf, 0.0], [wf, hf]],
        // mask (u,v) → (X,Y,Z) = (slice, v, u); V INVERTED so y=0 → image bottom.
        Face::NegX => [[wf, hf], [wf, 0.0], [0.0, 0.0], [0.0, hf]],
        // mask (u,v) → (X,Y,Z) = (u, v, slice); V INVERTED so y=0 → image bottom.
        Face::PosZ => [[wf, hf], [wf, 0.0], [0.0, 0.0], [0.0, hf]],
        // mask (u,v) → (X,Y,Z) = (u, v, slice); V INVERTED so y=0 → image bottom.
        Face::NegZ => [[0.0, hf], [0.0, 0.0], [wf, 0.0], [wf, hf]],
    }
}

fn merged_positions(face: Face, u0: i32, v0: i32, w: i32, h: i32, s: i32) -> [[f32; 3]; 4] {
    match face {
        Face::PosY => [
            [u0 as f32, (s + 1) as f32, v0 as f32],
            [u0 as f32, (s + 1) as f32, (v0 + h) as f32],
            [(u0 + w) as f32, (s + 1) as f32, (v0 + h) as f32],
            [(u0 + w) as f32, (s + 1) as f32, v0 as f32],
        ],
        Face::NegY => [
            [u0 as f32, s as f32, v0 as f32],
            [(u0 + w) as f32, s as f32, v0 as f32],
            [(u0 + w) as f32, s as f32, (v0 + h) as f32],
            [u0 as f32, s as f32, (v0 + h) as f32],
        ],
        Face::PosX => [
            [(s + 1) as f32, v0 as f32, u0 as f32],
            [(s + 1) as f32, (v0 + h) as f32, u0 as f32],
            [(s + 1) as f32, (v0 + h) as f32, (u0 + w) as f32],
            [(s + 1) as f32, v0 as f32, (u0 + w) as f32],
        ],
        Face::NegX => [
            [s as f32, v0 as f32, (u0 + w) as f32],
            [s as f32, (v0 + h) as f32, (u0 + w) as f32],
            [s as f32, (v0 + h) as f32, u0 as f32],
            [s as f32, v0 as f32, u0 as f32],
        ],
        Face::PosZ => [
            [(u0 + w) as f32, v0 as f32, (s + 1) as f32],
            [(u0 + w) as f32, (v0 + h) as f32, (s + 1) as f32],
            [u0 as f32, (v0 + h) as f32, (s + 1) as f32],
            [u0 as f32, v0 as f32, (s + 1) as f32],
        ],
        Face::NegZ => [
            [u0 as f32, v0 as f32, s as f32],
            [u0 as f32, (v0 + h) as f32, s as f32],
            [(u0 + w) as f32, (v0 + h) as f32, s as f32],
            [(u0 + w) as f32, v0 as f32, s as f32],
        ],
    }
}

fn greedy_emit(
    bundle: &mut ChunkMeshBundle,
    mask: &[[MaskCell; S]; S],
    face: Face,
    slice: i32,
    _chunk: &Chunk,
    reg: &BlockRegistry,
    directional: f32,
    _origin: glam::IVec3,
) {
    let fi = fi_from_face(face);
    let h_pairs = &H_MERGE_PAIRS[fi];
    let v_pairs = &V_MERGE_PAIRS[fi];
    let corner_map = &CORNER_AO_MAP[fi];
    let mut visited = [[false; S]; S];

    for v in 0..S {
        for u in 0..S {
            if visited[v][u] {
                continue;
            }
            let cell = &mask[v][u];
            if cell.block_id.is_air() {
                continue;
            }

            let block_light = cell.block_light;
            let tile = reg.get(cell.block_id).textures.tile(face);

            let mut w = 1usize;
            while u + w < S
                && !visited[v][u + w]
                && can_merge(&mask[v][u + w - 1], &mask[v][u + w], h_pairs)
            {
                w += 1;
            }

            let mut h = 1usize;
            'outer: while v + h < S {
                for i in 0..w {
                    if visited[v + h][u + i] {
                        break 'outer;
                    }
                }
                if !can_merge_down(&mask[v + h - 1], &mask[v + h], u, w, v_pairs) {
                    break;
                }
                h += 1;
            }

            for dy in 0..h {
                for dx in 0..w {
                    visited[v + dy][u + dx] = true;
                }
            }

            let corner_ao: [f32; 4] = std::array::from_fn(|c| {
                let (ao_idx, cell_corner) = corner_map[c];
                let (dv, du) = cell_offset(cell_corner, w, h);
                mask[v + dv][u + du].ao[ao_idx]
            });

            let light = directional * (block_light as f32 / 15.0).max(0.15);
            let positions = merged_positions(face, u as i32, v as i32, w as i32, h as i32, slice);
            let uvs = merged_uvs(face, w as u32, h as u32);
            // `cell.light_color` is already a packed RGBA8 u32 ready for the
            // vertex shader. The palette-quantization step that used to live
            // here (`pack_light_color`) is gone — torchlight colour is stored
            // verbatim in the chunk from the BFS through to the vertex.
            let packed_color = cell.light_color;

            let target = match reg.get(cell.block_id).kind {
                BlockKind::Transparent => &mut bundle.transparent,
                _ => &mut bundle.opaque,
            };
            let base = target.vertices.len() as u32;

            if corner_ao[0] + corner_ao[2] > corner_ao[1] + corner_ao[3] {
                for c in 0..4 {
                    target.vertices.push(ChunkVertex {
                        pos: positions[c],
                        uv: uvs[c],
                        light: light * corner_ao[c],
                        tile: tile as u32,
                        light_color: packed_color,
                    });
                }
                target.indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 3,
                    base + 1,
                    base + 2,
                    base + 3,
                ]);
            } else {
                for c in 0..4 {
                    target.vertices.push(ChunkVertex {
                        pos: positions[c],
                        uv: uvs[c],
                        light: light * corner_ao[c],
                        tile: tile as u32,
                        light_color: packed_color,
                    });
                }
                target.indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base,
                    base + 2,
                    base + 3,
                ]);
            }
        }
    }
}

fn emit_foliage_cross(
    bundle: &mut ChunkMeshBundle,
    chunk: &Chunk,
    lx: i32,
    ly: i32,
    lz: i32,
    def: &BlockDef,
) {
    let tile = def.textures.tile(Face::PosY);
    let block_light = chunk.get_combined_light(lx, ly, lz) as f32 / 15.0;
    let light = block_light.max(0.2);
    let bx = lx as f32;
    let by = ly as f32;
    let bz = lz as f32;
    let w = 0.45f32;
    let cx = bx + 0.5;
    let cz = bz + 0.5;
    // Two perpendicular UPRIGHT (vertical) planar billboards, joined at the
    // block's central vertical axis. Previously this built two diagonal
    // planes tilted 45° around Y (an X-shape from above), which made flowers
    // and tall grass look slanted; the new shape is a `+` from above with
    // each plane edge-on only in its own normal direction.
    //
    //   Plane 1: lies in the XY plane at z = cz (outward normal = +Z).
    //   Plane 2: lies in the YZ plane at x = cx (outward normal = -X).
    let positions: [[f32; 3]; 8] = [
        // Plane 1 (z = cz): bottom-left, bottom-right, top-right, top-left.
        [cx - w, by, cz],
        [cx + w, by, cz],
        [cx + w, by + 1.0, cz],
        [cx - w, by + 1.0, cz],
        // Plane 2 (x = cx): bottom-near, bottom-far, top-far, top-near.
        [cx, by, cz - w],
        [cx, by, cz + w],
        [cx, by + 1.0, cz + w],
        [cx, by + 1.0, cz - w],
    ];
    let uvs: [[f32; 2]; 8] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ];
    let tgt = &mut bundle.transparent;
    let base = tgt.vertices.len() as u32;
    let packed_color = chunk.get_combined_light_color(lx, ly, lz);
    for i in 0..8 {
        // Local tile-repeat UVs `[0, 1] × [0, 1]`; the shader applies
        // `fract()` per tile so neighbouring atlas tiles are never sampled.
        tgt.vertices.push(ChunkVertex {
            pos: positions[i],
            uv: [uvs[i][0], uvs[i][1]],
            light,
            tile: tile as u32,
            light_color: packed_color,
        });
    }
    tgt.indices.extend_from_slice(&[
        base,
        base + 1,
        base + 2,
        base,
        base + 2,
        base + 3,
        base + 4,
        base + 5,
        base + 6,
        base + 4,
        base + 6,
        base + 7,
    ]);
}

fn emit_water_block(
    bundle: &mut ChunkMeshBundle,
    chunk: &Chunk,
    reg: &BlockRegistry,
    sample: &impl Fn(i32, i32, i32) -> BlockId,
    sample_water: &impl Fn(i32, i32, i32) -> u8,
    sample_loaded: &impl Fn(i32, i32, i32) -> bool,
    lx: i32,
    ly: i32,
    lz: i32,
    def: &BlockDef,
    water_level: u8,
    origin: glam::IVec3,
) {
    let water_f = water_level as f32;
    let height_frac = (water_f / 8.0).clamp(0.0, 1.0);
    let wx = origin.x + lx;
    let wy = origin.y + ly;
    let wz = origin.z + lz;
    let tile = def.textures.tile(Face::PosY);

    for (fi, face) in Face::ALL.iter().enumerate() {
        if !should_emit_face(*face, lx, ly, lz, chunk, reg, sample, sample_loaded) {
            continue;
        }
        let n = face.normal();
        // For NegX / NegZ / NegY faces at chunk borders, should_emit_face
        // already verified the neighbour is air or not loaded. Skip the
        // redundant cross-chunk sample — use AIR as a dummy since all
        // checks below (opaque, same-liquid, solid) are no-ops for air.
        let is_neg_border = (*face == Face::NegX && lx == 0)
            || (*face == Face::NegZ && lz == 0)
            || (*face == Face::NegY && ly == 0 && chunk.pos.y() > 0);
        let neighbour = if is_neg_border {
            BlockId::AIR
        } else {
            sample(wx + n.x, wy + n.y, wz + n.z)
        };
        let neighbour_def = reg.get(neighbour);
        if neighbour_def.opaque {
            continue;
        }
        if !neighbour.is_air() && neighbour_def.kind == BlockKind::Liquid {
            let nw = sample_water(wx + n.x, wy + n.y, wz + n.z);
            if nw == water_level {
                continue;
            }
        }
        if *face == Face::NegY && neighbour_def.kind == BlockKind::Solid {
            continue;
        }
        if *face != Face::PosY && *face != Face::NegY && neighbour_def.kind == BlockKind::Solid {
            continue;
        }

        let (base, uvs, _) = FACE_GEO[fi];
        let mut p_base = GVec3::new(lx as f32, ly as f32, lz as f32) + base;
        let y_scale = if *face != Face::PosY && *face != Face::NegY {
            let nw = if neighbour_def.kind == BlockKind::Liquid {
                sample_water(wx + n.x, wy + n.y, wz + n.z)
            } else {
                0
            };
            (water_level.max(nw)) as f32 / 8.0
        } else {
            1.0
        };
        if *face == Face::PosY {
            p_base.y = ly as f32 + height_frac;
        }

        let corners = FACE_CORNERS[fi];
        let start = bundle.transparent.vertices.len() as u32;
        let packed_color = chunk.get_combined_light_color(lx, ly, lz);
        for c in 0..4 {
            let mut cp = p_base + corners[c];
            if *face != Face::PosY && *face != Face::NegY {
                cp.y = ly as f32 + corners[c].y * y_scale;
            }
            if *face == Face::PosY && water_level == 8 {
                cp.y -= 2.0 / 16.0;
            }
            let vertex_light = 1.0 + (water_f / 8.0) * 0.5;
            bundle.transparent.vertices.push(ChunkVertex {
                pos: [cp.x, cp.y, cp.z],
                uv: [uvs[c].x, uvs[c].y],
                light: vertex_light,
                tile: tile as u32,
                light_color: packed_color,
            });
        }
        bundle.transparent.indices.extend_from_slice(&[
            start,
            start + 1,
            start + 2,
            start,
            start + 2,
            start + 3,
        ]);
    }
}

fn emit_cactus_block(
    bundle: &mut ChunkMeshBundle,
    chunk: &Chunk,
    reg: &BlockRegistry,
    sample: &impl Fn(i32, i32, i32) -> BlockId,
    lx: i32,
    ly: i32,
    lz: i32,
    def: &BlockDef,
    origin: glam::IVec3,
    sun_dir: GVec3,
) {
    let wx = origin.x + lx;
    let wy = origin.y + ly;
    let wz = origin.z + lz;
    let block_light = chunk.get_combined_light(lx, ly, lz) as f32 / 15.0;

    for (fi, face) in Face::ALL.iter().enumerate() {
        if !should_emit_face(*face, lx, ly, lz, chunk, reg, sample, &|_, _, _| false) {
            continue;
        }
        let n = face.normal();
        let neighbour = sample(wx + n.x, wy + n.y, wz + n.z);
        if reg.get(neighbour).opaque {
            continue;
        }

        let (base, uvs, _) = FACE_GEO[fi];
        let fn_f = GVec3::new(n.x as f32, n.y as f32, n.z as f32);
        let diffuse = fn_f.dot(-sun_dir).max(0.0);
        let directional = (diffuse * 0.65 + 0.35).clamp(0.0, 1.0);
        let light = directional * block_light.max(0.15);

        let tile = def.textures.tile(*face);
        let face_normal = IVec3::new(n.x, n.y, n.z);
        let corners = FACE_CORNERS[fi];
        let p_base = GVec3::new(lx as f32, ly as f32, lz as f32) + base;
        // Cactus is inset 2px (of 16) on every side, like Minecraft. Side
        // faces are pushed inward along their normal; top/bottom faces are
        // pulled in on all four horizontal edges. This keeps the block visibly
        // smaller than a full cube (so you see the ground around its base).
        const INSET: f32 = 2.0 / 16.0;

        let start = bundle.opaque.vertices.len() as u32;
        let packed_color = chunk.get_combined_light_color(lx, ly, lz);
        for c in 0..4 {
            let mut cp = p_base + corners[c];
            match face {
                Face::PosX => cp.x = lx as f32 + 1.0 - INSET,
                Face::NegX => cp.x = lx as f32 + INSET,
                Face::PosZ => cp.z = lz as f32 + 1.0 - INSET,
                Face::NegZ => cp.z = lz as f32 + INSET,
                Face::PosY | Face::NegY => {
                    // Inset the horizontal footprint; leave the Y (set by base).
                    cp.x = (lx as f32 + 0.5) + (corners[c].x - 0.5) * (1.0 - 2.0 * INSET);
                    cp.z = (lz as f32 + 0.5) + (corners[c].z - 0.5) * (1.0 - 2.0 * INSET);
                }
            }
            let ao = crate::light::compute_vertex_ao(
                wx,
                wy,
                wz,
                face_normal,
                IVec3::new(
                    corners[c].x as i32,
                    corners[c].y as i32,
                    corners[c].z as i32,
                ),
                sample,
                reg,
            );
            bundle.opaque.vertices.push(ChunkVertex {
                pos: [cp.x, cp.y, cp.z],
                uv: [uvs[c].x, uvs[c].y],
                light: light * ao,
                tile: tile as u32,
                light_color: packed_color,
            });
        }
        bundle.opaque.indices.extend_from_slice(&[
            start,
            start + 1,
            start + 2,
            start,
            start + 2,
            start + 3,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use crate::registry::BlockRegistry;
    use glam::Vec3;
    use voxel_core::ChunkPos;

    fn air_sample(_x: i32, _y: i32, _z: i32) -> BlockId {
        BlockId::AIR
    }
    fn air_water_sample(_x: i32, _y: i32, _z: i32) -> u8 {
        0
    }
    fn air_loaded_sample(_x: i32, _y: i32, _z: i32) -> bool {
        false
    }

    fn chunk_with_water(lx: i32, ly: i32, lz: i32, level: u8) -> (Chunk, BlockRegistry) {
        let reg = BlockRegistry::with_builtins();
        let water_id = reg.id_of("water").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(lx, ly, lz, water_id);
        chunk.set_water_level(lx, ly, lz, level);
        (chunk, reg)
    }

    #[test]
    fn water_level_zero_produces_no_vertices() {
        let (chunk, reg) = chunk_with_water(7, 7, 7, 0);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert!(bundle.transparent.is_empty());
    }

    #[test]
    fn water_source_full_height() {
        let (chunk, reg) = chunk_with_water(7, 7, 7, 8);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert_eq!(bundle.transparent.vertices.len(), 24);
        assert_eq!(bundle.transparent.indices.len(), 36);
        let max_y = bundle
            .transparent
            .vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!(
            (max_y - 8.0).abs() < 0.01,
            "max y should be 8.0, got {}",
            max_y
        );
        for v in &bundle.transparent.vertices {
            assert!(
                (v.light - 1.5).abs() < 0.01,
                "light should be ~1.5, got {}",
                v.light
            );
        }
    }

    #[test]
    fn water_half_height() {
        let (chunk, reg) = chunk_with_water(7, 7, 7, 4);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert_eq!(bundle.transparent.vertices.len(), 24);
        let max_y = bundle
            .transparent
            .vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!(
            (max_y - 7.5).abs() < 0.01,
            "max y should be 7.5, got {}",
            max_y
        );
        for v in &bundle.transparent.vertices {
            assert!(
                (v.light - 1.25).abs() < 0.01,
                "light should be ~1.25, got {}",
                v.light
            );
        }
    }

    #[test]
    fn water_shallow_level_one() {
        let (chunk, reg) = chunk_with_water(7, 7, 7, 1);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert_eq!(bundle.transparent.vertices.len(), 24);
        let max_y = bundle
            .transparent
            .vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!(
            (max_y - 7.125).abs() < 0.01,
            "max y should be 7.125, got {}",
            max_y
        );
    }

    #[test]
    fn water_faces_emitted_count() {
        let (chunk, reg) = chunk_with_water(7, 7, 7, 8);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert_eq!(bundle.transparent.vertices.len(), 24);
        assert_eq!(bundle.transparent.indices.len(), 36);
    }

    #[test]
    fn greedy_single_block() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(7, 7, 7, stone);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert_eq!(
            bundle.opaque.vertices.len(),
            24,
            "single block = 6 faces = 24 verts"
        );
        assert_eq!(
            bundle.opaque.indices.len(),
            36,
            "single block = 6 faces = 36 indices"
        );
    }

    #[test]
    fn greedy_flat_layer() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        for x in 0..16i32 {
            for z in 0..16i32 {
                chunk.set(x, 0, z, stone);
            }
        }
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        // Top + bottom = 2×(16×16) quads = 8 verts
        // 4 side strips = 4×(16×1) quads = 16 verts
        // Total = 6 quads = 24 verts, 36 indices
        assert_eq!(bundle.opaque.vertices.len(), 24);
        assert_eq!(bundle.opaque.indices.len(), 36);
    }

    #[test]
    fn greedy_two_block_types_no_merge() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let dirt = reg.id_of("dirt").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(0, 0, 0, stone);
        chunk.set(1, 0, 0, dirt);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        // 2 blocks side by side: shared face is culled (neighbour is opaque).
        // Each block has 5 visible faces = 10 quads = 40 verts, 60 indices.
        assert_eq!(bundle.opaque.vertices.len(), 40);
        assert_eq!(bundle.opaque.indices.len(), 60);
    }

    #[test]
    fn merged_uvs_geometry_matches() {
        // Regression test for the atlas UV bleed/splatter bug.
        //
        // For a merged strip of w×h blocks of the same block type, `merged_uvs`
        // must produce LOCAL tile-repeat UVs in `[0, w] × [0, h]` (the
        // fragment shader applies `fract()` to wrap inside a single tile).
        // Critically, the UVs must NOT span atlas coordinates (i.e. must not
        // cover neighbouring tiles) — otherwise a 2-block-wide merged stone
        // quad would sample from the dirt tile next to it.
        //
        // We also assert the corner order matches the geometry corner order
        // produced by `merged_positions` for each face so flipped or rotated
        // textures don't leak in.
        for &face in &Face::ALL {
            for &(w, h) in &[(1u32, 1u32), (2, 1), (1, 2), (3, 2), (4, 1)] {
                let uvs = merged_uvs(face, w, h);
                let wf = w as f32;
                let hf = h as f32;

                let u_min = uvs.iter().map(|uv| uv[0]).fold(f32::INFINITY, f32::min);
                let u_max = uvs.iter().map(|uv| uv[0]).fold(f32::NEG_INFINITY, f32::max);
                let v_min = uvs.iter().map(|uv| uv[1]).fold(f32::INFINITY, f32::min);
                let v_max = uvs.iter().map(|uv| uv[1]).fold(f32::NEG_INFINITY, f32::max);

                // Local UVs must range over [0, w] × [0, h] exactly: never
                // spilling into the next atlas tile.
                assert!(
                    (u_min - 0.0).abs() < 1e-6,
                    "{:?} w={} h={}: U should start at 0, got min={}",
                    face,
                    w,
                    h,
                    u_min,
                );
                assert!(
                    (v_min - 0.0).abs() < 1e-6,
                    "{:?} w={} h={}: V should start at 0, got min={}",
                    face,
                    w,
                    h,
                    v_min,
                );
                assert!(
                    (u_max - u_min - wf).abs() < 1e-6,
                    "{:?} w={} h={}: U span should be w = {}, got {}",
                    face,
                    w,
                    h,
                    wf,
                    u_max - u_min,
                );
                assert!(
                    (v_max - v_min - hf).abs() < 1e-6,
                    "{:?} w={} h={}: V span should be h = {}, got {}",
                    face,
                    w,
                    h,
                    hf,
                    v_max - v_min,
                );
                assert!(
                    (u_max - 0.0 - wf).abs() < 1e-6,
                    "{:?} w={} h={}: U should end at w, got max={}",
                    face,
                    w,
                    h,
                    u_max,
                );
                assert!(
                    (v_max - 0.0 - hf).abs() < 1e-6,
                    "{:?} w={} h={}: V should end at h, got max={}",
                    face,
                    w,
                    h,
                    v_max,
                );
            }
        }
    }

    #[test]
    fn merged_uvs_side_face_v_axis_inverted() {
        // Regression test for the "grass blocks upside down" bug.
        //
        // For side faces (PosX / NegX / PosZ / NegZ) the geometry's `v` axis
        // maps directly to world Y, so V=0 should sample IMAGE TOP and V=h
        // IMAGE BOTTOM in atlas space. The atlas is uploaded WITHOUT a
        // Y-flip, so atlas row 0 = image top = grass (on `grass_side.png`)
        // and atlas row 15 = image bottom = dirt. Therefore:
        //
        //     world Y = 0 (block bottom)  ↔  V = h  (image bottom = dirt)
        //     world Y = h (block top)     ↔  V = 0  (image top    = grass)
        //
        // Top/bottom faces (PosY / NegY) keep the ORIGINAL V convention
        // because their geometry's `v` axis maps to world Z (not Y);
        // flipping V there would mirror textures along the Z axis instead.
        //
        // We cross-check against `merged_positions` so this test actually
        // fails if someone reverts the fix (a range-only assertion would
        // pass both before and after the V-flip).
        for &face in &[Face::PosX, Face::NegX, Face::PosZ, Face::NegZ] {
            for &(w, h) in &[(1u32, 1u32), (2, 1), (1, 2), (3, 4)] {
                let positions = merged_positions(face, 0, 0, w as i32, h as i32, 0);
                let uvs = merged_uvs(face, w, h);
                let hf = h as f32;
                for c in 0..4 {
                    let world_y = positions[c][1];
                    let uv_v = uvs[c][1];
                    if (world_y - 0.0).abs() < 1e-6 {
                        // corner at block bottom must sample image bottom.
                        assert!(
                            (uv_v - hf).abs() < 1e-6,
                            "{:?} corner {} (world Y=0) V should be h={}, got {}",
                            face,
                            c,
                            hf,
                            uv_v,
                        );
                    } else if (world_y - h as f32).abs() < 1e-6 {
                        // corner at block top must sample image top.
                        assert!(
                            (uv_v - 0.0).abs() < 1e-6,
                            "{:?} corner {} (world Y=h={}) V should be 0, got {}",
                            face,
                            c,
                            h as f32,
                            uv_v,
                        );
                    }
                }
            }
        }
        // Top/bottom faces should NOT have been changed by the side-face
        // V-flip. Sample one quad to be sure.
        let p_uvs = merged_uvs(Face::PosY, 1, 1);
        assert!((p_uvs[0][1] - 0.0).abs() < 1e-6);
        assert!((p_uvs[2][1] - 1.0).abs() < 1e-6);
        let n_uvs = merged_uvs(Face::NegY, 1, 1);
        assert!((n_uvs[0][1] - 0.0).abs() < 1e-6);
        assert!((n_uvs[2][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn foliage_cross_planes_are_vertical() {
        // Regression test for the "foliage slanted at 45 degrees" bug.
        // Previously `emit_foliage_cross` built two diagonal planes tilted
        // 45° around the block's Y axis (an X-shape from above). The fix
        // makes the planes UPRIGHT: a vertical plane at z=cz (outward +Z)
        // and a vertical plane at x=cx (outward -X), joined at the central
        // vertical axis. So every vertex must lie on EITHER x=cx OR z=cz
        // (never both varying), and its y must be exactly by or by+1.0
        // (never interpolated to a tilted angle).
        let reg = BlockRegistry::with_builtins();
        let poppy = reg.id_of("poppy").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(0, 0, 0, poppy);
        let mesher = ChunkMesher;
        let bundle = mesher.build(
            &chunk,
            &reg,
            air_sample,
            air_water_sample,
            air_loaded_sample,
            Vec3::new(0.0, -1.0, 0.0),
        );
        // Two perpendicular planes, four vertices each → 8 verts total.
        assert_eq!(
            bundle.transparent.vertices.len(),
            8,
            "foliage cross should be 2 perpendicular planes × 4 verts",
        );
        let cx = 0.5f32;
        let cz = 0.5f32;
        for v in &bundle.transparent.vertices {
            let p = v.pos;
            let on_x_plane = (p[0] - cx).abs() < 1e-5;
            let on_z_plane = (p[2] - cz).abs() < 1e-5;
            assert!(
                on_x_plane || on_z_plane,
                "foliage vertex {:?} not on z={} plane or x={} plane",
                p,
                cz,
                cx,
            );
            // No tilted-Y: the vertex's y must be exactly the block's
            // bottom (0) or top (1) — never a fractional interpolation.
            assert!(
                (p[1] - 0.0).abs() < 1e-5 || (p[1] - 1.0).abs() < 1e-5,
                "foliage vertex {:?} has tilted y (expected 0 or 1 exactly)",
                p,
            );
        }
    }

    #[test]
    fn merged_uvs_no_atlas_bleed() {
        // Direct regression test for the splatter bug: for any merge size
        // and any tile index, merged_uvs must NOT span atlas coordinates
        // outside the LOCAL `[0, w] × [0, h]` box. Previously the function
        // produced `[tu .. tu+w/16] × [tv .. tv+h/16]`, which on a 2D atlas
        // caused a 2-block-wide merged stone quad (tile=1) to bleed into
        // dirt (tile=2) on the right half of the quad.
        for tile in [0u16, 1, 5, 17, 27, 38] {
            for face in Face::ALL {
                for (w, h) in [(1u32, 1u32), (2, 1), (1, 4), (3, 2), (4, 1)] {
                    let uvs = merged_uvs(face, w, h);
                    for uv in uvs {
                        assert!(
                            uv[0] >= 0.0 && uv[0] <= w as f32 + 1e-6,
                            "tile={} face={:?} w={} h={}: U={} out of [0,w]",
                            tile,
                            face,
                            w,
                            h,
                            uv[0],
                        );
                        assert!(
                            uv[1] >= 0.0 && uv[1] <= h as f32 + 1e-6,
                            "tile={} face={:?} w={} h={}: V={} out of [0,h]",
                            tile,
                            face,
                            w,
                            h,
                            uv[1],
                        );
                    }
                }
            }
        }
    }
}
