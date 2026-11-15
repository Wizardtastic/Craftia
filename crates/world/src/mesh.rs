//! Chunk meshing: greedy meshing with face culling.
//!
//! Two meshes are produced per chunk:
//! - `opaque`: stone, dirt, grass, wood, ...
//! - `transparent`: water, leaves, glass
//!
//! Greedy meshing merges adjacent coplanar faces of the same block type into
//! larger quads. The pass is bitmask-driven: one voxel scan fills per-axis
//! cube/opaque occupancy masks, each face slice reduces to an `and-not` that
//! yields its candidate cells, and the emitter clears consumed bits instead of
//! tracking a visited array. Neighbour data comes from a padded 18³ snapshot,
//! so the inner loops never re-enter the world lock or the registry.
//!
//! No lighting is baked into the geometry: the GPU ray tracer evaluates
//! sun/moon shadows, ambient occlusion, sky GI and reflections per pixel
//! against the world voxel grid.
//!
//! Vertex layout (16 bytes):
//!   pos:   [u16; 3] - local block corner in [0, 16], fixed point 1/256
//!   uv:    [u16; 2] - LOCAL tile UV in [0, w] x [0, h], fixed point 1/256;
//!                     the fragment shader applies `fract()` per tile.
//!   tile:  u16      - atlas tile index (flat, never interpolated).
//!   water: u32      - packed water animation (kind, surface flag, amplitude,
//!                     flow direction/speed, rapids); 0 for non-water geometry.
//!                     Natural alignment puts it at byte 12 with no padding.
//!
//! Fixed point represents every mesher position/UV exactly: voxel geometry
//! only uses multiples of 1/16, and water heights use 1/255 steps.

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec2, Vec3 as GVec3};
use voxel_core::{
    math::chunk_origin,
    BlockId, CHUNK_SIZE,
};

use crate::chunk::Chunk;
use crate::registry::{material_flag, BlockDef, BlockKind, BlockRegistry, Face};
use crate::water::WaterVertexInfo;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct ChunkVertex {
    pub pos: [u16; 3],
    pub uv: [u16; 2],
    pub tile: u16,
    /// Packed water animation as described in [`WaterVertexInfo::pack`].
    /// Sits at byte 12: `u32` alignment needs two bytes of padding after the
    /// 10-byte `uv` field, which the 2-byte `tile` exactly fills.
    pub water: u32,
}

/// Fixed-point scale for chunk positions and UVs (1/256 block).
const VERTEX_SCALE: f32 = 256.0;

impl ChunkVertex {
    pub fn new(pos: [f32; 3], uv: [f32; 2], tile: u32, water: u32) -> Self {
        let q = |v: f32| (v * VERTEX_SCALE).round().clamp(0.0, u16::MAX as f32) as u16;
        Self {
            pos: [q(pos[0]), q(pos[1]), q(pos[2])],
            uv: [q(uv[0]), q(uv[1])],
            tile: tile as u16,
            water,
        }
    }

    pub fn position(&self) -> [f32; 3] {
        [
            self.pos[0] as f32 / VERTEX_SCALE,
            self.pos[1] as f32 / VERTEX_SCALE,
            self.pos[2] as f32 / VERTEX_SCALE,
        ]
    }

    pub fn uv(&self) -> [f32; 2] {
        [
            self.uv[0] as f32 / VERTEX_SCALE,
            self.uv[1] as f32 / VERTEX_SCALE,
        ]
    }
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

    /// Append another mesh, rebasing its indices.
    pub fn append(&mut self, other: &ChunkMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices
            .extend(other.indices.iter().map(|i| i + base));
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

// ── Geometry constants ───────────────────────────────────────────────
/// Half-width of the foliage cross planes, in blocks.
const FOLIAGE_PLANE_WIDTH: f32 = 0.45;
/// Cactus sides are pulled in from the block edge by this amount.
const CACTUS_INSET: f32 = 3.0 / 16.0;

// Base corner offsets and per-face UVs used by the block-shaped liquid and
// cactus emitters. Keep geometry positions independent from texture data.
#[rustfmt::skip]
const FACE_BASE: [GVec3; 6] = [
    GVec3::new(0.0, 0.0, 0.0),
    GVec3::new(1.0, 0.0, 0.0),
    GVec3::new(0.0, 0.0, 0.0),
    GVec3::new(0.0, 1.0, 0.0),
    GVec3::new(0.0, 0.0, 0.0),
    GVec3::new(0.0, 0.0, 1.0),
];

#[rustfmt::skip]
const FACE_UVS: [[Vec2; 4]; 6] = [
    [Vec2::new(0.0,1.0), Vec2::new(0.0,0.0), Vec2::new(1.0,0.0), Vec2::new(1.0,1.0)],
    [Vec2::new(0.0,1.0), Vec2::new(1.0,1.0), Vec2::new(1.0,0.0), Vec2::new(0.0,0.0)],
    [Vec2::new(0.0,0.0), Vec2::new(1.0,0.0), Vec2::new(1.0,1.0), Vec2::new(0.0,1.0)],
    [Vec2::new(0.0,0.0), Vec2::new(0.0,1.0), Vec2::new(1.0,1.0), Vec2::new(1.0,0.0)],
    [Vec2::new(1.0,1.0), Vec2::new(0.0,1.0), Vec2::new(0.0,0.0), Vec2::new(1.0,0.0)],
    [Vec2::new(0.0,1.0), Vec2::new(1.0,1.0), Vec2::new(1.0,0.0), Vec2::new(0.0,0.0)],
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

/// Blocks in one chunk (used for bulk capacity reservations).
const BLOCKS: usize = S * S * S;

/// Padded 18³ copy of a chunk: the 16³ interior plus a one-voxel shell sampled
/// from the six neighbouring chunks. Face visibility is resolved against this
/// grid, so the greedy pass never touches the world lock or the registry.
///
/// `ids` uses `UNLOADED` as a sentinel for "neighbour chunk not loaded", which
/// removes the second loaded-flag array entirely. Borders are sampled lazily:
/// only cells whose interior neighbour is a greedy cube can ever be read.
struct PadGrid {
    ids: Vec<u16>,
}

/// Sentinel for a padded cell whose neighbour chunk has not loaded.
const UNLOADED: u16 = u16::MAX;

impl PadGrid {
    #[inline]
    fn index(x: i32, y: i32, z: i32) -> usize {
        ((z as usize) * 18 + (y as usize)) * 18 + (x as usize)
    }

    fn build<Smp, Ld>(chunk: &Chunk, props: &[CubeProp], sample: &Smp, sample_loaded: &Ld) -> Self
    where
        Smp: Fn(i32, i32, i32) -> BlockId,
        Ld: Fn(i32, i32, i32) -> bool,
    {
        const N: usize = 18;
        let mut ids = vec![0u16; N * N * N];

        // Interior: resolve the palette once for all 4096 voxels.
        let mut blocks = vec![0u16; BLOCKS];
        chunk.copy_blocks_into(&mut blocks);
        for y in 0..S {
            for z in 0..S {
                let src = (y * S + z) * S;
                let dst = Self::index(1, y as i32 + 1, z as i32 + 1);
                ids[dst..dst + S].copy_from_slice(&blocks[src..src + S]);
            }
        }

        // Six face planes: exactly the neighbour cells a face lookup can read.
        // Edges and corners of the shell are never queried (a face test only
        // steps one axis). A plane is only sampled where the interior block
        // bordering it is a cube, which keeps the sample count proportional to
        // the exposed surface instead of the full 16×16 shell.
        let origin = chunk_origin(chunk.pos);
        for (axis, sign) in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
            for v in 0..S {
                for u in 0..S {
                    let (lx, ly, lz) = match axis {
                        0 => (if sign < 0 { -1 } else { S as i32 }, v as i32, u as i32),
                        1 => (u as i32, if sign < 0 { -1 } else { S as i32 }, v as i32),
                        _ => (u as i32, v as i32, if sign < 0 { -1 } else { S as i32 }),
                    };
                    // Interior cell touching this shell cell (one step back
                    // along the face normal).
                    let ix = lx.clamp(0, S as i32 - 1);
                    let iy = ly.clamp(0, S as i32 - 1);
                    let iz = lz.clamp(0, S as i32 - 1);
                    let inside = ids[Self::index(ix + 1, iy + 1, iz + 1)];
                    if !prop(props, BlockId(inside)).cube {
                        ids[Self::index(lx + 1, ly + 1, lz + 1)] = UNLOADED;
                        continue;
                    }
                    let wx = origin.x + lx;
                    let wy = origin.y + ly;
                    let wz = origin.z + lz;
                    let i = Self::index(lx + 1, ly + 1, lz + 1);
                    ids[i] = if sample_loaded(wx, wy, wz) {
                        sample(wx, wy, wz).0
                    } else {
                        UNLOADED
                    };
                }
            }
        }
        Self { ids }
    }

    /// Raw ID at padding coordinates (already offset by +1). The caller must
    /// pass coordinates in `0..18`.
    #[inline]
    fn raw(&self, x: i32, y: i32, z: i32) -> u16 {
        self.ids[Self::index(x, y, z)]
    }
}

/// Compact per-block data used by the greedy pass. Looked up in a dense table
/// indexed by block ID, built once per mesh instead of touching the (large)
/// registry definition dozens of times per voxel.
#[derive(Clone, Copy)]
struct CubeProp {
    kind: BlockKind,
    opaque: bool,
    /// Drawn as a full cube by the greedy pass (not air, liquid, foliage or a
    /// per-block inset model like cactus).
    cube: bool,
    /// True for glass-style blocks that need the blended transparent pass.
    /// Other non-opaque cubes (leaves, torches) are alpha-cutout and go into
    /// the depth-writing opaque pass, where chunk.frag's `discard` handles
    /// their transparent texels.
    blend: bool,
    tiles: [u16; 6],
}

const SKIP_PROP: CubeProp = CubeProp {
    kind: BlockKind::Air,
    opaque: false,
    cube: false,
    blend: false,
    tiles: [0; 6],
};

fn mesh_props(reg: &BlockRegistry) -> Vec<CubeProp> {
    (0..reg.len())
        .map(|i| {
            let def = reg.get(BlockId(i as u16));
            let blend = def.kind == BlockKind::Transparent
                && (def.material.flags & material_flag::TRANSLUCENT_ABSORB) != 0;
            CubeProp {
                kind: def.kind,
                opaque: def.opaque,
                cube: def.is_rendered()
                    && !matches!(
                        def.kind,
                        BlockKind::Liquid | BlockKind::Foliage | BlockKind::Cactus
                    ),
                blend,
                tiles: def.textures.tiles,
            }
        })
        .collect()
}

#[inline]
fn prop(props: &[CubeProp], id: BlockId) -> CubeProp {
    props.get(id.0 as usize).copied().unwrap_or(SKIP_PROP)
}

/// Per-axis occupancy bitmasks for the whole chunk. For each face axis, slice
/// and row, `cube` has a bit set for every full-cube block and `opaque` for
/// every fully occluding block. A face only exists where the own cell is a cube
/// and the neighbouring cell is not opaque, so interior slices of solid
/// terrain reduce to a single `and-not` on a 16-bit mask instead of 256 cell
/// tests.
struct Occupancy {
    cube: [[[u32; S]; S]; 3],
    opaque: [[[u32; S]; S]; 3],
}

impl Occupancy {
    fn build(pad: &PadGrid, props: &[CubeProp]) -> Self {
        let mut occ = Self {
            cube: [[[0; S]; S]; 3],
            opaque: [[[0; S]; S]; 3],
        };
        // One pass over the chunk fills all three axis masks. Mask axes are:
        //   X faces: slice=x, row=y, bit=z
        //   Y faces: slice=y, row=z, bit=x
        //   Z faces: slice=z, row=y, bit=x
        for x in 0..S {
            for y in 0..S {
                for z in 0..S {
                    let p = prop(props, BlockId(pad.raw(x as i32 + 1, y as i32 + 1, z as i32 + 1)));
                    let bit = 1u32 << z;
                    if p.cube {
                        occ.cube[0][x][y] |= bit;
                        occ.cube[1][y][z] |= 1u32 << x;
                        occ.cube[2][z][y] |= 1u32 << x;
                    }
                    // Opacity drives neighbour culling even for blocks the
                    // greedy pass does not draw as a cube.
                    if p.opaque {
                        occ.opaque[0][x][y] |= bit;
                        occ.opaque[1][y][z] |= 1u32 << x;
                        occ.opaque[2][z][y] |= 1u32 << x;
                    }
                }
            }
        }
        occ
    }
}

/// One face's exposed-cell bitmask for a single slice. `rows[v]` has bit `u`
/// set when the block at that mask cell owns a visible face; `ids` holds the
/// block ID for each set bit. The greedy emitter clears consumed bits, which
/// removes the old per-slice visited array and its repeated clearing pass.
struct FaceMask {
    rows: [u32; S],
    ids: [[u16; S]; S],
}

impl FaceMask {
    fn build(
        &mut self,
        pad: &PadGrid,
        props: &[CubeProp],
        occ: &Occupancy,
        face: Face,
        slice: i32,
        chunk: &Chunk,
    ) {
        self.rows = [0; S];
        let axis = face as usize / 2;
        let positive = matches!(face, Face::PosX | Face::PosY | Face::PosZ);
        let neighbor_slice = slice + if positive { 1 } else { -1 };

        if (0..CHUNK_SIZE).contains(&neighbor_slice) {
            // Interior slice: every neighbour is another chunk voxel, so the
            // occupancy masks give the candidate faces directly.
            let n = face.normal();
            for v in 0..S {
                let mut bits = occ.cube[axis][slice as usize][v]
                    & !occ.opaque[axis][neighbor_slice as usize][v];
                while bits != 0 {
                    let u = bits.trailing_zeros() as usize;
                    let (lx, ly, lz) = mask_to_block(face, slice, u as i32, v as i32);
                    let id = BlockId(pad.raw(lx + 1, ly + 1, lz + 1));
                    let nid = BlockId(pad.raw(lx + n.x + 1, ly + n.y + 1, lz + n.z + 1));
                    let own = prop(props, id);
                    let nb = prop(props, nid);
                    if nid.is_air() || nb.kind != own.kind {
                        self.rows[v] |= 1u32 << u;
                        self.ids[v][u] = id.0;
                    }
                    bits &= bits - 1;
                }
            }
            return;
        }

        // Chunk-border slice: the neighbour comes from the padded shell and
        // the negative-side ownership rule applies.
        let n = face.normal();
        for v in 0..S {
            for u in 0..S {
                let (lx, ly, lz) = mask_to_block(face, slice, u as i32, v as i32);
                let raw = pad.raw(lx + 1, ly + 1, lz + 1);
                if raw == UNLOADED {
                    continue;
                }
                let id = BlockId(raw);
                let own = prop(props, id);
                if !own.cube {
                    continue;
                }
                let nraw = pad.raw(lx + n.x + 1, ly + n.y + 1, lz + n.z + 1);
                let n_loaded = nraw != UNLOADED;
                let nid = BlockId(if n_loaded { nraw } else { 0 });

                // Negative chunk borders belong to the neighbouring chunk:
                // it draws the shared face. If it is unloaded or air there is
                // no face on that side, so this chunk draws one.
                let negative_border = match face {
                    Face::NegX => lx == 0,
                    Face::NegZ => lz == 0,
                    Face::NegY => ly == 0 && chunk.pos.y() > 0,
                    _ => false,
                };
                let visible = if negative_border {
                    !n_loaded || nid.is_air()
                } else if !n_loaded {
                    // The far edge of the loaded world: treat as open air.
                    true
                } else {
                    let nb = prop(props, nid);
                    !nb.opaque && (nid.is_air() || nb.kind != own.kind)
                };
                if visible {
                    self.rows[v] |= 1u32 << u;
                    self.ids[v][u] = raw;
                }
            }
        }
    }
}

// ── Sub-chunk sections ───────────────────────────────────────────────
/// One mesh section is 8³ blocks, so a chunk is 2×2×2 sections. Section
/// indices use `x | y << 1 | z << 2`.
const SECTION_SIZE: usize = 8;
const SECTION_BLOCKS: usize = 8 * 8 * 8;

fn section_base(section: u8) -> IVec3 {
    IVec3::new(
        (section as i32 & 1) * 8,
        ((section as i32 >> 1) & 1) * 8,
        ((section as i32 >> 2) & 1) * 8,
    )
}

/// Padded 10³ copy of one 8³ section. The one-voxel shell is sampled from the
/// chunk/world so a section can resolve faces that cross into its neighbour.
struct SectionPad {
    ids: Vec<u16>,
}

impl SectionPad {
    #[inline]
    fn index(x: i32, y: i32, z: i32) -> usize {
        (((z + 1) as usize) * 10 + ((y + 1) as usize)) * 10 + ((x + 1) as usize)
    }

    fn build<Smp, Ld>(
        chunk: &Chunk,
        base: IVec3,
        sample: &Smp,
        sample_loaded: &Ld,
    ) -> Self
    where
        Smp: Fn(i32, i32, i32) -> BlockId,
        Ld: Fn(i32, i32, i32) -> bool,
    {
        let mut ids = vec![UNLOADED; 10 * 10 * 10];
        for y in 0..SECTION_SIZE as i32 {
            for z in 0..SECTION_SIZE as i32 {
                for x in 0..SECTION_SIZE as i32 {
                    ids[Self::index(x, y, z)] = chunk.get(base.x + x, base.y + y, base.z + z).0;
                }
            }
        }
        let origin = chunk_origin(chunk.pos);
        for (axis, sign) in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
            for v in 0..SECTION_SIZE as i32 {
                for u in 0..SECTION_SIZE as i32 {
                    let (x, y, z) = match axis {
                        0 => (if sign < 0 { -1 } else { 8 }, v, u),
                        1 => (u, if sign < 0 { -1 } else { 8 }, v),
                        _ => (u, v, if sign < 0 { -1 } else { 8 }),
                    };
                    let wx = origin.x + base.x + x;
                    let wy = origin.y + base.y + y;
                    let wz = origin.z + base.z + z;
                    ids[Self::index(x, y, z)] = if sample_loaded(wx, wy, wz) {
                        sample(wx, wy, wz).0
                    } else {
                        UNLOADED
                    };
                }
            }
        }
        Self { ids }
    }

    #[inline]
    fn raw(&self, x: i32, y: i32, z: i32) -> u16 {
        self.ids[Self::index(x, y, z)]
    }
}

struct SectionOccupancy {
    cube: [[[u8; SECTION_SIZE]; SECTION_SIZE]; 3],
    opaque: [[[u8; SECTION_SIZE]; SECTION_SIZE]; 3],
}

impl SectionOccupancy {
    fn build(pad: &SectionPad, props: &[CubeProp]) -> Self {
        let mut occ = Self {
            cube: [[[0; SECTION_SIZE]; SECTION_SIZE]; 3],
            opaque: [[[0; SECTION_SIZE]; SECTION_SIZE]; 3],
        };
        for x in 0..SECTION_SIZE {
            for y in 0..SECTION_SIZE {
                for z in 0..SECTION_SIZE {
                    let p = prop(
                        props,
                        BlockId(pad.raw(x as i32, y as i32, z as i32)),
                    );
                    let bit = 1u8 << z;
                    if p.cube {
                        occ.cube[0][x][y] |= bit;
                        occ.cube[1][y][z] |= 1u8 << x;
                        occ.cube[2][z][y] |= 1u8 << x;
                    }
                    if p.opaque {
                        occ.opaque[0][x][y] |= bit;
                        occ.opaque[1][y][z] |= 1u8 << x;
                        occ.opaque[2][z][y] |= 1u8 << x;
                    }
                }
            }
        }
        occ
    }
}

struct SectionFaceMask {
    rows: [u8; SECTION_SIZE],
    ids: [[u16; SECTION_SIZE]; SECTION_SIZE],
}

impl SectionFaceMask {
    fn build(
        &mut self,
        pad: &SectionPad,
        props: &[CubeProp],
        _occ: &SectionOccupancy,
        face: Face,
        slice: i32,
        base: IVec3,
        chunk: &Chunk,
    ) {
        self.rows = [0; SECTION_SIZE];
        let n = face.normal();
        for v in 0..SECTION_SIZE as i32 {
            for u in 0..SECTION_SIZE as i32 {
                let (x, y, z) = mask_to_block(face, slice, u, v);
                let raw = pad.raw(x, y, z);
                if raw == UNLOADED {
                    continue;
                }
                let own = prop(props, BlockId(raw));
                if !own.cube {
                    continue;
                }
                let nraw = pad.raw(x + n.x, y + n.y, z + n.z);
                let n_loaded = nraw != UNLOADED;
                let nid = BlockId(if n_loaded { nraw } else { 0 });
                let gx = base.x + x;
                let gy = base.y + y;
                let gz = base.z + z;
                // Only real chunk borders use the negative-side ownership
                // rule. Internal section boundaries are normal faces: each
                // side evaluates its own block against the other.
                let negative_border = match face {
                    Face::NegX => gx == 0,
                    Face::NegZ => gz == 0,
                    Face::NegY => gy == 0 && chunk.pos.y() > 0,
                    _ => false,
                };
                let visible = if negative_border {
                    !n_loaded || nid.is_air()
                } else if !n_loaded {
                    true
                } else {
                    let nb = prop(props, nid);
                    !nb.opaque && (nid.is_air() || nb.kind != own.kind)
                };
                if visible {
                    self.rows[v as usize] |= 1u8 << u;
                    self.ids[v as usize][u as usize] = raw;
                }
            }
        }
    }
}

fn greedy_emit_section(
    bundle: &mut ChunkMeshBundle,
    mask: &mut SectionFaceMask,
    face: Face,
    slice: i32,
    base: IVec3,
    props: &[CubeProp],
) {
    for v in 0..SECTION_SIZE {
        let mut row = mask.rows[v];
        while row != 0 {
            let u = row.trailing_zeros() as usize;
            let id = mask.ids[v][u];
            let mut w = 1usize;
            while u + w < SECTION_SIZE {
                let bit = 1u8 << (u + w);
                if row & bit == 0 || mask.ids[v][u + w] != id {
                    break;
                }
                w += 1;
            }
            let col = (((1u16 << w) - 1) as u8) << u;
            let mut h = 1usize;
            'grow: while v + h < SECTION_SIZE {
                if mask.rows[v + h] & col != col {
                    break;
                }
                for i in u..u + w {
                    if mask.ids[v + h][i] != id {
                        break 'grow;
                    }
                }
                h += 1;
            }
            for r in v..v + h {
                mask.rows[r] &= !col;
            }

            let p = prop(props, BlockId(id));
            let target = if p.blend {
                &mut bundle.transparent
            } else {
                &mut bundle.opaque
            };
            let vertex_base = target.vertices.len() as u32;
            let local = merged_positions(face, u as i32, v as i32, w as i32, h as i32, slice);
            let positions = local.map(|p| [p[0] + base.x as f32, p[1] + base.y as f32, p[2] + base.z as f32]);
            let uvs = merged_uvs(face, w as u32, h as u32);
            let tile = p.tiles[face as usize] as u32;
            for c in 0..4 {
                target
                    .vertices
                    .push(ChunkVertex::new(positions[c], uvs[c], tile, 0));
            }
            target.indices.extend_from_slice(&[
                vertex_base,
                vertex_base + 1,
                vertex_base + 2,
                vertex_base,
                vertex_base + 2,
                vertex_base + 3,
            ]);
            row = mask.rows[v];
        }
    }
}

pub struct ChunkMesher;

impl ChunkMesher {
    /// Build one 8³ section at chunk-local `base`. The result is chunk-local
    /// (positions still in 0..16), so eight section meshes can be concatenated
    /// into the normal chunk bundle.
    pub fn build_section(
        &self,
        section: u8,
        chunk: &Chunk,
        reg: &BlockRegistry,
        sample: impl Fn(i32, i32, i32) -> BlockId,
        sample_water: impl Fn(i32, i32, i32) -> WaterVertexInfo,
        sample_loaded: impl Fn(i32, i32, i32) -> bool,
    ) -> ChunkMeshBundle {
        let mut bundle = ChunkMeshBundle::default();
        bundle.opaque.vertices.reserve(SECTION_BLOCKS / 4);
        bundle.transparent.vertices.reserve(SECTION_BLOCKS / 8);
        let base = section_base(section);
        let origin = chunk_origin(chunk.pos);
        let props = mesh_props(reg);
        let pad = SectionPad::build(chunk, base, &sample, &sample_loaded);
        let build = MeshBuild {
            chunk,
            reg,
            sample,
            sample_water,
            sample_loaded,
            origin,
        };

        for ly in 0..SECTION_SIZE as i32 {
            for lz in 0..SECTION_SIZE as i32 {
                for lx in 0..SECTION_SIZE as i32 {
                    let id = BlockId(pad.raw(lx, ly, lz));
                    if id.is_air() {
                        continue;
                    }
                    let def = reg.get(id);
                    if !def.is_rendered() {
                        continue;
                    }
                    let pos = base + IVec3::new(lx, ly, lz);
                    match def.kind {
                        BlockKind::Foliage => {
                            emit_foliage_cross(&mut bundle, chunk, pos.x, pos.y, pos.z, def);
                        }
                        BlockKind::Liquid => {
                            let wx = origin.x + pos.x;
                            let wy = origin.y + pos.y;
                            let wz = origin.z + pos.z;
                            let info = (build.sample_water)(wx, wy, wz);
                            if info.fill == 0 {
                                continue;
                            }
                            // Height-aware interior test. A side neighbour only
                            // hides this face when its surface is at least as
                            // high; a lower adjacent column still exposes a
                            // vertical water wall above it.
                            let own_height = info.height();
                            let covered_side = |dx: i32, dz: i32| {
                                let nid = (build.sample)(wx + dx, wy, wz + dz);
                                let nd = reg.get(nid);
                                nd.opaque
                                    || (nd.kind == BlockKind::Liquid
                                        && (build.sample_water)(wx + dx, wy, wz + dz)
                                            .height()
                                            >= own_height)
                            };
                            let covered_top = {
                                let nid = (build.sample)(wx, wy + 1, wz);
                                let nd = reg.get(nid);
                                nd.opaque || nd.kind == BlockKind::Liquid
                            };
                            if covered_top
                                && covered_side(0, 1)
                                && covered_side(0, -1)
                                && covered_side(1, 0)
                                && covered_side(-1, 0)
                            {
                                continue;
                            }
                            build.emit_water(&mut bundle, pos, def, info);
                        }
                        _ if def.kind == BlockKind::Cactus => {
                            build.emit_cactus(&mut bundle, pos, def);
                        }
                        _ => {}
                    }
                }
            }
        }

        let occ = SectionOccupancy::build(&pad, &props);
        let mut mask = SectionFaceMask {
            rows: [0; SECTION_SIZE],
            ids: [[0; SECTION_SIZE]; SECTION_SIZE],
        };
        for face in Face::ALL {
            for slice in 0..SECTION_SIZE as i32 {
                mask.build(&pad, &props, &occ, face, slice, base, chunk);
                greedy_emit_section(&mut bundle, &mut mask, face, slice, base, &props);
            }
        }
        bundle
    }
}


impl ChunkMesher {
    pub fn build(
        &self,
        chunk: &Chunk,
        reg: &BlockRegistry,
        sample: impl Fn(i32, i32, i32) -> BlockId,
        sample_water: impl Fn(i32, i32, i32) -> WaterVertexInfo,
        sample_loaded: impl Fn(i32, i32, i32) -> bool,
    ) -> ChunkMeshBundle {
        let mut bundle = ChunkMeshBundle::default();
        let nac = chunk.non_air_count();
        if nac == 0 {
            return bundle;
        }
        // Greedy quads cover many cells, so the old `nac * 4` reservation
        // overshoots badly. These still avoid most reallocations.
        bundle.opaque.vertices.reserve(nac * 2);
        bundle.opaque.indices.reserve(nac * 3);
        bundle.transparent.vertices.reserve(nac);
        bundle.transparent.indices.reserve(nac * 2);

        let origin = chunk_origin(chunk.pos);
        let props = mesh_props(reg);
        let pad = PadGrid::build(chunk, &props, &sample, &sample_loaded);
        let build = MeshBuild {
            chunk,
            reg,
            sample,
            sample_water,
            sample_loaded,
            origin,
        };

        // Special geometry (water surfaces, foliage crosses, inset cacti) is
        // emitted per voxel; the greedy pass below handles full cubes only.
        for ly in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let id = BlockId(pad.raw(lx + 1, ly + 1, lz + 1));
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
                            // Specialized water fast path: a liquid voxel whose
                            // top and four horizontal neighbours are all
                            // liquid/opaque cannot own a visible face. Deep
                            // ocean/interior water is skipped before any wave
                            // sampling or face work.
                            let wx = origin.x + lx;
                            let wy = origin.y + ly;
                            let wz = origin.z + lz;
                            let info = (build.sample_water)(wx, wy, wz);
                            if info.fill == 0 {
                                continue;
                            }
                            // Height-aware interior test. A side neighbour only
                            // hides this face when its surface is at least as
                            // high; a lower adjacent column still exposes a
                            // vertical water wall above it.
                            let own_height = info.height();
                            let covered_side = |dx: i32, dz: i32| {
                                let nid = (build.sample)(wx + dx, wy, wz + dz);
                                let nd = reg.get(nid);
                                nd.opaque
                                    || (nd.kind == BlockKind::Liquid
                                        && (build.sample_water)(wx + dx, wy, wz + dz)
                                            .height()
                                            >= own_height)
                            };
                            let covered_top = {
                                let nid = (build.sample)(wx, wy + 1, wz);
                                let nd = reg.get(nid);
                                nd.opaque || nd.kind == BlockKind::Liquid
                            };
                            if covered_top
                                && covered_side(0, 1)
                                && covered_side(0, -1)
                                && covered_side(1, 0)
                                && covered_side(-1, 0)
                            {
                                continue;
                            }
                            build.emit_water(&mut bundle, IVec3::new(lx, ly, lz), def, info);
                        }
                        _ if def.kind == BlockKind::Cactus => {
                            build.emit_cactus(&mut bundle, IVec3::new(lx, ly, lz), def);
                        }
                        _ => {}
                    }
                }
            }
        }

        let occ = Occupancy::build(&pad, &props);
        let mut mask = FaceMask {
            rows: [0; S],
            ids: [[0; S]; S],
        };
        for face in Face::ALL {
            for slice in 0..CHUNK_SIZE {
                mask.build(&pad, &props, &occ, face, slice, chunk);
                greedy_emit(&mut bundle, &mut mask, face, slice, &props);
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

/// Shared inputs for one chunk's mesh build. Bundling them keeps the
/// per-face helpers to a handful of arguments while avoiding per-voxel
/// dynamic dispatch on the sampler closures.
struct MeshBuild<'a, S, W, L>
where
    S: Fn(i32, i32, i32) -> BlockId,
    W: Fn(i32, i32, i32) -> WaterVertexInfo,
    L: Fn(i32, i32, i32) -> bool,
{
    chunk: &'a Chunk,
    reg: &'a BlockRegistry,
    sample: S,
    sample_water: W,
    sample_loaded: L,
    origin: IVec3,
}

/// True for negative-side faces on chunk borders whose neighbour was checked
/// during face ownership resolution. The caller can skip sampling it again.
fn is_neg_border(face: Face, lx: i32, ly: i32, lz: i32, chunk: &Chunk) -> bool {
    (face == Face::NegX && lx == 0)
        || (face == Face::NegZ && lz == 0)
        || (face == Face::NegY && ly == 0 && chunk.pos.y() > 0)
}

impl<S, W, L> MeshBuild<'_, S, W, L>
where
    S: Fn(i32, i32, i32) -> BlockId,
    W: Fn(i32, i32, i32) -> WaterVertexInfo,
    L: Fn(i32, i32, i32) -> bool,
{
    /// Border ownership test. `check_borders` is false for blocks that ignore
    /// neighbouring chunks (cacti), matching their historical behaviour.
    fn should_emit_face(&self, face: Face, pos: IVec3, check_borders: bool) -> bool {
        let MeshBuild {
            chunk,
            sample,
            sample_loaded,
            origin,
            ..
        } = self;
        let IVec3 { x: lx, y: ly, z: lz } = pos;
        if !check_borders {
            return true;
        }
        let wx = origin.x + lx;
        let wy = origin.y + ly;
        let wz = origin.z + lz;
        let n = face.normal();
        // Chunk-border face ownership: the negative-side chunk owns each shared
        // border face. So on our negative faces we suppress only when the neighbour
        // is loaded AND non-air (it will draw the face); if the neighbour is air we
        // must draw it ourselves. Suppressing on opacity instead would let both
        // chunks emit the same face and z-fight.
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
            Face::NegY if ly == 0 && chunk.pos.y() > 0 && sample_loaded(wx, wy - 1, wz) => {
                let nb = sample(wx + n.x, wy + n.y, wz + n.z);
                if !nb.is_air() {
                    return false;
                }
            }
            _ => {}
        }
    true
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
    mask: &mut FaceMask,
    face: Face,
    slice: i32,
    props: &[CubeProp],
) {
    for v in 0..S {
        let mut row = mask.rows[v];
        while row != 0 {
            let u = row.trailing_zeros() as usize;
            let id = mask.ids[v][u];

            // Grow along +u while the run stays set and keeps the same block.
            let mut w = 1usize;
            while u + w < S {
                let bit = 1u32 << (u + w);
                if row & bit == 0 || mask.ids[v][u + w] != id {
                    break;
                }
                w += 1;
            }

            // Grow along +v while every row covers the run and matches the ID.
            let col = ((1u32 << w) - 1) << u;
            let mut h = 1usize;
            'grow: while v + h < S {
                if mask.rows[v + h] & col != col {
                    break;
                }
                for i in u..u + w {
                    if mask.ids[v + h][i] != id {
                        break 'grow;
                    }
                }
                h += 1;
            }

            for r in v..v + h {
                mask.rows[r] &= !col;
            }

            let p = prop(props, BlockId(id));
            // Only genuinely blended materials (glass) use the transparent
            // pass. Alpha-tested leaves/torches are opaque-pass geometry with
            // depth writes, which fixes their sort order and overdraw.
            let target = if p.blend {
                &mut bundle.transparent
            } else {
                &mut bundle.opaque
            };
            let vertex_base = target.vertices.len() as u32;
            let positions = merged_positions(face, u as i32, v as i32, w as i32, h as i32, slice);
            let uvs = merged_uvs(face, w as u32, h as u32);
            let tile = p.tiles[face as usize] as u32;
            for c in 0..4 {
                target
                    .vertices
                    .push(ChunkVertex::new(positions[c], uvs[c], tile, 0));
            }
            target.indices.extend_from_slice(&[
                vertex_base,
                vertex_base + 1,
                vertex_base + 2,
                vertex_base,
                vertex_base + 2,
                vertex_base + 3,
            ]);

            row = mask.rows[v];
        }
    }
}

fn emit_quad_indices(target: &mut ChunkMesh, base: u32) {
    target
        .indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn emit_foliage_cross(
    bundle: &mut ChunkMeshBundle,
    _chunk: &Chunk,
    lx: i32,
    ly: i32,
    lz: i32,
    def: &BlockDef,
) {
    let tile = def.textures.tile(Face::PosY);
    let bx = lx as f32;
    let by = ly as f32;
    let bz = lz as f32;
    let w = FOLIAGE_PLANE_WIDTH;
    let cx = bx + 0.5;
    let cz = bz + 0.5;
    // Two perpendicular upright planes forming a `+` from above (not an X):
    //   Plane 1: XY plane at z = cz (outward normal +Z).
    //   Plane 2: YZ plane at x = cx (outward normal -X).
    let positions: [[f32; 3]; 8] = [
        [cx - w, by, cz],
        [cx + w, by, cz],
        [cx + w, by + 1.0, cz],
        [cx - w, by + 1.0, cz],
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
    // Foliage is alpha-cutout, not blended: render it in the depth-writing
    // opaque pass. Both windings are emitted so the cross remains visible
    // from either side with the opaque pipeline's back-face culling.
    let tgt = &mut bundle.opaque;
    let base = tgt.vertices.len() as u32;
    for i in 0..8 {
        tgt.vertices
            .push(ChunkVertex::new(positions[i], uvs[i], tile as u32, 0));
    }
    tgt.indices.extend_from_slice(&[
        base,
        base + 1,
        base + 2,
        base,
        base + 2,
        base + 3,
        base,
        base + 2,
        base + 1,
        base,
        base + 3,
        base + 2,
        base + 4,
        base + 5,
        base + 6,
        base + 4,
        base + 6,
        base + 7,
        base + 4,
        base + 6,
        base + 5,
        base + 4,
        base + 7,
        base + 6,
    ]);
}

impl<S, W, L> MeshBuild<'_, S, W, L>
where
    S: Fn(i32, i32, i32) -> BlockId,
    W: Fn(i32, i32, i32) -> WaterVertexInfo,
    L: Fn(i32, i32, i32) -> bool,
{
    fn emit_water(
        &self,
        bundle: &mut ChunkMeshBundle,
        pos: IVec3,
        def: &BlockDef,
        info: WaterVertexInfo,
    ) {
        let MeshBuild {
            chunk,
            reg,
            sample,
            sample_water,
            sample_loaded,
            origin,
            ..
        } = self;
        let IVec3 { x: lx, y: ly, z: lz } = pos;
        let height_frac = info.height().clamp(0.0, 1.0);
        let wx = origin.x + lx;
        let wy = origin.y + ly;
        let wz = origin.z + lz;
        let tile = def.textures.tile(Face::PosY);
        let animated = info.kind != 0;

        for (fi, face) in Face::ALL.iter().enumerate() {
            if !self.should_emit_face(*face, pos, true) {
                continue;
            }
            let n = face.normal();
            // A border face resolved by should_emit_face already knows its
            // neighbour is air or unloaded; avoid sampling it again.
            let resolved_border = is_neg_border(*face, lx, ly, lz, chunk);
            // An unloaded chunk border is not an exposed water wall. Emitting
            // a side quad there creates a translucent vertical seam inside the
            // ocean (the neighbouring chunk will draw its own top surface).
            // Loaded air borders still own a real shoreline step.
            let border_side = *face != Face::PosY && *face != Face::NegY;
            if border_side && !sample_loaded(wx + n.x, wy + n.y, wz + n.z) {
                continue;
            }
            let neighbour = if resolved_border {
                BlockId::AIR
            } else {
                sample(wx + n.x, wy + n.y, wz + n.z)
            };
            let neighbour_def = reg.get(neighbour);
            if neighbour_def.opaque {
                continue;
            }
            let neighbour_info = if !neighbour.is_air() && neighbour_def.kind == BlockKind::Liquid {
                sample_water(wx + n.x, wy + n.y, wz + n.z)
            } else {
                WaterVertexInfo::default()
            };
            let water_height = info.height();
            let neighbour_height = neighbour_info.height();
            if neighbour_info.fill > 0 {
                match face {
                    Face::PosX | Face::NegX | Face::PosZ | Face::NegZ => {
                        // Only the taller cell owns the exposed step between
                        // adjacent liquid surfaces. This avoids overlapping
                        // transparent quads and z-fighting at different levels.
                        if water_height <= neighbour_height {
                            continue;
                        }
                    }
                    Face::PosY if water_height >= 1.0 => continue,
                    Face::NegY if neighbour_height >= 1.0 => continue,
                    _ => {}
                }
            }
            if *face == Face::NegY && neighbour_def.kind == BlockKind::Solid {
                continue;
            }
            if *face != Face::PosY && *face != Face::NegY && neighbour_def.kind == BlockKind::Solid {
                continue;
            }

            let base = FACE_BASE[fi];
            let uvs = FACE_UVS[fi];
            let mut p_base = GVec3::new(lx as f32, ly as f32, lz as f32) + base;
            let side_neighbour_height = if *face == Face::PosX
                || *face == Face::NegX
                || *face == Face::PosZ
                || *face == Face::NegZ
            {
                neighbour_height
            } else {
                0.0
            };
            let side_height = (water_height - side_neighbour_height).max(0.0);
            if *face == Face::PosY {
                p_base.y = ly as f32 + height_frac;
            }

            // Wave amplitude is sampled per **surface corner** rather than
            // per column. Adjacent surface quads then agree exactly on their
            // shared edge, which keeps the displaced ocean surface watertight
            // across a depth gradient. The minimum over the four columns
            // touching the corner also fades swell out at the shoreline.
            let corner_amplitude = |cx: i32, cz: i32| -> u8 {
                if !animated {
                    return 0;
                }
                let mut amp = u8::MAX;
                let mut found = false;
                for dx in -1..=0 {
                    for dz in -1..=0 {
                        let ni = sample_water(wx + cx + dx, wy, wz + cz + dz);
                        if ni.kind != 0 {
                            amp = amp.min(ni.amplitude);
                            found = true;
                        }
                    }
                }
                if found {
                    amp
                } else {
                    info.amplitude
                }
            };

            // The wave direction/rapids are packed per column. At a corner
            // shared by two surface quads, taking those fields from each
            // block's own column makes the two vertices land on different
            // points of the same wave, opening hairline gaps along every
            // block edge. Pick one deterministic water column from the four
            // touching the corner so both quads evaluate the exact same wave.
            let corner_info = |cx: i32, cz: i32| -> WaterVertexInfo {
                for dx in -1..=0 {
                    for dz in -1..=0 {
                        let ni = sample_water(wx + cx + dx, wy, wz + cz + dz);
                        if ni.kind != 0 {
                            return ni;
                        }
                    }
                }
                info
            };

            let corners = FACE_CORNERS[fi];
            let start = bundle.transparent.vertices.len() as u32;
            for (c, corner) in corners.iter().enumerate() {
                let mut cp = p_base + *corner;
                if *face != Face::PosY && *face != Face::NegY {
                    cp.y = ly as f32 + side_neighbour_height + corner.y * side_height;
                }
                let surface_vertex = if *face == Face::PosY {
                    info.surface
                } else if *face == Face::NegY {
                    false
                } else {
                    info.surface && corner.y > 0.5
                };
                let cx = corner.x.round() as i32;
                let cz = corner.z.round() as i32;
                let vertex_amplitude = if surface_vertex {
                    corner_amplitude(cx, cz)
                } else {
                    0
                };
                let water = if animated {
                    let base_info = if surface_vertex { corner_info(cx, cz) } else { info };
                    WaterVertexInfo {
                        surface: surface_vertex,
                        amplitude: vertex_amplitude,
                        ..base_info
                    }
                    .pack()
                } else {
                    0
                };
                bundle.transparent.vertices.push(ChunkVertex::new(
                    [cp.x, cp.y, cp.z],
                    [uvs[c].x, uvs[c].y],
                    tile as u32,
                    water,
                ));
            }
            emit_quad_indices(&mut bundle.transparent, start);
        }
    }
}

impl<S, W, L> MeshBuild<'_, S, W, L>
where
    S: Fn(i32, i32, i32) -> BlockId,
    W: Fn(i32, i32, i32) -> WaterVertexInfo,
    L: Fn(i32, i32, i32) -> bool,
{
    fn emit_cactus(&self, bundle: &mut ChunkMeshBundle, pos: IVec3, def: &BlockDef) {
        let MeshBuild {
            reg,
            sample,
            origin,
            ..
        } = self;
        let IVec3 { x: lx, y: ly, z: lz } = pos;
        let wx = origin.x + lx;
        let wy = origin.y + ly;
        let wz = origin.z + lz;
        for (fi, face) in Face::ALL.iter().enumerate() {
            if !self.should_emit_face(*face, pos, false) {
                continue;
            }
            let n = face.normal();
            let neighbour = sample(wx + n.x, wy + n.y, wz + n.z);
            if reg.get(neighbour).opaque {
                continue;
            }

            let base = FACE_BASE[fi];

            let tile = def.textures.tile(*face);
            let corners = FACE_CORNERS[fi];

            // Cactus renders as a clean box inset 2px (of 16) on all four
            // horizontal sides, like Minecraft. Each side face is BOTH pushed in
            // along its normal AND narrowed tangentially, so its edges meet the
            // inset top/bottom faces exactly — no corner overhang, and the top
            // lines up with the sides. UVs are derived from the corner's local
            // position (V up the +Y axis) so the bark texture stays upright on
            // every face instead of inheriting the greedy-mesh table's per-face
            // rotation.
            // Pull a 0..1 tangential coordinate toward the block centre by
            // CACTUS_INSET.
            let narrow = |t: f32| 0.5 + (t - 0.5) * (1.0 - 2.0 * CACTUS_INSET);

            let start = bundle.opaque.vertices.len() as u32;
            for corner in corners {
                // Local 0..1 corner within the block (normal axis at its face value).
                let f = base + corner;
                // (fx,fy,fz) = inset local position; (u,v) = upright tile UV.
                let (fx, fy, fz, u, v) = match face {
                    Face::PosX => (1.0 - CACTUS_INSET, f.y, narrow(f.z), f.z, 1.0 - f.y),
                    Face::NegX => (CACTUS_INSET, f.y, narrow(f.z), 1.0 - f.z, 1.0 - f.y),
                    Face::PosZ => (narrow(f.x), f.y, 1.0 - CACTUS_INSET, f.x, 1.0 - f.y),
                    Face::NegZ => (narrow(f.x), f.y, CACTUS_INSET, f.x, 1.0 - f.y),
                    Face::PosY => (narrow(f.x), f.y, narrow(f.z), f.x, f.z),
                    Face::NegY => (narrow(f.x), f.y, narrow(f.z), f.x, 1.0 - f.z),
                };
                let cp = GVec3::new(lx as f32 + fx, ly as f32 + fy, lz as f32 + fz);
                bundle.opaque.vertices.push(ChunkVertex::new(
                    [cp.x, cp.y, cp.z],
                    [u, v],
                    tile as u32,
                    0,
                ));
            }
            emit_quad_indices(&mut bundle.opaque, start);
        }
    }
}

impl ChunkMesher {
    /// Build a 2x-downsampled opaque LOD mesh (8³ cells covering the 16³
    /// chunk). Water, foliage and cacti are deliberately omitted: those stay on
    /// the full-detail mesh and the cull shader falls back to LOD0 for the
    /// transparent slot.
    ///
    /// A 4-block skirt is generated around the chunk perimeter so a coarse
    /// chunk next to a full-detail neighbour has no visible crack when the
    /// sampled surface heights differ.
    pub fn build_lod1(
        &self,
        chunk: &Chunk,
        reg: &BlockRegistry,
        sample: impl Fn(i32, i32, i32) -> BlockId,
        sample_water: impl Fn(i32, i32, i32) -> WaterVertexInfo,
        sample_loaded: impl Fn(i32, i32, i32) -> bool,
    ) -> ChunkMeshBundle {
        const G: usize = 8;
        const CELL: i32 = 2;
        let mut mesh = ChunkMesh::default();
        let props = mesh_props(reg);
        let origin = chunk_origin(chunk.pos);
        let mut ids = [0u16; G * G * G];
        for cy in 0..G {
            for cz in 0..G {
                for cx in 0..G {
                    // Prefer the cell centre, then any opaque cell, then any
                    // non-air cell. This keeps terrain silhouettes stable
                    // without a full 2³ majority vote per cell.
                    let mut picked = BlockId::AIR;
                    let mut saw_air = false;
                    for dy in 0..2 {
                        for dz in 0..2 {
                            for dx in 0..2 {
                                let id = chunk.get(
                                    (cx * 2 + dx) as i32,
                                    (cy * 2 + dy) as i32,
                                    (cz * 2 + dz) as i32,
                                );
                                if id.is_air() {
                                    saw_air = true;
                                } else if reg.get(id).opaque {
                                    picked = id;
                                    break;
                                } else if picked.is_air() {
                                    picked = id;
                                }
                            }
                        }
                    }
                    if picked.is_air() && saw_air {
                        picked = BlockId::AIR;
                    }
                    ids[(cy * G + cz) * G + cx] = picked.0;
                }
            }
        }

        // 10³ padded cell grid. Shell cells sample the centre of the adjacent
        // 2-block cell in the neighbouring chunk.
        let mut pad = [UNLOADED; 10 * 10 * 10];
        let pidx = |x: i32, y: i32, z: i32| -> usize {
            (((z + 1) as usize) * 10 + ((y + 1) as usize)) * 10 + ((x + 1) as usize)
        };
        for y in 0..G as i32 {
            for z in 0..G as i32 {
                for x in 0..G as i32 {
                    pad[pidx(x, y, z)] = ids[(y as usize * G + z as usize) * G + x as usize];
                }
            }
        }
        for (axis, sign) in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
            for v in 0..G as i32 {
                for u in 0..G as i32 {
                    let (x, y, z) = match axis {
                        0 => (if sign < 0 { -1 } else { G as i32 }, v, u),
                        1 => (u, if sign < 0 { -1 } else { G as i32 }, v),
                        _ => (u, v, if sign < 0 { -1 } else { G as i32 }),
                    };
                    let wx = origin.x + x * CELL + CELL / 2;
                    let wy = origin.y + y * CELL + CELL / 2;
                    let wz = origin.z + z * CELL + CELL / 2;
                    pad[pidx(x, y, z)] = if sample_loaded(wx, wy, wz) {
                        sample(wx, wy, wz).0
                    } else {
                        UNLOADED
                    };
                }
            }
        }

        let cell_id = |x: i32, y: i32, z: i32| -> u16 { pad[pidx(x, y, z)] };
        for face in Face::ALL {
            let n = face.normal();
            let neg_border = matches!(face, Face::NegX | Face::NegY | Face::NegZ);
            for slice in 0..G as i32 {
                let mut rows = [0u32; G];
                let mut mids = [[0u16; G]; G];
                for v in 0..G as i32 {
                    for u in 0..G as i32 {
                        let (x, y, z) = match face {
                            Face::NegX | Face::PosX => (slice, v, u),
                            Face::NegY | Face::PosY => (u, slice, v),
                            Face::NegZ | Face::PosZ => (u, v, slice),
                        };
                        let raw = cell_id(x, y, z);
                        if raw == UNLOADED {
                            continue;
                        }
                        let id = BlockId(raw);
                        let own = prop(&props, id);
                        // Coarse mesh is opaque/cutout only; blended glass and
                        // liquids stay on the full-detail transparent mesh.
                        if !own.cube || own.blend {
                            continue;
                        }
                        let nraw = cell_id(x + n.x, y + n.y, z + n.z);
                        let visible = if nraw == UNLOADED {
                            true
                        } else if neg_border && !BlockId(nraw).is_air() {
                            false
                        } else {
                            let nb = prop(&props, BlockId(nraw));
                            !nb.opaque && (BlockId(nraw).is_air() || nb.kind != own.kind)
                        };
                        if visible {
                            rows[v as usize] |= 1u32 << u;
                            mids[v as usize][u as usize] = raw;
                        }
                    }
                }
                // Greedy merge the 8x8 mask.
                for v in 0..G {
                    let mut row = rows[v];
                    while row != 0 {
                        let u = row.trailing_zeros() as usize;
                        let id = mids[v][u];
                        let mut w = 1usize;
                        while u + w < G {
                            let bit = 1u32 << (u + w);
                            if row & bit == 0 || mids[v][u + w] != id {
                                break;
                            }
                            w += 1;
                        }
                        let col = ((1u32 << w) - 1) << u;
                        let mut h = 1usize;
                        'grow: while v + h < G {
                            if rows[v + h] & col != col {
                                break;
                            }
                            for i in u..u + w {
                                if mids[v + h][i] != id {
                                    break 'grow;
                                }
                            }
                            h += 1;
                        }
                        for r in v..v + h {
                            rows[r] &= !col;
                        }
                        // Build in 8³ cell coordinates, then scale to blocks:
                        // `merged_positions` already applies the +1 plane offset
                        // around a cell, so scaling the whole result by CELL is
                        // what makes a coarse cell exactly 2 blocks wide.
                        let cell_pos = merged_positions(
                            face,
                            u as i32,
                            v as i32,
                            w as i32,
                            h as i32,
                            slice,
                        );
                        let mut positions = [[0.0f32; 3]; 4];
                        for c in 0..4 {
                            for k in 0..3 {
                                positions[c][k] = cell_pos[c][k] * CELL as f32;
                            }
                        }
                        let uvs = merged_uvs(face, (w * 2) as u32, (h * 2) as u32);
                        let tile = prop(&props, BlockId(id)).tiles[face as usize] as u32;
                        let base = mesh.vertices.len() as u32;
                        for c in 0..4 {
                            mesh.vertices.push(ChunkVertex::new(
                                positions[c],
                                uvs[c],
                                tile,
                                0,
                            ));
                        }
                        emit_quad_indices(&mut mesh, base);
                        row = rows[v];
                    }
                }
            }
        }

        // Perimeter skirts: cover height discontinuities between this coarse
        // mesh and a full-detail neighbour on either side.
        let mut top = [[0i32; G]; G];
        for z in 0..G {
            for x in 0..G {
                for y in (0..G).rev() {
                    let id = BlockId(ids[(y * G + z) * G + x]);
                    let p = prop(&props, id);
                    if !id.is_air() && p.cube && !p.blend {
                        top[z][x] = y as i32 + 1;
                        break;
                    }
                }
            }
        }
        for z in 0..G {
            for x in 0..G {
                let hy = top[z][x];
                if hy == 0 {
                    continue;
                }
                let id = BlockId(ids[((hy - 1) as usize * G + z) * G + x]);
                let y0 = ((hy - 1) * CELL).max(0) as f32;
                let y1 = ((hy + 1) * CELL).min(16) as f32;
                let x0 = x as f32 * 2.0;
                let x1 = x0 + 2.0;
                let z0 = z as f32 * 2.0;
                let z1 = z0 + 2.0;
                // Only the four perimeter rings get skirts.
                if x == 0 {
                    let v = [[0.0, y0, z0], [0.0, y0, z1], [0.0, y1, z1], [0.0, y1, z0]];
                    add_skirt(&mut mesh, v, id, &props, Face::NegX);
                }
                if x + 1 == G {
                    let v = [[16.0, y0, z1], [16.0, y0, z0], [16.0, y1, z0], [16.0, y1, z1]];
                    add_skirt(&mut mesh, v, id, &props, Face::PosX);
                }
                if z == 0 {
                    let v = [[x1, y0, 0.0], [x0, y0, 0.0], [x0, y1, 0.0], [x1, y1, 0.0]];
                    add_skirt(&mut mesh, v, id, &props, Face::NegZ);
                }
                if z + 1 == G {
                    let v = [[x0, y0, 16.0], [x1, y0, 16.0], [x1, y1, 16.0], [x0, y1, 16.0]];
                    add_skirt(&mut mesh, v, id, &props, Face::PosZ);
                }
            }
        }
        // Coarse water surface. LOD1 used to be opaque-only, which made every
        // lake and ocean vanish as soon as a chunk crossed the LOD cutoff.
        // Keep one top plane per 2x2 cell; it is enough at distance and it
        // follows the same water-surface metadata as the full-detail mesh.
        let mut transparent = ChunkMesh::default();
        let water_id = reg.id_of("water");
        if let Some(water_id) = water_id {
            let water_tile = reg.get(water_id).textures.tile(Face::PosY) as u32;
            for cz in 0..G as i32 {
                for cx in 0..G as i32 {
                    let mut top: Option<(i32, WaterVertexInfo)> = None;
                    for dz in 0..2 {
                        for dx in 0..2 {
                            let lx = cx * 2 + dx;
                            let lz = cz * 2 + dz;
                            for ly in (0..16).rev() {
                                let id = chunk.get(lx, ly, lz);
                                if id.is_air() {
                                    continue;
                                }
                                if reg.get(id).kind == BlockKind::Liquid {
                                    let wx = origin.x + lx;
                                    let wy = origin.y + ly;
                                    let wz = origin.z + lz;
                                    let info = sample_water(wx, wy, wz);
                                    if info.kind != 0
                                        && info.surface
                                        && top.map_or(true, |(ty, _)| wy > ty)
                                    {
                                        top = Some((wy, info));
                                    }
                                }
                                break;
                            }
                        }
                    }
                    let Some((wy, mut info)) = top else { continue };
                    // A LOD1 water cell is one large quad; suppressing wave
                    // displacement keeps adjacent cells watertight when their
                    // packed flow metadata differs.
                    info.amplitude = 0;
                    let x0 = (cx * 2) as f32;
                    let z0 = (cz * 2) as f32;
                    let y = wy as f32 + info.height();
                    let positions = [
                        [x0, y, z0],
                        [x0, y, z0 + 2.0],
                        [x0 + 2.0, y, z0 + 2.0],
                        [x0 + 2.0, y, z0],
                    ];
                    let uvs = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]];
                    let base = transparent.vertices.len() as u32;
                    for c in 0..4 {
                        transparent.vertices.push(ChunkVertex::new(
                            positions[c],
                            uvs[c],
                            water_tile,
                            info.pack(),
                        ));
                    }
                    emit_quad_indices(&mut transparent, base);
                }
            }
        }
        ChunkMeshBundle {
            opaque: mesh,
            transparent,
        }
    }
}

fn add_skirt(
    mesh: &mut ChunkMesh,
    positions: [[f32; 3]; 4],
    id: BlockId,
    props: &[CubeProp],
    face: Face,
) {
    let p = prop(props, id);
    let tile = p.tiles[face as usize] as u32;
    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let base = mesh.vertices.len() as u32;
    for c in 0..4 {
        mesh.vertices
            .push(ChunkVertex::new(positions[c], uvs[c], tile, 0));
    }
    emit_quad_indices(mesh, base);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use crate::registry::BlockRegistry;
    use crate::water::WaterKind;
    use voxel_core::ChunkPos;

    #[test]
    fn section_boundaries_do_not_duplicate_or_drop_slab_faces() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        for z in 0..16 {
            for x in 0..16 {
                chunk.set(x, 0, z, stone);
            }
        }
        let sample = move |x: i32, y: i32, z: i32| {
            if y == 0 && (0..16).contains(&x) && (0..16).contains(&z) {
                stone
            } else {
                BlockId::AIR
            }
        };
        let mut combined = ChunkMeshBundle::default();
        for section in 0..8u8 {
            let b = ChunkMesher.build_section(
                section,
                &chunk,
                &reg,
                sample,
                air_water_sample,
                |_, _, _| true,
            );
            combined.opaque.append(&b.opaque);
        }
        // Four occupied 8x8 sections, each with top+bottom and two outer
        // sides: 4 quads per section. Internal section boundaries must add
        // nothing.
        assert_eq!(combined.opaque.vertices.len(), 4 * 4 * 4);
        assert_eq!(combined.opaque.indices.len(), 4 * 4 * 6);
    }

    #[test]
    fn water_fast_path_keeps_wall_above_lower_neighbour() {
        let reg = BlockRegistry::with_builtins();
        let water = reg.id_of("water").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(0, 0, 0, water);
        chunk.set(0, 1, 0, water);
        chunk.set(1, 0, 0, water);
        let sample = move |x: i32, y: i32, z: i32| {
            if matches!((x, y, z), (0, 0, 0) | (0, 1, 0) | (1, 0, 0)) {
                water
            } else {
                BlockId::AIR
            }
        };
        let water_sample = move |x: i32, y: i32, z: i32| {
            if matches!((x, y, z), (0, 0, 0) | (0, 1, 0) | (1, 0, 0)) {
                WaterVertexInfo {
                    kind: 1,
                    fill: if (x, y, z) == (1, 0, 0) { 128 } else { 255 },
                    surface: true,
                    ..Default::default()
                }
            } else {
                WaterVertexInfo::default()
            }
        };
        let bundle = ChunkMesher.build(&chunk, &reg, sample, water_sample, |_, _, _| true);
        // The full-height block at (0,0,0) has a lower water neighbour at
        // +X, so its exposed vertical wall must survive the interior skip.
        assert!(
            bundle.transparent.vertices.iter().any(|v| {
                let p = v.position();
                (p[0] - 1.0).abs() < 0.001 && p[1] <= 1.0
            }),
            "lower adjacent water must not hide the full-height wall"
        );
    }

    #[test]
    fn sections_cover_chunk_and_isolate_geometry() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(8, 8, 8, stone);
        let water = air_water_sample;
        let mut combined = ChunkMeshBundle::default();
        for section in 0..8u8 {
            let bundle = ChunkMesher.build_section(
                section,
                &chunk,
                &reg,
                air_sample,
                water,
                |_, _, _| true,
            );
            combined.opaque.append(&bundle.opaque);
            combined.transparent.append(&bundle.transparent);
        }
        // A single isolated block is six quads, all owned by section 7
        // (x=1, y=1, z=1).
        assert_eq!(combined.opaque.vertices.len(), 24);
        assert!(combined.transparent.is_empty());
        for v in &combined.opaque.vertices {
            let p = v.position();
            assert!((8.0..=9.0).contains(&p[0]));
            assert!((8.0..=9.0).contains(&p[1]));
            assert!((8.0..=9.0).contains(&p[2]));
        }
    }

    #[test]
    fn lod1_stays_within_chunk_extent() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        for z in 0..16 {
            for x in 0..16 {
                chunk.set(x, 0, z, stone);
            }
        }
        let lod = ChunkMesher.build_lod1(&chunk, &reg, air_sample, air_water_sample, |_, _, _| true);
        assert!(!lod.opaque.is_empty(), "LOD1 must emit terrain quads");
        for v in &lod.opaque.vertices {
            let p = v.position();
            assert!(
                p[0] >= -1.0 && p[0] <= 16.0 && p[1] >= -1.0 && p[1] <= 16.0 && p[2] >= -1.0 && p[2] <= 16.0,
                "LOD1 vertex {p:?} escaped the 16-block chunk (skirts may extend by 1)"
            );
        }
    }

    #[test]
    fn chunk_vertex_is_packed_to_16_bytes() {
        assert_eq!(std::mem::size_of::<ChunkVertex>(), 16);
        assert_eq!(std::mem::offset_of!(ChunkVertex, water), 12);
        let v = ChunkVertex::new([1.5, 2.0, 16.0], [3.25, 0.5], 7, 0x1234);
        assert_eq!(v.position(), [1.5, 2.0, 16.0]);
        assert_eq!(v.uv(), [3.25, 0.5]);
        assert_eq!(v.tile, 7);
        let round = bytemuck::from_bytes::<ChunkVertex>(bytemuck::bytes_of(&v));
        assert_eq!(round.water, 0x1234);
    }

    fn air_sample(_x: i32, _y: i32, _z: i32) -> BlockId {
        BlockId::AIR
    }
    fn air_water_sample(_x: i32, _y: i32, _z: i32) -> WaterVertexInfo {
        WaterVertexInfo::default()
    }
    fn air_loaded_sample(_x: i32, _y: i32, _z: i32) -> bool {
        false
    }

    #[test]
    fn border_face_ownership_and_same_kind_culling() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let glass = reg.id_of("glass").unwrap();
        let mesher = ChunkMesher;
        let water = air_water_sample;

        // One stone block with a loaded stone neighbour on the -X border:
        // the neighbour owns the shared face, so only 5 quads are emitted.
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(0, 0, 0, stone);
        let sample = move |x: i32, y: i32, z: i32| {
            if x == -1 && y == 0 && z == 0 {
                stone
            } else {
                BlockId::AIR
            }
        };
        let bundle = mesher.build(&chunk, &reg, sample, water, |_, _, _| true);
        assert_eq!(bundle.opaque.vertices.len(), 5 * 4);

        // Same block, but the neighbour chunk is loaded and air: this chunk
        // must draw its own -X face (6 quads).
        let bundle = mesher.build(
            &chunk,
            &reg,
            |_, _, _| BlockId::AIR,
            water,
            |_, _, _| true,
        );
        assert_eq!(bundle.opaque.vertices.len(), 6 * 4);

        // Unloaded neighbour: also 6 quads (the edge of the loaded world is
        // treated as open air).
        let bundle = mesher.build(
            &chunk,
            &reg,
            |_, _, _| BlockId::AIR,
            water,
            |_, _, _| false,
        );
        assert_eq!(bundle.opaque.vertices.len(), 6 * 4);

        // Two glass blocks side by side: the shared face is culled because
        // both are non-opaque and of the same kind. The four remaining
        // coplanar side pairs greedily merge, leaving 6 quads.
        let mut glass_chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        glass_chunk.set(5, 5, 5, glass);
        glass_chunk.set(6, 5, 5, glass);
        let bundle = mesher.build(
            &glass_chunk,
            &reg,
            |_, _, _| BlockId::AIR,
            water,
            |_, _, _| true,
        );
        assert_eq!(bundle.transparent.vertices.len(), 6 * 4);
    }

    #[test]
    fn water_surface_vertices_carry_wave_params() {
        let reg = BlockRegistry::with_builtins();
        let water = reg.id_of("water").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(0, 1, 0, water);
        let info = WaterVertexInfo {
            kind: WaterKind::Ocean.as_u8(),
            fill: 255,
            amplitude: 48,
            flow_angle: 10,
            flow_speed: 90,
            rapids: 0,
            surface: true,
        };
        let sample_water = move |x: i32, y: i32, z: i32| {
            if (x, y, z) == (0, 1, 0) {
                info
            } else {
                WaterVertexInfo::default()
            }
        };
        let bundle = ChunkMesher.build(
            &chunk,
            &reg,
            air_sample,
            sample_water,
            air_loaded_sample,
        );
        assert!(!bundle.transparent.vertices.is_empty());
        let top = bundle
            .transparent
            .vertices
            .iter()
            .find(|v| (v.position()[1] - 2.0).abs() < 1e-5 && v.water != 0)
            .expect("top water vertex carries animation data");
        assert_eq!((top.water >> 3) & 1, 1, "surface flag must be set");
        assert_eq!((top.water >> 8) & 0xFF, 48, "amplitude must round-trip");
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
        );
        // Two perpendicular planes, four vertices each → 8 verts total.
        assert_eq!(
            bundle.opaque.vertices.len(),
            8,
            "foliage cross should be 2 perpendicular planes × 4 verts",
        );
        let cx = 0.5f32;
        let cz = 0.5f32;
        for v in &bundle.opaque.vertices {
            let p = v.position();
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

    #[test]
    fn lod1_emits_a_transparent_water_surface() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let water = reg.id_of("water").unwrap();
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        for z in 0..16 {
            for x in 0..16 {
                chunk.set(x, 0, z, stone);
                chunk.set(x, 1, z, water);
            }
        }
        let bundle = ChunkMesher.build_lod1(
            &chunk,
            &reg,
            |x, y, z| chunk.get(x, y, z),
            |_, _, _| WaterVertexInfo {
                kind: WaterKind::Ocean as u8,
                fill: 255,
                surface: true,
                ..WaterVertexInfo::default()
            },
            |_, _, _| true,
        );
        assert!(!bundle.opaque.indices.is_empty());
        assert!(
            !bundle.transparent.indices.is_empty(),
            "LOD1 must keep the water surface in the transparent pass"
        );
        assert_eq!(bundle.transparent.indices.len() % 6, 0);
    }
}
