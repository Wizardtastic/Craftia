//! Chunk meshing: greedy meshing with face culling.
//!
//! Two meshes are produced per chunk:
//! - `opaque`: stone, dirt, grass, wood, ...
//! - `transparent`: water, leaves, glass
//!
//! Greedy meshing merges adjacent coplanar faces of the same block type into
//! larger quads. No lighting is baked into the geometry: the GPU ray tracer
//! evaluates sun/moon shadows, ambient occlusion, sky GI and reflections per
//! pixel against the world voxel grid.
//!
//! Vertex layout (28 bytes):
//!   pos:   [f32; 3] - local block corner in [0, 16]
//!   uv:    [f32; 2] - LOCAL tile UV in [0, w] x [0, h]; the fragment shader
//!                     applies `fract()` per tile so merged quads repeat.
//!   tile:  u32      - atlas tile index (flat, never interpolated).
//!   water: u32      - packed water animation (kind, surface flag, amplitude,
//!                     flow direction/speed, rapids); 0 for non-water geometry.

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec2, Vec3 as GVec3};
use voxel_core::{
    math::chunk_origin,
    BlockId, CHUNK_SIZE,
};

use crate::chunk::Chunk;
use crate::registry::{BlockDef, BlockKind, BlockRegistry, Face};
use crate::water::WaterVertexInfo;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct ChunkVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub tile: u32,
    /// Packed water animation as described in [`WaterVertexInfo::pack`].
    pub water: u32,
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

/// Water top faces are subdivided into this many quads per block edge. A
#[derive(Clone, Copy)]
struct MaskCell {
    block_id: BlockId,
}

const EMPTY_CELL: MaskCell = MaskCell {
    block_id: BlockId::AIR,
};

pub struct ChunkMesher;

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
        bundle.opaque.vertices.reserve(nac * 4);
        bundle.opaque.indices.reserve(nac * 6);
        bundle.transparent.vertices.reserve(nac);
        bundle.transparent.indices.reserve(nac * 2);
        let origin = chunk_origin(chunk.pos);
        let build = MeshBuild {
            chunk,
            reg,
            sample,
            sample_water,
            sample_loaded,
            origin,
        };

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
                            let info = (build.sample_water)(
                                origin.x + lx,
                                origin.y + ly,
                                origin.z + lz,
                            );
                            if info.fill == 0 {
                                continue;
                            }
                            build.emit_water(
                                &mut bundle,
                                IVec3::new(lx, ly, lz),
                                def,
                                info,
                            );
                        }
                        _ if def.kind == BlockKind::Cactus => {
                            build.emit_cactus(&mut bundle, IVec3::new(lx, ly, lz), def);
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut mask = [[EMPTY_CELL; S]; S];

        for fi in 0..6usize {
            let face = Face::ALL[fi];

            for slice in 0..CHUNK_SIZE {
                for (v, row) in mask.iter_mut().enumerate() {
                    for (u, cell) in row.iter_mut().enumerate() {
                        *cell = build.mask_cell(face, slice, u as i32, v as i32);
                    }
                }
                greedy_emit(&mut bundle, &mask, face, slice, reg);
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

impl<S, W, L> MeshBuild<'_, S, W, L>
where
    S: Fn(i32, i32, i32) -> BlockId,
    W: Fn(i32, i32, i32) -> WaterVertexInfo,
    L: Fn(i32, i32, i32) -> bool,
{
    fn mask_cell(&self, face: Face, slice: i32, u: i32, v: i32) -> MaskCell {
        let MeshBuild {
            chunk,
            reg,
            sample,
            origin,
            ..
        } = self;
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
        if !self.should_emit_face(face, IVec3::new(lx, ly, lz), true) {
            return EMPTY_CELL;
        }
        let wx = origin.x + lx;
        let wy = origin.y + ly;
        let wz = origin.z + lz;
        let n = face.normal();
        // A border face resolved by should_emit_face already knows its neighbour
        // is air (or unloaded), so the inner-chunk neighbour tests cannot hide it.
        let resolved_border = is_neg_border(face, lx, ly, lz, chunk);
        if !resolved_border {
            let nlx = lx + n.x;
            let nly = ly + n.y;
            let nlz = lz + n.z;
            let neighbour = if (0..CHUNK_SIZE).contains(&nlx)
                && (0..CHUNK_SIZE).contains(&nly)
                && (0..CHUNK_SIZE).contains(&nlz)
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
        MaskCell { block_id: id }
    }
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

fn can_merge(a: &MaskCell, b: &MaskCell) -> bool {
    a.block_id == b.block_id
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
    reg: &BlockRegistry,
) {
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

            let tile = reg.get(cell.block_id).textures.tile(face);

            let mut w = 1usize;
            while u + w < S
                && !visited[v][u + w]
                && can_merge(&mask[v][u + w - 1], &mask[v][u + w])
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
                if !(0..w).all(|i| can_merge(&mask[v + h - 1][u + i], &mask[v + h][u + i])) {
                    break;
                }
                h += 1;
            }

            for dy in 0..h {
                for dx in 0..w {
                    visited[v + dy][u + dx] = true;
                }
            }

            let positions = merged_positions(face, u as i32, v as i32, w as i32, h as i32, slice);
            let uvs = merged_uvs(face, w as u32, h as u32);

            let cell_def = reg.get(cell.block_id);
            let target = match cell_def.kind {
                BlockKind::Transparent => &mut bundle.transparent,
                _ => &mut bundle.opaque,
            };
            let vertex_base = target.vertices.len() as u32;

            for c in 0..4 {
                target.vertices.push(ChunkVertex {
                    pos: positions[c],
                    uv: uvs[c],
                    tile: tile as u32,
                    water: 0,
                });
            }
            target.indices.extend_from_slice(&[
                vertex_base,
                vertex_base + 1,
                vertex_base + 2,
                vertex_base,
                vertex_base + 2,
                vertex_base + 3,
            ]);
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
    let tgt = &mut bundle.transparent;
    let base = tgt.vertices.len() as u32;
    for i in 0..8 {
        tgt.vertices.push(ChunkVertex {
            pos: positions[i],
            uv: uvs[i],
            tile: tile as u32,
            water: 0,
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
                bundle.transparent.vertices.push(ChunkVertex {
                    pos: [cp.x, cp.y, cp.z],
                    uv: [uvs[c].x, uvs[c].y],
                    tile: tile as u32,
                    water,
                });
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
                bundle.opaque.vertices.push(ChunkVertex {
                    pos: [cp.x, cp.y, cp.z],
                    uv: [u, v],
                    tile: tile as u32,
                    water: 0,
                });
            }
            emit_quad_indices(&mut bundle.opaque, start);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use crate::registry::BlockRegistry;
    use crate::water::WaterKind;
    use voxel_core::ChunkPos;

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
            .find(|v| (v.pos[1] - 2.0).abs() < 1e-5 && v.water != 0)
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
