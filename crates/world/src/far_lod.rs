//! Voxy-style far terrain LOD tiles.
//!
//! Full 16³ chunks only exist inside the standard render distance. Beyond
//! that, terrain is sampled from the same `TerrainGenerator` on a coarse
//! global grid and emitted as large surface tiles. Cells use the same block
//! textures as the nearby chunks, and every water body contributes a coarse
//! transparent surface instead of disappearing at the LOD boundary.

use voxel_core::{BlockId, WORLD_HEIGHT_BLOCKS, SEA_LEVEL};

use crate::gen::{BiomeId, ColumnInfo, TerrainGenerator};
use crate::mesh::{ChunkMesh, ChunkVertex};
use crate::registry::{BlockRegistry, Face};
use crate::water::WaterVertexInfo;

/// Side length of one far tile, in chunks.
pub const FAR_TILE_CHUNKS: i32 = 8;
/// Side length of one far tile, in blocks.
pub const FAR_TILE_BLOCKS: i32 = FAR_TILE_CHUNKS * voxel_core::CHUNK_SIZE;
/// Horizontal resolution of a far tile. Four blocks is close enough to the
/// full-chunk LOD1 mesh (two-block cells) that the hand-off stays smooth,
/// while a 128-chunk view still fits comfortably in memory.
pub const FAR_CELL_BLOCKS: i32 = 4;
const GRID: i32 = FAR_TILE_BLOCKS / FAR_CELL_BLOCKS;

/// Renderer key for a far tile. The negative Y namespace keeps far tiles from
/// colliding with real chunks in the renderer's chunk map.
pub fn far_tile_chunk_pos(key: (i32, i32)) -> voxel_core::math::ChunkPos {
    voxel_core::math::ChunkPos::new(key.0 * FAR_TILE_CHUNKS, -1, key.1 * FAR_TILE_CHUNKS)
}

#[derive(Clone, Copy)]
struct Cell {
    top: f32,
    block: BlockId,
    water: Option<(f32, u32)>,
}

/// One generated far tile. Positions are tile-local in X/Z and world-space in
/// Y, so the renderer can use a single shared origin at the tile's corner.
pub struct FarLodTile {
    pub opaque: ChunkMesh,
    pub transparent: ChunkMesh,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

fn surface_block(info: &ColumnInfo, reg: &BlockRegistry) -> BlockId {
    let h = info.height.round() as i32;
    let underwater = info.water_surface > info.height + 0.05;
    let name = match info.biome {
        BiomeId::Desert | BiomeId::Beach => "sand",
        BiomeId::Mountains if h > SEA_LEVEL + 55 => "snow",
        BiomeId::Ocean if h < SEA_LEVEL => "gravel",
        _ if underwater => "dirt",
        _ => "grass",
    };
    reg.id_of(name).or_else(|| reg.id_of("stone")).unwrap_or(BlockId::AIR)
}

fn water_payload(info: &ColumnInfo) -> u32 {
    WaterVertexInfo {
        kind: info.water_kind.as_u8(),
        fill: 255,
        // Far water planes are only 8 blocks wide per cell; keeping them flat
        // avoids cracks between cells with different flow metadata while the
        // shader still gives them ripples and reflections.
        amplitude: 0,
        flow_angle: info.flow_angle,
        flow_speed: info.flow_speed,
        rapids: info.rapids,
        surface: true,
    }
    .pack()
}

fn sample_cell(gen: &TerrainGenerator, reg: &BlockRegistry, wx: i32, wz: i32) -> Cell {
    let info = gen.column_info(wx, wz);
    // Sit slightly below the rounded height so coarse tiles depth-test behind
    // the full-detail chunks they overlap at the standard-radius boundary.
    let top = (info.height.round() - 0.4).max(0.0);
    let block = surface_block(&info, reg);
    let water = if info.water_kind != crate::water::WaterKind::None
        && info.water_surface > info.height + 0.05
    {
        Some((info.water_surface - 0.35, water_payload(&info)))
    } else {
        None
    };
    Cell { top, block, water }
}

fn push_quad(mesh: &mut ChunkMesh, p: [[f32; 3]; 4], uv: [[f32; 2]; 4], tile: u16, water: u32) {
    let base = mesh.vertices.len() as u32;
    for i in 0..4 {
        mesh.vertices
            .push(ChunkVertex::new(p[i], uv[i], tile as u32, water));
    }
    mesh.indices.extend_from_slice(&[
        base,
        base + 1,
        base + 2,
        base,
        base + 2,
        base + 3,
    ]);
}

fn tile_of(reg: &BlockRegistry, id: BlockId, face: Face) -> u16 {
    reg.get(id).textures.tile(face)
}

fn emit_side(
    mesh: &mut ChunkMesh,
    face: Face,
    id: BlockId,
    reg: &BlockRegistry,
    p: [[f32; 3]; 4],
    uv_w: f32,
    uv_h: f32,
) {
    let tile = tile_of(reg, id, face);
    let uv = [[0.0, 0.0], [0.0, uv_h], [uv_w, uv_h], [uv_w, 0.0]];
    push_quad(mesh, p, uv, tile, 0);
}

/// Generate the far-LOD tile whose lower-left chunk coordinate is
/// `(tile_x, tile_z) * FAR_TILE_CHUNKS`.
pub fn build_far_lod_tile(
    gen: &TerrainGenerator,
    reg: &BlockRegistry,
    tile_x: i32,
    tile_z: i32,
) -> FarLodTile {
    let origin_x = tile_x * FAR_TILE_BLOCKS;
    let origin_z = tile_z * FAR_TILE_BLOCKS;
    // One-cell border on every side. Adjacent tiles sample the exact same
    // global positions, so their top surfaces meet without cracks.
    let mut cells = vec![Cell { top: 0.0, block: BlockId::AIR, water: None }; ((GRID + 2) * (GRID + 2)) as usize];
    let idx = |x: i32, z: i32| ((z + 1) * (GRID + 2) + (x + 1)) as usize;
    for z in -1..=GRID {
        for x in -1..=GRID {
            let wx = origin_x + x * FAR_CELL_BLOCKS + FAR_CELL_BLOCKS / 2;
            let wz = origin_z + z * FAR_CELL_BLOCKS + FAR_CELL_BLOCKS / 2;
            cells[idx(x, z)] = sample_cell(gen, reg, wx, wz);
        }
    }

    let mut opaque = ChunkMesh::default();
    let mut transparent = ChunkMesh::default();

    // Top surfaces, merged into horizontal runs of equal height/material.
    for z in 0..GRID {
        let mut x = 0;
        while x < GRID {
            let c = cells[idx(x, z)];
            let mut run = 1;
            while x + run < GRID {
                let n = cells[idx(x + run, z)];
                if n.block != c.block || (n.top - c.top).abs() > 0.01 {
                    break;
                }
                run += 1;
            }
            let x0 = (x * FAR_CELL_BLOCKS) as f32;
            let x1 = ((x + run) * FAR_CELL_BLOCKS) as f32;
            let z0 = (z * FAR_CELL_BLOCKS) as f32;
            let z1 = z0 + FAR_CELL_BLOCKS as f32;
            let tile = tile_of(reg, c.block, Face::PosY);
            let w = (run * FAR_CELL_BLOCKS) as f32;
            let d = FAR_CELL_BLOCKS as f32;
            push_quad(
                &mut opaque,
                [[x0, c.top, z0], [x0, c.top, z1], [x1, c.top, z1], [x1, c.top, z0]],
                [[0.0, 0.0], [0.0, d], [w, d], [w, 0.0]],
                tile,
                0,
            );
            x += run;
        }
    }

    // Interior vertical walls. The outer tile edge gets a short skirt below
    // instead, so neighbouring tiles do not emit the same wall twice.
    for x in 1..GRID {
        let mut z = 0;
        while z < GRID {
            let a = cells[idx(x - 1, z)];
            let b = cells[idx(x, z)];
            let (hi, lo, face) = if b.top > a.top + 0.01 {
                (b, a, Face::PosX)
            } else if a.top > b.top + 0.01 {
                (a, b, Face::NegX)
            } else {
                z += 1;
                continue;
            };
            let mut run = 1;
            while z + run < GRID {
                let a2 = cells[idx(x - 1, z + run)];
                let b2 = cells[idx(x, z + run)];
                let same = (hi.top - if face == Face::PosX { b2.top } else { a2.top }).abs() < 0.01
                    && (lo.top - if face == Face::PosX { a2.top } else { b2.top }).abs() < 0.01
                    && (if face == Face::PosX { b2.top > a2.top } else { a2.top > b2.top });
                if !same {
                    break;
                }
                run += 1;
            }
            let xe = (x * FAR_CELL_BLOCKS) as f32;
            let z0 = (z * FAR_CELL_BLOCKS) as f32;
            let z1 = ((z + run) * FAR_CELL_BLOCKS) as f32;
            let p = if face == Face::PosX {
                [[xe, lo.top, z0], [xe, hi.top, z0], [xe, hi.top, z1], [xe, lo.top, z1]]
            } else {
                [[xe, lo.top, z1], [xe, hi.top, z1], [xe, hi.top, z0], [xe, lo.top, z0]]
            };
            emit_side(&mut opaque, face, hi.block, reg, p, (run * FAR_CELL_BLOCKS) as f32, hi.top - lo.top);
            z += run;
        }
    }
    for z in 1..GRID {
        let mut x = 0;
        while x < GRID {
            let a = cells[idx(x, z - 1)];
            let b = cells[idx(x, z)];
            let (hi, lo, face) = if b.top > a.top + 0.01 {
                (b, a, Face::PosZ)
            } else if a.top > b.top + 0.01 {
                (a, b, Face::NegZ)
            } else {
                x += 1;
                continue;
            };
            let mut run = 1;
            while x + run < GRID {
                let a2 = cells[idx(x + run, z - 1)];
                let b2 = cells[idx(x + run, z)];
                let same = (hi.top - if face == Face::PosZ { b2.top } else { a2.top }).abs() < 0.01
                    && (lo.top - if face == Face::PosZ { a2.top } else { b2.top }).abs() < 0.01
                    && (if face == Face::PosZ { b2.top > a2.top } else { a2.top > b2.top });
                if !same {
                    break;
                }
                run += 1;
            }
            let ze = (z * FAR_CELL_BLOCKS) as f32;
            let x0 = (x * FAR_CELL_BLOCKS) as f32;
            let x1 = ((x + run) * FAR_CELL_BLOCKS) as f32;
            let p = if face == Face::PosZ {
                [[x1, lo.top, ze], [x1, hi.top, ze], [x0, hi.top, ze], [x0, lo.top, ze]]
            } else {
                [[x0, lo.top, ze], [x0, hi.top, ze], [x1, hi.top, ze], [x1, lo.top, ze]]
            };
            emit_side(&mut opaque, face, hi.block, reg, p, (run * FAR_CELL_BLOCKS) as f32, hi.top - lo.top);
            x += run;
        }
    }

    // Water: one transparent top plane per cell, merged into texture runs.
    for z in 0..GRID {
        let mut x = 0;
        while x < GRID {
            let Some((wy, payload)) = cells[idx(x, z)].water else {
                x += 1;
                continue;
            };
            let mut run = 1;
            while x + run < GRID {
                if cells[idx(x + run, z)].water != Some((wy, payload)) {
                    break;
                }
                run += 1;
            }
            let x0 = (x * FAR_CELL_BLOCKS) as f32;
            let x1 = ((x + run) * FAR_CELL_BLOCKS) as f32;
            let z0 = (z * FAR_CELL_BLOCKS) as f32;
            let z1 = z0 + FAR_CELL_BLOCKS as f32;
            let w = (run * FAR_CELL_BLOCKS) as f32;
            let d = FAR_CELL_BLOCKS as f32;
            let tile = reg
                .id_of("water")
                .map(|id| tile_of(reg, id, Face::PosY))
                .unwrap_or(0);
            push_quad(
                &mut transparent,
                [[x0, wy, z0], [x0, wy, z1], [x1, wy, z1], [x1, wy, z0]],
                [[0.0, 0.0], [0.0, d], [w, d], [w, 0.0]],
                tile,
                payload,
            );
            x += run;
        }
    }

    // Short perimeter skirts hide the seam where a coarse tile meets the
    // full-detail world or the next LOD tile. They only drop downward, so they
    // never float above the surface.
    const SKIRT_DEPTH: f32 = 32.0;
    for i in 0..GRID {
        let c0 = cells[idx(i, 0)];
        let c1 = cells[idx(i, GRID - 1)];
        let c2 = cells[idx(0, i)];
        let c3 = cells[idx(GRID - 1, i)];
        let x0 = (i * FAR_CELL_BLOCKS) as f32;
        let x1 = x0 + FAR_CELL_BLOCKS as f32;
        let z0 = (i * FAR_CELL_BLOCKS) as f32;
        let z1 = z0 + FAR_CELL_BLOCKS as f32;
        for (cell, p, face) in [
            (c0, [[x0, c0.top, 0.0], [x1, c0.top, 0.0], [x1, (c0.top - SKIRT_DEPTH).max(0.0), 0.0], [x0, (c0.top - SKIRT_DEPTH).max(0.0), 0.0]], Face::NegZ),
            (c1, [[x1, (c1.top - SKIRT_DEPTH).max(0.0), FAR_TILE_BLOCKS as f32], [x1, c1.top, FAR_TILE_BLOCKS as f32], [x0, c1.top, FAR_TILE_BLOCKS as f32], [x0, (c1.top - SKIRT_DEPTH).max(0.0), FAR_TILE_BLOCKS as f32]], Face::PosZ),
            (c2, [[0.0, (c2.top - SKIRT_DEPTH).max(0.0), z1], [0.0, c2.top, z1], [0.0, c2.top, z0], [0.0, (c2.top - SKIRT_DEPTH).max(0.0), z0]], Face::NegX),
            (c3, [[FAR_TILE_BLOCKS as f32, c3.top, z0], [FAR_TILE_BLOCKS as f32, c3.top, z1], [FAR_TILE_BLOCKS as f32, (c3.top - SKIRT_DEPTH).max(0.0), z1], [FAR_TILE_BLOCKS as f32, (c3.top - SKIRT_DEPTH).max(0.0), z0]], Face::PosX),
        ] {
            emit_side(&mut opaque, face, cell.block, reg, p, FAR_CELL_BLOCKS as f32, SKIRT_DEPTH);
        }
    }

    FarLodTile {
        opaque,
        transparent,
        bounds_min: [origin_x as f32, 0.0, origin_z as f32],
        bounds_max: [(origin_x + FAR_TILE_BLOCKS) as f32, WORLD_HEIGHT_BLOCKS as f32, (origin_z + FAR_TILE_BLOCKS) as f32],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_tiles_have_surfaces_water_and_tile_sized_bounds() {
        let gen = TerrainGenerator::new(5);
        let reg = BlockRegistry::with_builtins();
        let mut saw_water = false;
        for tz in -2..=2 {
            for tx in -2..=2 {
                let tile = build_far_lod_tile(&gen, &reg, tx, tz);
                assert!(!tile.opaque.indices.is_empty());
                assert_eq!(tile.opaque.indices.len() % 6, 0);
                assert_eq!(tile.transparent.indices.len() % 6, 0);
                assert_eq!(tile.bounds_min[1], 0.0);
                assert_eq!(tile.bounds_max[1], WORLD_HEIGHT_BLOCKS as f32);
                assert_eq!(
                    tile.bounds_max[0] - tile.bounds_min[0],
                    FAR_TILE_BLOCKS as f32
                );
                assert_eq!(
                    tile.bounds_max[2] - tile.bounds_min[2],
                    FAR_TILE_BLOCKS as f32
                );
                if !tile.transparent.indices.is_empty() {
                    saw_water = true;
                }
            }
        }
        assert!(saw_water, "expected at least one far tile with water");
    }
}