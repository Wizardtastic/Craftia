//! Tree generation — deterministic per-column tree placement with multiple
//! tree varieties (oak, big oak, birch, spruce) keyed by biome.
//!
//! All placement is deterministic from the world seed via `hash2` (same
//! hasher used by `gen.rs`). Cross-chunk writes are supported through a
//! `neighbour_set` closure: blocks outside the current chunk are written
//! via the closure; if the neighbouring chunk isn't loaded the write is
//! silently dropped (the block just doesn't appear until that chunk
//! generates and re-triggers decoration).

use glam::IVec3;
use voxel_core::{BlockId, CHUNK_SIZE};

use crate::registry::BlockRegistry;

// ---------------------------------------------------------------------------
// Deterministic hash (identical to `gen::hash2`; duplicated here to avoid a
// circular module dependency between `gen` and `tree`).
// ---------------------------------------------------------------------------

/// Deterministic hash of (seed, x, z) into [0, 1). Identical to `gen::hash2`.
#[inline]
pub fn hash2(seed: i32, x: i32, z: i32) -> f32 {
    let mut h = (seed as u32).wrapping_mul(374761393);
    h = h.wrapping_add(x as u32).wrapping_mul(668265263);
    h = h.wrapping_add(z as u32).wrapping_mul(1274126177);
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    (h >> 8) as f32 / ((1u32 << 24) as f32)
}

// ---------------------------------------------------------------------------
// Tree variety
// ---------------------------------------------------------------------------

/// Which tree shape to place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeType {
    /// Standard small oak: 4–6 log trunk, 3×3 canopy.
    Oak,
    /// Large branching oak: 2–3 main limbs, expansive canopy. Forest-only.
    BigOak,
    /// White birch: 5–7 log trunk, compact 2×2×2 canopy at the top.
    Birch,
    /// Tall spruce: 7–10 log trunk, tapered conical canopy (3×3 → 1×1).
    Spruce,
}

// ---------------------------------------------------------------------------
// TreePlacer — writes blocks either into the current chunk or via the
//              neighbour_set closure for cross-chunk placement.
// ---------------------------------------------------------------------------

/// Thin helper that routes `set(world_x, world_y, world_z, block)` into
/// either `chunk.set()` (when the coordinate falls inside the chunk) or the
/// `neighbour_set` closure (for cross-chunk spills). Assembled once per tree
/// and passed through to each shape function.
struct TreePlacer<'a> {
    chunk: &'a mut crate::chunk::Chunk,
    origin: IVec3,
    neighbour_set: &'a dyn Fn(i32, i32, i32, BlockId) -> bool,
}

impl<'a> TreePlacer<'a> {
    /// Place a block at world coordinates `(wx, wy, wz)`. Only overwrites
    /// positions that are currently air. Returns true if the block was
    /// actually written.
    fn set(&mut self, wx: i32, wy: i32, wz: i32, id: BlockId) -> bool {
        let lx = wx - self.origin.x;
        let ly = wy - self.origin.y;
        let lz = wz - self.origin.z;
        if (0..CHUNK_SIZE).contains(&lx)
            && (0..CHUNK_SIZE).contains(&ly)
            && (0..CHUNK_SIZE).contains(&lz)
        {
            if self.chunk.get(lx, ly, lz).is_air() {
                self.chunk.set(lx, ly, lz, id);
                return true;
            }
            false
        } else {
            (self.neighbour_set)(wx, wy, wz, id)
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Try to place a tree at world column `(wx, wz)` where the surface block
/// sits at local y `surface_ly` inside the chunk. `seed` is the world seed.
/// `neighbour_set` writes blocks outside the current chunk.
/// Placement parameters for [`try_place_tree`].
#[derive(Clone, Copy, Debug)]
pub struct TreePlacement {
    pub seed: i32,
    pub wx: i32,
    pub wz: i32,
    pub surface_ly: i32,
    pub tree_type: TreeType,
}

///
/// Returns the number of blocks placed (0 if no tree was placed).
pub fn try_place_tree(
    chunk: &mut crate::chunk::Chunk,
    reg: &BlockRegistry,
    placement: TreePlacement,
    neighbour_set: &dyn Fn(i32, i32, i32, BlockId) -> bool,
) -> u32 {
    let TreePlacement {
        seed,
        wx,
        wz,
        surface_ly,
        tree_type,
    } = placement;
    let origin = voxel_core::math::chunk_origin(chunk.pos);
    let mut placer = TreePlacer {
        chunk,
        origin,
        neighbour_set,
    };

    let (wood, leaves) = tree_blocks(reg, tree_type);
    let (Some(wood), Some(leaves)) = (wood, leaves) else {
        return 0;
    };

    match tree_type {
        TreeType::Oak => place_oak(&mut placer, seed, wx, wz, surface_ly, wood, leaves),
        TreeType::BigOak => place_big_oak(&mut placer, seed, wx, wz, surface_ly, wood, leaves),
        TreeType::Birch => place_birch(&mut placer, seed, wx, wz, surface_ly, wood, leaves),
        TreeType::Spruce => place_spruce(&mut placer, seed, wx, wz, surface_ly, wood, leaves),
    }
}

/// Pick a tree type for the given column and biome. Returns `None` if no
/// tree should be placed (density / biome check failed).
pub fn pick_tree_type(
    seed: i32,
    wx: i32,
    wz: i32,
    biome: super::gen::BiomeId,
    noise_val: f32,
    hash_val: f32,
) -> Option<TreeType> {
    // Density gating: per-biome thresholds on the cellular-noise + hash combo.
    let density_pass = match biome {
        super::gen::BiomeId::Ocean | super::gen::BiomeId::Beach => {
            return None; // no trees on beaches or in oceans
        }
        super::gen::BiomeId::Plains => {
            // Sparse: fewer trees, mostly small oaks
            noise_val >= 0.60 && hash_val <= 0.06
        }
        super::gen::BiomeId::Forest => {
            // Dense: many trees, mix of oaks, birch, and big oaks
            noise_val >= 0.45 && hash_val <= 0.16
        }
        super::gen::BiomeId::Desert => {
            // Desert surfaces are sand, not grass. The surface scan in
            // `decorate()` only looks for `grass` blocks, so trees won't
            // actually place in deserts. Return None rather than a
            // misleading tree type.
            return None;
        }
        super::gen::BiomeId::Mountains => {
            // Medium: spruce trees
            noise_val >= 0.52 && hash_val <= 0.10
        }
    };
    if !density_pass {
        return None;
    }

    let secondary = hash2(seed, wx + 1, wz);
    match biome {
        super::gen::BiomeId::Forest => {
            if secondary < 0.25 {
                Some(TreeType::Birch)
            } else if secondary < 0.40 {
                Some(TreeType::BigOak)
            } else {
                Some(TreeType::Oak)
            }
        }
        super::gen::BiomeId::Mountains => {
            if secondary < 0.45 {
                Some(TreeType::Spruce)
            } else {
                Some(TreeType::Oak) // occasional oak in mountain valleys
            }
        }
        // Desert was already handled above by the density_pass `return None`.
        super::gen::BiomeId::Desert => unreachable!(),
        _ => {
            if secondary < 0.10 {
                Some(TreeType::BigOak) // rare big oak in plains
            } else {
                Some(TreeType::Oak)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn tree_blocks(reg: &BlockRegistry, tt: TreeType) -> (Option<BlockId>, Option<BlockId>) {
    match tt {
        TreeType::Oak | TreeType::BigOak => (reg.id_of("wood"), reg.id_of("leaves")),
        TreeType::Birch => (reg.id_of("birch_log"), reg.id_of("birch_leaves")),
        TreeType::Spruce => (reg.id_of("spruce_log"), reg.id_of("spruce_leaves")),
    }
}

// ---------------------------------------------------------------------------
// Tree shapes
// ---------------------------------------------------------------------------

/// Standard oak: 4–6 log trunk + 3×3×2 canopy.
fn place_oak(
    p: &mut TreePlacer<'_>,
    seed: i32,
    wx: i32,
    wz: i32,
    surface_ly: i32,
    wood: BlockId,
    leaves: BlockId,
) -> u32 {
    let sy = p.origin.y + surface_ly;
    let trunk_h = 4 + (hash2(seed, wx + 100, wz + 100) * 3.0) as i32;
    let trunk_top = sy + trunk_h;
    let mut count = 0u32;

    // Trunk.
    for ty in (sy + 1)..=trunk_top {
        if p.set(wx, ty, wz, wood) {
            count += 1;
        }
    }

    // Canopy: wider ring at the base (radius 2), then 3×3 above (radius 1).
    let canopy_base = trunk_top;
    for dy in 0..=2 {
        let cy = canopy_base + dy;
        let radius: i32 = if dy == 0 { 2 } else { 1 };
        for dx in -radius..=radius {
            for dz in -radius..=radius {
                // Leave the trunk column empty except at the top canopy layer.
                if dx == 0 && dz == 0 && dy < 2 {
                    continue;
                }
                // Round corners on the wide base ring.
                if dy == 0 && dx.abs() == 2 && dz.abs() == 2 {
                    continue;
                }
                if p.set(wx + dx, cy, wz + dz, leaves) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Big branching oak: 2–3 block trunk then 2–4 diagonal limbs extending
/// 2–3 blocks, each capped with leaf blobs.
fn place_big_oak(
    p: &mut TreePlacer<'_>,
    seed: i32,
    wx: i32,
    wz: i32,
    surface_ly: i32,
    wood: BlockId,
    leaves: BlockId,
) -> u32 {
    let sy = p.origin.y + surface_ly;
    let trunk_h = 3 + (hash2(seed, wx + 200, wz + 200) * 2.0) as i32;
    let branch_origin_y = sy + trunk_h;
    let mut count = 0u32;

    // Main trunk (short; enough to support the canopy).
    for ty in (sy + 1)..=branch_origin_y {
        if p.set(wx, ty, wz, wood) {
            count += 1;
        }
    }

    // 3–5 diagonal limbs.
    let limb_count = 3 + (hash2(seed, wx + 300, wz + 300) * 3.0) as i32;
    for i in 0..limb_count {
        let angle = hash2(seed, wx + 400 + i, wz + 400 + i) * std::f32::consts::TAU;
        let limb_len = 2 + (hash2(seed, wx + 500 + i, wz + 500 + i) * 3.0) as i32;
        let dx = (angle.cos().round() as i32).signum();
        let dz = (angle.sin().round() as i32).signum();

        let mut cx = wx;
        let mut cz = wz;
        for step in 1..=limb_len {
            let limb_y = branch_origin_y + step / 2;
            cx += dx;
            cz += dz;
            if p.set(cx, limb_y, cz, wood) {
                count += 1;
            }
            // Leaf blob at the end of each limb.
            if step == limb_len {
                for ldy in -1..=1 {
                    for ldx in -1..=1 {
                        for ldz in -1..=1 {
                            if ldx == 0 && ldz == 0 && ldy <= 0 {
                                continue;
                            }
                            if p.set(cx + ldx, limb_y + ldy, cz + ldz, leaves) {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Cap the trunk with a central leaf cluster.
    for dy in 0..=2 {
        for dx in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dz == 0 && dy < 1 {
                    continue;
                }
                if p.set(wx + dx, branch_origin_y + dy, wz + dz, leaves) {
                    count += 1;
                }
            }
        }
    }

    count
}

/// White birch: 5–7 log trunk + compact 2×2×2 canopy at the very top.
fn place_birch(
    p: &mut TreePlacer<'_>,
    seed: i32,
    wx: i32,
    wz: i32,
    surface_ly: i32,
    wood: BlockId,
    leaves: BlockId,
) -> u32 {
    let sy = p.origin.y + surface_ly;
    let trunk_h = 5 + (hash2(seed, wx + 600, wz + 600) * 3.0) as i32;
    let trunk_top = sy + trunk_h;
    let mut count = 0u32;

    // Trunk.
    for ty in (sy + 1)..=trunk_top {
        if p.set(wx, ty, wz, wood) {
            count += 1;
        }
    }

    // Canopy: tight 2×2×3 cluster at the top of the trunk.
    for dy in 0..3 {
        let cy = trunk_top + dy;
        for dx in -1i32..=0 {
            for dz in -1i32..=0 {
                if dx == 0 && dz == 0 && dy < 1 {
                    continue;
                }
                if p.set(wx + dx, cy, wz + dz, leaves) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Tall spruce: 7–10 log trunk + tapered conical canopy.
fn place_spruce(
    p: &mut TreePlacer<'_>,
    seed: i32,
    wx: i32,
    wz: i32,
    surface_ly: i32,
    wood: BlockId,
    leaves: BlockId,
) -> u32 {
    let sy = p.origin.y + surface_ly;
    let trunk_h = 7 + (hash2(seed, wx + 700, wz + 700) * 4.0) as i32;
    let trunk_top = sy + trunk_h;
    let mut count = 0u32;

    // Trunk.
    for ty in (sy + 1)..=trunk_top {
        if p.set(wx, ty, wz, wood) {
            count += 1;
        }
    }

    // Tapered canopy: 3×3 at bottom (trunk_top - 3), narrowing to 1×1 at
    // the very tip (trunk_top + 1).
    let canopy_bottom = trunk_top - 3;
    let canopy_top = trunk_top + 1;
    for cy in canopy_bottom..=canopy_top {
        let dist_from_bottom = cy - canopy_bottom;
        let radius: i32 = match dist_from_bottom {
            0..=2 => 1, // 3×3
            3 | 4 => 0,     // 1×1
            _ => continue,
        };
        for dx in -radius..=radius {
            for dz in -radius..=radius {
                if dx == 0 && dz == 0 && cy < canopy_top {
                    continue; // don't overwrite the trunk
                }
                if p.set(wx + dx, cy, wz + dz, leaves) {
                    count += 1;
                }
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use voxel_core::ChunkPos;

    fn test_chunk() -> crate::chunk::Chunk {
        let mut c = Chunk::new(ChunkPos::new(0, 4, 0)); // y=64..80
                                                        // Fill bottom with grass to simulate a surface.
        let reg = BlockRegistry::with_builtins();
        let grass = reg.id_of("grass").unwrap();
        let dirt = reg.id_of("dirt").unwrap();
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                c.set(lx, 0, lz, grass);
                c.set(lx, 1, lz, dirt);
                c.set(lx, 2, lz, dirt);
            }
        }
        c
    }

    #[test]
    fn oak_places_trunk_and_leaves() {
        let mut c = test_chunk();
        let reg = BlockRegistry::with_builtins();
        let noop_neighbour = |_: i32, _: i32, _: i32, _: BlockId| false;
        let count = try_place_tree(
            &mut c,
            &reg,
            TreePlacement {
                seed: 42,
                wx: 4,
                wz: 4, // world x,z within chunk origin (0,64,0)
                surface_ly: 0, // surface_ly = 0 (grass at local y=0)
                tree_type: TreeType::Oak,
            },
            &noop_neighbour,
        );
        assert!(count > 0, "oak should place blocks");
        // Should have placed wood above the soil layer (test_chunk fills
        // local y=1..=2 with dirt, and trunks only overwrite air).
        let wood = reg.id_of("wood").unwrap();
        assert_eq!(c.get(4, 3, 4), wood, "trunk should grow above the soil");
    }

    #[test]
    fn birch_places_trunk_and_leaves() {
        let mut c = test_chunk();
        let reg = BlockRegistry::with_builtins();
        let noop_neighbour = |_: i32, _: i32, _: i32, _: BlockId| false;
        let count = try_place_tree(
            &mut c,
            &reg,
            TreePlacement {
                seed: 42,
                wx: 4,
                wz: 4,
                surface_ly: 0,
                tree_type: TreeType::Birch,
            },
            &noop_neighbour,
        );
        assert!(count > 0, "birch should place blocks");
    }

    #[test]
    fn spruce_places_trunk_and_leaves() {
        let mut c = test_chunk();
        let reg = BlockRegistry::with_builtins();
        let noop_neighbour = |_: i32, _: i32, _: i32, _: BlockId| false;
        let count = try_place_tree(
            &mut c,
            &reg,
            TreePlacement {
                seed: 42,
                wx: 4,
                wz: 4,
                surface_ly: 0,
                tree_type: TreeType::Spruce,
            },
            &noop_neighbour,
        );
        assert!(count > 0, "spruce should place blocks");
    }

    #[test]
    fn big_oak_places_trunk_and_leaves() {
        let mut c = test_chunk();
        let reg = BlockRegistry::with_builtins();
        let noop_neighbour = |_: i32, _: i32, _: i32, _: BlockId| false;
        let count = try_place_tree(
            &mut c,
            &reg,
            TreePlacement {
                seed: 42,
                wx: 4,
                wz: 4,
                surface_ly: 0,
                tree_type: TreeType::BigOak,
            },
            &noop_neighbour,
        );
        assert!(count > 0, "big oak should place blocks");
    }

    #[test]
    fn pick_tree_type_ocean_returns_none() {
        assert!(pick_tree_type(42, 0, 0, crate::gen::BiomeId::Ocean, 0.6, 0.05).is_none());
    }

    #[test]
    fn pick_tree_type_beach_returns_none() {
        assert!(pick_tree_type(42, 0, 0, crate::gen::BiomeId::Beach, 0.6, 0.05).is_none());
    }

    #[test]
    fn pick_tree_type_desert_returns_none() {
        assert!(pick_tree_type(42, 0, 0, crate::gen::BiomeId::Desert, 0.6, 0.03).is_none());
    }

    #[test]
    fn pick_tree_type_forest_dense() {
        // Forest should be dense: lower noise threshold, higher hash threshold.
        assert!(pick_tree_type(42, 0, 0, crate::gen::BiomeId::Forest, 0.5, 0.12).is_some());
    }

    #[test]
    fn pick_tree_type_plains_sparse() {
        // Plains at moderate noise+hash shouldn't spawn.
        assert!(pick_tree_type(42, 0, 0, crate::gen::BiomeId::Plains, 0.55, 0.10).is_none());
        // But high noise + low hash should.
        assert!(pick_tree_type(42, 0, 0, crate::gen::BiomeId::Plains, 0.65, 0.04).is_some());
    }

    #[test]
    fn cross_chunk_uses_neighbour_set() {
        let mut c = test_chunk();
        let reg = BlockRegistry::with_builtins();
        // `RefCell` keeps the closure `Fn`: `try_place_tree` expects a
        // `&dyn Fn`, so the closure must not capture `cross_writes` by `&mut`.
        let cross_writes = std::cell::RefCell::new(Vec::new());
        let neighbour_set = |x: i32, y: i32, z: i32, id: BlockId| {
            cross_writes.borrow_mut().push((x, y, z, id));
            true
        };
        // Place a tree near the chunk edge so some canopy spills to
        // neighbouring chunks.
        let _count = try_place_tree(
            &mut c,
            &reg,
            TreePlacement {
                seed: 42,
                wx: 0,
                wz: 0,
                surface_ly: 0,
                tree_type: TreeType::Oak,
            },
            &neighbour_set,
        );
        // Oak canopy at radius 2 from (wx=0,wz=0) should spill into
        // negative X/Z which is outside chunk origin (0, 64, 0).
        assert!(
            !cross_writes.borrow().is_empty(),
            "canopy near chunk edge should spill cross-chunk"
        );
    }

    #[test]
    fn hash2_deterministic() {
        let a = hash2(42, 100, 200);
        let b = hash2(42, 100, 200);
        assert_eq!(a, b);
    }
}
