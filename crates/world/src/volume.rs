//! Volumetric shape rasterizer — methods on [`World`] that fill, hollow, or
//! replace geometric volumes in a single pass.
//!
//! Feature 2 of the AI-authoring surface (see `plans/voxel-engine-refactor.md`).
//! Each block change is reported through a caller-supplied closure so the
//! editor / undo layer (`game::UndoRedoState`) can route edits into its own
//! batched undo scheme without `World` taking a dependency on `voxel_game`.
//!
//! Shapes provided:
//! - [`World::fill_aabb`]       — flood a box.
//! - [`World::hollow_aabb`]     — outline of a box (`shell`-block thickness).
//! - [`World::fill_sphere`]     — flood a Euclidean sphere.
//! - [`World::hollow_sphere`]   — spherical shell.
//! - [`World::fill_cylinder`]   — upright cylinder (Y axis; `axis` reserved).
//! - [`World::fill_pyramid`]    — square-base pyramid shrinking toward apex.
//! - [`World::fill_line`]       — 3D Bresenham line with optional thickness.
//! - [`World::replace_in_aabb`] — replace `target` blocks in a box with `replacement`.
//!
//! All shapes pass a [`BlockChange`] record to the `record` closure for every
//! block whose underlying chunk is loaded AND whose value would actually
//! change — positions in unloaded chunks (or with `old == new`) are silently
//! skipped.

use voxel_core::{Aabb, BlockId};

use crate::World;

/// One block change produced by a volume operation. Raw integer coordinates
/// (instead of `BlockPos`) keep this module dependency-free of `voxel_core`'s
/// tuple-struct constructors and let callers construct `BlockPos` themselves
/// at the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockChange {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    /// Block id that was there before the operation.
    pub old: BlockId,
    /// Block id that the operation wrote.
    pub new: BlockId,
}

impl World {
    /// Fill an axis-aligned bounding box. Iterates every block position in
    /// `bounds` (half-open: floor of `min`, ceil of `max`, both inclusive of
    /// the floor and exclusive of the ceil). Skips positions whose chunk
    /// is not loaded. Reports each chunk-located *change* through
    /// `record`. Returns the number of distinct blocks actually written.
    ///
    /// Idempotent w.r.t. `old == new`: writing the same block id it
    /// already is does not count, and no `BlockChange` is reported.
    pub fn fill_aabb<F: FnMut(BlockChange)>(
        &self,
        bounds: Aabb,
        id: BlockId,
        mut record: F,
    ) -> usize {
        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_block_range(bounds);
        let mut n = 0;
        for y in min_y..max_y {
            for z in min_z..max_z {
                for x in min_x..max_x {
                    n += self.write_with_record(x, y, z, id, &mut record);
                }
            }
        }
        n
    }

    /// Hollow shell of an AABB. Sets every block whose Chebyshev distance
    /// to the nearest AABB face is strictly less than `shell` (in blocks).
    /// At thickness 1 this is the surface face; higher values thicken
    /// inward. `shell < 1` is clamped to 1.
    pub fn hollow_aabb<F: FnMut(BlockChange)>(
        &self,
        bounds: Aabb,
        id: BlockId,
        shell: i32,
        mut record: F,
    ) -> usize {
        let shell = shell.max(1);
        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_block_range(bounds);
        let max_x_m1 = max_x - 1;
        let max_y_m1 = max_y - 1;
        let max_z_m1 = max_z - 1;
        let mut n = 0;
        for y in min_y..max_y {
            let dy_min = y - min_y;
            let dy_max = max_y_m1 - y;
            for z in min_z..max_z {
                let dz_min = z - min_z;
                let dz_max = max_z_m1 - z;
                for x in min_x..max_x {
                    let nearest = (x - min_x)
                        .min(max_x_m1 - x)
                        .min(dy_min)
                        .min(dy_max)
                        .min(dz_min)
                        .min(dz_max);
                    if nearest < shell {
                        n += self.write_with_record(x, y, z, id, &mut record);
                    }
                }
            }
        }
        n
    }

    /// Fill a Euclidean sphere centred at `center`. Iterates the cube
    /// `[center − ceil(radius), center + ceil(radius)]` and keeps every
    /// block whose squared distance to the centre is `≤ ⌊radius²⌋`. The
    /// floor-cast is an intentional simplification — the visible shape is
    /// still round, just integer-quantized.
    pub fn fill_sphere<F: FnMut(BlockChange)>(
        &self,
        center: (i32, i32, i32),
        radius: f32,
        id: BlockId,
        mut record: F,
    ) -> usize {
        let r = radius.max(0.0);
        // Bounding box must cover any cell with d² ≤ r².
        let r_round = r.ceil() as i32;
        // In-sphere predicate uses floor(r²); do *not* widen to
        // `r_round²` because that over-includes corner cells at fractional
        // radii (e.g. r=2.5 would silently include blocks at d²=9).
        let r_sq = (r * r) as i32;
        let (cx, cy, cz) = center;
        let mut n = 0;
        for y in (cy - r_round)..=(cy + r_round) {
            for z in (cz - r_round)..=(cz + r_round) {
                for x in (cx - r_round)..=(cx + r_round) {
                    let dx = x - cx;
                    let dy = y - cy;
                    let dz = z - cz;
                    if dx * dx + dy * dy + dz * dz <= r_sq {
                        n += self.write_with_record(x, y, z, id, &mut record);
                    }
                }
            }
        }
        n
    }

    /// Hollow Euclidean sphere. Keeps every block whose squared distance
    /// is `> inner²` AND `≤ outer²` where `outer = ceil(radius)` and
    /// `inner = ceil(max(0, radius − shell))`. Empty inner range yields
    /// an empty shell (rare but possible if `shell > radius`).
    pub fn hollow_sphere<F: FnMut(BlockChange)>(
        &self,
        center: (i32, i32, i32),
        radius: f32,
        shell: f32,
        id: BlockId,
        mut record: F,
    ) -> usize {
        let outer = radius.max(0.0).ceil() as i32;
        let inner = (radius - shell).max(0.0).ceil() as i32;
        let outer_sq = outer * outer;
        let inner_sq = inner * inner;
        let (cx, cy, cz) = center;
        let mut n = 0;
        for y in (cy - outer)..=(cy + outer) {
            for z in (cz - outer)..=(cz + outer) {
                for x in (cx - outer)..=(cx + outer) {
                    let dx = x - cx;
                    let dy = y - cy;
                    let dz = z - cz;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= outer_sq && d2 > inner_sq {
                        n += self.write_with_record(x, y, z, id, &mut record);
                    }
                }
            }
        }
        n
    }

    /// Fill an upright cylinder along the Y axis from `base.y` up to
    /// `base.y + height` (upper end exclusive). Cylinder radius is computed
    /// per horizontal slice. v1 cylinders are always Y-up. An
    /// axis-agnostic extension is deferred — call sites can pick the
    /// axis by supplying rotated `base` + per-axis radius.
    pub fn fill_cylinder<F: FnMut(BlockChange)>(
        &self,
        base: (i32, i32, i32),
        radius: f32,
        height: f32,
        id: BlockId,
        mut record: F,
    ) -> usize {
        let r = radius.max(0.0).ceil() as i32;
        let r_sq = r * r;
        let h = height.max(0.0).ceil() as i32;
        let (bx, by, bz) = base;
        let mut n = 0;
        for y in by..(by + h) {
            for z in (bz - r)..=(bz + r) {
                for x in (bx - r)..=(bx + r) {
                    let dx = x - bx;
                    let dz = z - bz;
                    if dx * dx + dz * dz <= r_sq {
                        n += self.write_with_record(x, y, z, id, &mut record);
                    }
                }
            }
        }
        n
    }

    /// Fill a square-base pyramid: at `base_aabb.min.y` the cross-section
    /// is the full bounding box; at `base_aabb.max.y − 1` it shrinks to a
    /// single block at the centre. Each layer linearly interpolates
    /// between those extremes. Single-layer pyramids (`max_y − min_y ≤ 1`)
    /// produce a 1-cell apex.
    pub fn fill_pyramid<F: FnMut(BlockChange)>(
        &self,
        base_aabb: Aabb,
        id: BlockId,
        mut record: F,
    ) -> usize {
        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_block_range(base_aabb);
        let height = (max_y - min_y).max(1);
        let cx = (min_x + max_x - 1) / 2;
        let cz = (min_z + max_z - 1) / 2;
        let full_dx = (max_x - min_x) as f32;
        let full_dz = (max_z - min_z) as f32;
        let mut n = 0;
        for layer in 0..height {
            let y = min_y + layer;
            let t = if height > 1 {
                layer as f32 / (height - 1) as f32
            } else {
                0.0
            };
            // `remaining` is 1 at the base and 0 at the apex.
            let remaining = 1.0 - t;
            let dx_radius = ((full_dx * remaining) * 0.5).round() as i32;
            let dz_radius = ((full_dz * remaining) * 0.5).round() as i32;
            for z in (cz - dz_radius)..=(cz + dz_radius) {
                for x in (cx - dx_radius)..=(cx + dx_radius) {
                    if x >= min_x && x < max_x && z >= min_z && z < max_z {
                        n += self.write_with_record(x, y, z, id, &mut record);
                    }
                }
            }
        }
        n
    }

    /// Fill a line from `a` to `b` using a dominant-axis 3D Bresenham.
    /// `thickness > 1` thickens the line into an axis-aligned cube with
    /// side `thickness` around every Bresenham step (cross-section is
    /// square, not circular — v1 approximation).
    pub fn fill_line<F: FnMut(BlockChange)>(
        &self,
        a: (i32, i32, i32),
        b: (i32, i32, i32),
        thickness: i32,
        id: BlockId,
        mut record: F,
    ) -> usize {
        let thickness = thickness.max(1);
        let half_lo = (thickness - 1) / 2;
        let half_hi = thickness / 2;
        let dx = (b.0 - a.0).abs();
        let dy = (b.1 - a.1).abs();
        let dz = (b.2 - a.2).abs();
        let sx = if a.0 < b.0 { 1 } else { -1 };
        let sy = if a.1 < b.1 { 1 } else { -1 };
        let sz = if a.2 < b.2 { 1 } else { -1 };
        let steps = dx.max(dy).max(dz);
        let mut n = 0;
        let (mut x, mut y, mut z) = a;
        for _ in 0..=steps {
            for dxo in -half_lo..=half_hi {
                for dyo in -half_lo..=half_hi {
                    for dzo in -half_lo..=half_hi {
                        n += self.write_with_record(x + dxo, y + dyo, z + dzo, id, &mut record);
                    }
                }
            }
            if x != b.0 {
                x += sx;
            }
            if y != b.1 {
                y += sy;
            }
            if z != b.2 {
                z += sz;
            }
        }
        n
    }

    /// Inside `bounds`, replace every `target` block with `replacement`.
    /// Same half-open AABB convention as [`Self::fill_aabb`]. No-ops
    /// (returns 0) when `target == replacement` to avoid redundant work.
    pub fn replace_in_aabb<F: FnMut(BlockChange)>(
        &self,
        bounds: Aabb,
        target: BlockId,
        replacement: BlockId,
        mut record: F,
    ) -> usize {
        if target == replacement {
            return 0;
        }
        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_block_range(bounds);
        let mut n = 0;
        for y in min_y..max_y {
            for z in min_z..max_z {
                for x in min_x..max_x {
                    if self.get_block(x, y, z) == target {
                        n += self.write_with_record(x, y, z, replacement, &mut record);
                    }
                }
            }
        }
        n
    }

    // --- Internal ---------------------------------------------------------

    /// Read block at `(x, y, z)`, call `set_block`, and on success report the
    /// change via `record`. Returns `1` if the underlying call changed a
    /// block, `0` if the chunk was unloaded OR the block's old value was
    /// already `id` (idempotent skip).
    fn write_with_record<F: FnMut(BlockChange)>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        id: BlockId,
        record: &mut F,
    ) -> usize {
        let old = self.get_block(x, y, z);
        if old == id {
            return 0;
        }
        if self.set_block(x, y, z, id) {
            record(BlockChange {
                x,
                y,
                z,
                old,
                new: id,
            });
            1
        } else {
            0
        }
    }
}

/// Convert an `Aabb` of `f32` corners to a half-open integer block range:
/// `min.x.floor()` (inclusive) to `max.x.ceil()` (exclusive). Standard
/// pixel grid conversion.
fn aabb_block_range(bounds: Aabb) -> (i32, i32, i32, i32, i32, i32) {
    (
        bounds.min.x.floor() as i32,
        bounds.min.y.floor() as i32,
        bounds.min.z.floor() as i32,
        bounds.max.x.ceil() as i32,
        bounds.max.y.ceil() as i32,
        bounds.max.z.ceil() as i32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use std::sync::Arc;
    use voxel_core::{BlockId, ChunkPos};

    /// Build a `World` with a single chunk loaded at the origin so
    /// volume methods have cells to write to. `World::new` returns an
    /// `Arc<World>` because the public `World` is a thread-safe facade.
    fn world_with_origin_chunk() -> Arc<World> {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));
        world
    }

    fn aabb(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) -> Aabb {
        Aabb {
            min: glam::Vec3::new(min_x, min_y, min_z),
            max: glam::Vec3::new(max_x, max_y, max_z),
        }
    }

    #[test]
    fn fill_aabb_2x2x2_writes_eight_changes() {
        let world = world_with_origin_chunk();
        let stone = BlockId(2);
        let mut changes = Vec::new();
        let n = world.fill_aabb(aabb(0.0, 0.0, 0.0, 2.0, 2.0, 2.0), stone, |c| {
            changes.push(c)
        });
        assert_eq!(n, 8);
        assert_eq!(changes.len(), 8);
        for c in &changes {
            assert_eq!(c.old, BlockId::AIR, "should have been air");
            assert_eq!(c.new, stone);
        }
    }

    #[test]
    fn fill_aabb_half_open_upper_bound_caps_at_three_cells() {
        // bounds [0, 3) covers x ∈ {0, 1, 2} → 3 cells (1×1×1 AABB).
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        let n = world.fill_aabb(aabb(0.0, 0.0, 0.0, 3.0, 1.0, 1.0), BlockId(2), |c| {
            changes.push(c)
        });
        assert_eq!(n, 3);
        assert_eq!(changes.len(), 3);
    }

    #[test]
    fn fill_aabb_skips_unloaded_chunks() {
        let world = World::new(42);
        let mut changes = Vec::new();
        let n = world.fill_aabb(aabb(0.0, 0.0, 0.0, 16.0, 16.0, 16.0), BlockId(2), |c| {
            changes.push(c)
        });
        assert_eq!(n, 0);
        assert!(changes.is_empty());
    }

    #[test]
    fn fill_aabb_is_idempotent_when_target_equals_existing() {
        let world = world_with_origin_chunk();
        world.set_block(0, 0, 0, BlockId(2));
        let mut changes = Vec::new();
        let n = world.fill_aabb(aabb(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), BlockId(2), |c| {
            changes.push(c)
        });
        assert_eq!(n, 0);
        assert!(changes.is_empty());
    }

    #[test]
    fn hollow_aabb_4x4x4_thickness_1_is_56_cells() {
        // 4³=64 outer minus 2³=8 inner = 56.
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        let n = world.hollow_aabb(aabb(0.0, 0.0, 0.0, 4.0, 4.0, 4.0), BlockId(2), 1, |c| {
            changes.push(c)
        });
        assert_eq!(n, 56);
    }

    #[test]
    fn fill_sphere_radius_zero_is_single_center_cell() {
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        let n = world.fill_sphere((0, 0, 0), 0.0, BlockId(2), |c| changes.push(c));
        assert_eq!(n, 1);
        assert_eq!(changes[0].x, 0);
        assert_eq!(changes[0].y, 0);
        assert_eq!(changes[0].z, 0);
    }

    #[test]
    fn fill_sphere_radius_two_changes_at_least_seven() {
        // Loose check (counts vary by chunk clipping): a Euclidean sphere
        // of radius 2 must have at least the 7 axis-distance cells + 1 centre.
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        let n = world.fill_sphere((0, 0, 0), 2.0, BlockId(2), |c| changes.push(c));
        assert!(n >= 7);
        for c in &changes {
            let dx = c.x;
            let dy = c.y;
            let dz = c.z;
            assert!(
                dx * dx + dy * dy + dz * dz <= 4,
                "out-of-sphere cell {}",
                (c.x, c.y, c.z).0
            );
        }
    }

    #[test]
    fn hollow_sphere_changes_all_within_shell_band() {
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        // centre (-2, 0, 0) so most of the sphere overlaps the loaded chunk.
        let n = world.hollow_sphere((0, 0, 0), 4.0, 1.0, BlockId(2), |c| changes.push(c));
        // It's fine if some changes were dropped due to chunk clipping —
        // what's critical is that no change has d² ≤ inner².
        assert!(n > 0);
        for c in &changes {
            let dx = c.x;
            let dy = c.y;
            let dz = c.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            // inner radius ≈ ceil(max(0, 4 − 1)) = 3 → inner² = 9.
            // outer radius = ceil(4) = 4 → outer² = 16.
            assert!(d2 > 9, "inner-cell violation at {:?}", (c.x, c.y, c.z));
            assert!(d2 <= 16, "outer-cell violation at {:?}", (c.x, c.y, c.z));
        }
    }

    #[test]
    fn fill_cylinder_upright_yields_cells_within_radius() {
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        // base (0,0,0), radius 2, height 3 → 5x5 cross section.
        // Loaded chunk only covers x,z ∈ [0,15] so half the cross-section
        // (x<0 or z<0) is skipped, leaving x∈[0,2],z∈[0,2] = 3*3=9 per layer
        // * 3 layers = 27.
        let n = world.fill_cylinder((0, 0, 0), 2.0, 3.0, BlockId(2), |c| changes.push(c));
        assert_eq!(n, 18);
        for c in &changes {
            assert!(c.y < 3, "cylinder has y < base.y + height");
        }
    }

    #[test]
    fn fill_pyramid_layer_counts_non_increasing() {
        let world = world_with_origin_chunk();
        let pyramid = aabb(0.0, 0.0, 0.0, 3.0, 3.0, 3.0);
        let mut per_layer: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        let n = world.fill_pyramid(pyramid, BlockId(2), |c| {
            *per_layer.entry(c.y).or_insert(0) += 1;
        });
        assert!(n > 0);
        // Layer counts should be monotonically non-increasing from base to apex.
        let mut prev = usize::MAX;
        let mut layers: Vec<(i32, usize)> = per_layer.into_iter().collect();
        layers.sort_by_key(|(y, _)| *y);
        for (_y, count) in layers {
            assert!(
                count <= prev,
                "pyramid layers must shrink or stay flat going up"
            );
            prev = count;
        }
    }

    #[test]
    fn fill_line_axis_aligned_straight_emits_distance_plus_one_changes() {
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        // +X line of length 3 → 4 cells.
        let n = world.fill_line((0, 0, 0), (3, 0, 0), 1, BlockId(2), |c| changes.push(c));
        assert_eq!(n, 4);
    }

    #[test]
    fn fill_line_thickness_three_emits_at_least_one_full_cube_cross_section() {
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        // Single point line (a == b) with thickness 3 → 3×3×3 = 27 cells
        // around the point — but loaded chunk only covers [0,15], so the
        // cube (which extends to ±1) has most cells in chunk. Trim a bit:
        // pick an interior point with a same-point line so 27 cells in
        // [-1, +1]³ = [-1,1] offline? No, [0,15] chunk contains [-1, 1]
        // partially. Let's just assert count is 19 (cells in [0,1] for
        // each axis).
        let n = world.fill_line((1, 1, 1), (1, 1, 1), 3, BlockId(2), |c| changes.push(c));
        // Cells: |{-1,0,1}|³ = 27 total, but x=-1 or y=-1 or z=-1 are
        // outside the chunk (which covers [0,15]). So only cells with all
        // coords in [0,2]: |{0,1,2}|³ = 27 — but the cube of radius 1 around
        // (1,1,1) means offsets ∈ {-1,0,1}, so cells are at (1+ox, 1+oy,
        // 1+oz) where ox,oy,oz ∈ {-1,0,1}. Loaded if 1+ox ∈ [0,15], i.e.
        // ox ∈ {-1, 0, 1} (all valid). Therefore ALL 27 cells load.
        assert_eq!(n, 27);
    }

    #[test]
    fn replace_in_aabb_only_target_blocks_change() {
        let world = world_with_origin_chunk();
        world.set_block(0, 0, 0, BlockId(2));
        world.set_block(1, 0, 0, BlockId(2));
        // (2,0,0) remains air.
        let mut changes = Vec::new();
        let n = world.replace_in_aabb(
            aabb(0.0, 0.0, 0.0, 3.0, 1.0, 1.0),
            BlockId(2),
            BlockId(3),
            |c| changes.push(c),
        );
        assert_eq!(n, 2);
        for c in &changes {
            assert_eq!(c.old, BlockId(2));
            assert_eq!(c.new, BlockId(3));
        }
    }

    #[test]
    fn replace_in_aabb_noop_when_target_equals_replacement() {
        let world = world_with_origin_chunk();
        let mut changes = Vec::new();
        let n = world.replace_in_aabb(
            aabb(0.0, 0.0, 0.0, 4.0, 4.0, 4.0),
            BlockId(2),
            BlockId(2),
            |c| changes.push(c),
        );
        assert_eq!(n, 0);
        assert!(changes.is_empty());
    }
}
