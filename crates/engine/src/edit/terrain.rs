//! Terrain editing tool: raise, lower, flatten, smooth, noise operations.
//!
//! Operates on heightmap (top solid block Y) within the brush volume.

use glam::IVec3;
use std::sync::Arc;
use voxel_core::BlockId;
use voxel_world::World;

use super::BrushShape;

/// Terrain operation type.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainOp {
    Raise {
        amount: f32,
    },
    Lower {
        amount: f32,
    },
    Flatten {
        target_height: Option<i32>,
    },
    Smooth {
        iterations: u32,
    },
    Noise {
        scale: f32,
        amplitude: f32,
        seed: i32,
    },
}

impl TerrainOp {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Raise { .. } => "Raise",
            Self::Lower { .. } => "Lower",
            Self::Flatten { .. } => "Flatten",
            Self::Smooth { .. } => "Smooth",
            Self::Noise { .. } => "Noise",
        }
    }
}

/// Terrain tool state.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainTool {
    pub op: TerrainOp,
    pub radius: f32,
    pub shape: BrushShape,
    pub block: BlockId,
}

impl Default for TerrainTool {
    fn default() -> Self {
        Self {
            op: TerrainOp::Raise { amount: 1.0 },
            radius: 5.0,
            shape: BrushShape::Cylinder,
            block: BlockId::new(1), // stone
        }
    }
}

/// Find the top non-air block Y in a column. Returns None if column is empty.
fn surface_y(world: &World, x: i32, z: i32) -> Option<i32> {
    for y in (0..256).rev() {
        let block = world.get_block(x, y, z);
        if !block.is_air() {
            return Some(y);
        }
    }
    None
}

/// Apply terrain raise: for each column in brush AABB, fill from surface+1 to surface+amount.
pub fn apply_raise(
    world: &Arc<World>,
    center: IVec3,
    radius: f32,
    amount: i32,
    block: BlockId,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let r = radius.ceil() as i32;
    let min = center - IVec3::new(r, 0, r);
    let max = center + IVec3::new(r, 0, r);

    undo_redo.begin_batch("Raise Terrain");

    for x in min.x..=max.x {
        for z in min.z..=max.z {
            // Check if within cylindrical radius.
            let dx = x - center.x;
            let dz = z - center.z;
            if (dx * dx + dz * dz) as f32 > radius * radius {
                continue;
            }
            let sy = match surface_y(world, x, z) {
                Some(y) => y,
                None => continue,
            };
            for y in (sy + 1)..=(sy + amount) {
                if y >= 256 {
                    break;
                }
                let old = world.get_block(x, y, z);
                if old.is_air()
                    && world.set_block(x, y, z, block) {
                        let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                            x,
                            y,
                            z,
                            old_block: old.0,
                            new_block: block.0,
                        });
                    }
            }
        }
    }

    undo_redo.commit_batch();
    let min_b = min;
    let max_b = max + IVec3::new(0, amount, 0);
    super::brush::affected_chunks_range(min_b, max_b)
}

/// Apply terrain lower: remove blocks from surface-amount+1 to surface.
pub fn apply_lower(
    world: &Arc<World>,
    center: IVec3,
    radius: f32,
    amount: i32,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let r = radius.ceil() as i32;
    let min = center - IVec3::new(r, 0, r);
    let max = center + IVec3::new(r, 0, r);

    undo_redo.begin_batch("Lower Terrain");

    for x in min.x..=max.x {
        for z in min.z..=max.z {
            let dx = x - center.x;
            let dz = z - center.z;
            if (dx * dx + dz * dz) as f32 > radius * radius {
                continue;
            }
            let sy = match surface_y(world, x, z) {
                Some(y) => y,
                None => continue,
            };
            let remove_from = (sy - amount + 1).max(0);
            for y in remove_from..=sy {
                let old = world.get_block(x, y, z);
                if !old.is_air()
                    && world.set_block(x, y, z, BlockId::AIR) {
                        let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                            x,
                            y,
                            z,
                            old_block: old.0,
                            new_block: 0,
                        });
                    }
            }
        }
    }

    undo_redo.commit_batch();
    super::brush::affected_chunks_range(min, max + IVec3::new(0, 10, 0))
}

/// Apply terrain flatten: fill or remove blocks to reach target_y.
pub fn apply_flatten(
    world: &Arc<World>,
    center: IVec3,
    radius: f32,
    target_y: i32,
    block: BlockId,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let r = radius.ceil() as i32;
    let min = center - IVec3::new(r, 0, r);
    let max = center + IVec3::new(r, 0, r);

    undo_redo.begin_batch("Flatten Terrain");

    for x in min.x..=max.x {
        for z in min.z..=max.z {
            let dx = x - center.x;
            let dz = z - center.z;
            if (dx * dx + dz * dz) as f32 > radius * radius {
                continue;
            }
            let sy = surface_y(world, x, z).unwrap_or(target_y);

            if sy < target_y {
                // Fill up to target.
                for y in (sy + 1)..=target_y {
                    if y >= 256 {
                        break;
                    }
                    let old = world.get_block(x, y, z);
                    if old.is_air()
                        && world.set_block(x, y, z, block) {
                            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                                x,
                                y,
                                z,
                                old_block: old.0,
                                new_block: block.0,
                            });
                        }
                }
            } else if sy > target_y {
                // Remove down to target.
                for y in (target_y + 1)..=sy {
                    let old = world.get_block(x, y, z);
                    if !old.is_air()
                        && world.set_block(x, y, z, BlockId::AIR) {
                            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                                x,
                                y,
                                z,
                                old_block: old.0,
                                new_block: 0,
                            });
                        }
                }
            }
        }
    }

    undo_redo.commit_batch();
    super::brush::affected_chunks_range(min, max + IVec3::new(0, target_y.max(10), 0))
}

/// Apply terrain smooth: average height with neighbors.
pub fn apply_smooth(
    world: &Arc<World>,
    center: IVec3,
    radius: f32,
    iterations: u32,
    block: BlockId,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let r = radius.ceil() as i32;
    let min = center - IVec3::new(r, 0, r);
    let max = center + IVec3::new(r, 0, r);

    undo_redo.begin_batch("Smooth Terrain");

    for _iter in 0..iterations {
        // Collect heights first (read-only pass).
        let mut heights = std::collections::HashMap::new();
        for x in min.x..=max.x {
            for z in min.z..=max.z {
                let dx = x - center.x;
                let dz = z - center.z;
                if (dx * dx + dz * dz) as f32 > radius * radius {
                    continue;
                }
                heights.insert((x, z), surface_y(world, x, z));
            }
        }

        // Apply smoothing.
        for (&(x, z), &sy) in &heights {
            let Some(sy) = sy else { continue };
            // Average with 4 neighbors.
            let mut total = sy as f32;
            let mut count = 1.0;
            for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                if let Some(Some(ny)) = heights.get(&(nx, nz)) {
                    total += *ny as f32;
                    count += 1.0;
                }
            }
            let target = (total / count).round() as i32;

            if target > sy {
                for y in (sy + 1)..=target {
                    if y >= 256 {
                        break;
                    }
                    let old = world.get_block(x, y, z);
                    if old.is_air()
                        && world.set_block(x, y, z, block) {
                            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                                x,
                                y,
                                z,
                                old_block: old.0,
                                new_block: block.0,
                            });
                        }
                }
            } else if target < sy {
                for y in (target + 1)..=sy {
                    let old = world.get_block(x, y, z);
                    if !old.is_air()
                        && world.set_block(x, y, z, BlockId::AIR) {
                            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                                x,
                                y,
                                z,
                                old_block: old.0,
                                new_block: 0,
                            });
                        }
                }
            }
        }
    }

    undo_redo.commit_batch();
    super::brush::affected_chunks_range(min, max + IVec3::new(0, 10, 0))
}

/// Parameters for the noise terrain operation.
#[derive(Clone, Copy, Debug)]
pub struct NoiseParams {
    pub radius: f32,
    pub scale: f32,
    pub amplitude: f32,
    pub seed: i32,
}

/// Apply terrain noise: add noise displacement to each column's height.
pub fn apply_noise(
    world: &Arc<World>,
    center: IVec3,
    params: NoiseParams,
    block: BlockId,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let NoiseParams {
        radius,
        scale,
        amplitude,
        seed,
    } = params;
    let r = radius.ceil() as i32;
    let min = center - IVec3::new(r, 0, r);
    let max = center + IVec3::new(r, 0, r);

    undo_redo.begin_batch("Noise Terrain");

    for x in min.x..=max.x {
        for z in min.z..=max.z {
            let dx = x - center.x;
            let dz = z - center.z;
            if (dx * dx + dz * dz) as f32 > radius * radius {
                continue;
            }
            let sy = match surface_y(world, x, z) {
                Some(y) => y,
                None => continue,
            };

            // Simple hash-based noise.
            let fx = x as f32 / scale + seed as f32;
            let fz = z as f32 / scale + seed as f32 * 0.7;
            let noise = ((fx.sin() * 43_758.547 + fz.cos() * 12_345.679).fract() - 0.5) * 2.0;
            let displacement = (noise * amplitude).round() as i32;

            let target = sy + displacement;
            if target > sy {
                for y in (sy + 1)..=target {
                    if y >= 256 {
                        break;
                    }
                    let old = world.get_block(x, y, z);
                    if old.is_air()
                        && world.set_block(x, y, z, block) {
                            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                                x,
                                y,
                                z,
                                old_block: old.0,
                                new_block: block.0,
                            });
                        }
                }
            } else if target < sy {
                for y in (target + 1)..=sy {
                    let old = world.get_block(x, y, z);
                    if !old.is_air()
                        && world.set_block(x, y, z, BlockId::AIR) {
                            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                                x,
                                y,
                                z,
                                old_block: old.0,
                                new_block: 0,
                            });
                        }
                }
            }
        }
    }

    undo_redo.commit_batch();
    super::brush::affected_chunks_range(min, max + IVec3::new(0, 20, 0))
}
