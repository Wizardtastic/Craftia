//! Filter stack: apply procedural filters to a selection or brush area.
//!
//! Filters: noise threshold, erode, dilate, smooth.

use glam::IVec3;
use std::sync::Arc;
use voxel_core::BlockId;
use voxel_world::World;

/// A single filter operation.
#[allow(dead_code)] // Noise is constructed in UI; Erode/Dilate/Smooth are Phase 6 infrastructure
#[derive(Clone, Debug, PartialEq)]
pub enum FilterOp {
    /// Replace blocks based on noise threshold.
    Noise {
        scale: f32,
        threshold: f32,
        block_a: BlockId,
        block_b: BlockId,
        seed: i32,
    },
    /// Remove blocks that have <= N solid neighbors.
    Erode {
        neighbor_threshold: u8,
        iterations: u32,
    },
    /// Fill air blocks that have >= N solid neighbors.
    Dilate {
        neighbor_threshold: u8,
        iterations: u32,
        block: BlockId,
    },
    /// Replace each block with the majority block type among its neighbors.
    Smooth { iterations: u32, radius: u8 },
}

impl FilterOp {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Noise { .. } => "Noise",
            Self::Erode { .. } => "Erode",
            Self::Dilate { .. } => "Dilate",
            Self::Smooth { .. } => "Smooth",
        }
    }
}

/// A stack of filters to apply in sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterStack {
    pub filters: Vec<FilterOp>,
    pub apply_to_selection: bool,
}

impl Default for FilterStack {
    fn default() -> Self {
        Self {
            filters: Vec::new(),
            apply_to_selection: true,
        }
    }
}

impl FilterStack {
    pub fn add(&mut self, op: FilterOp) {
        self.filters.push(op);
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.filters.len() {
            self.filters.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.filters.clear();
    }
}

/// Count solid neighbors (6-connected) at a position.
fn solid_neighbors(world: &World, x: i32, y: i32, z: i32) -> u8 {
    let mut count = 0;
    for (dx, dy, dz) in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ] {
        let b = world.get_block(x + dx, y + dy, z + dz);
        if !b.is_air() {
            count += 1;
        }
    }
    count
}

/// Simple hash-based noise.
fn noise_at(x: i32, y: i32, z: i32, scale: f32, seed: i32) -> f32 {
    let fx = x as f32 / scale + seed as f32;
    let fy = y as f32 / scale + seed as f32 * 0.7;
    let fz = z as f32 / scale + seed as f32 * 1.3;
    (fx.sin() * 43758.5453 + fy.cos() * 12345.6789 + fz.sin() * 67890.1234).fract()
}

/// Get the majority block among neighbors (excluding air).
fn majority_neighbor(world: &World, x: i32, y: i32, z: i32, radius: u8) -> BlockId {
    let mut counts = std::collections::HashMap::new();
    let r = radius as i32;
    for dy in -r..=r {
        for dz in -r..=r {
            for dx in -r..=r {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let b = world.get_block(x + dx, y + dy, z + dz);
                if !b.is_air() {
                    *counts.entry(b).or_insert(0u32) += 1;
                }
            }
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(b, _)| b)
        .unwrap_or(BlockId::AIR)
}

/// Apply a single filter to a region.
fn apply_filter(
    op: &FilterOp,
    world: &Arc<World>,
    min: IVec3,
    max: IVec3,
    undo_redo: &mut voxel_game::UndoRedoState,
) {
    match op {
        FilterOp::Noise {
            scale,
            threshold,
            block_a,
            block_b,
            seed,
        } => {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    for x in min.x..=max.x {
                        let n = noise_at(x, y, z, *scale, *seed);
                        let block = if n > *threshold { *block_b } else { *block_a };
                        let old = world.get_block(x, y, z);
                        if old != block {
                            if world.set_block(x, y, z, block) {
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
            }
        }
        FilterOp::Erode {
            neighbor_threshold,
            iterations,
        } => {
            for _ in 0..*iterations {
                let mut to_remove = Vec::new();
                for y in min.y..=max.y {
                    for z in min.z..=max.z {
                        for x in min.x..=max.x {
                            let block = world.get_block(x, y, z);
                            if !block.is_air() {
                                let n = solid_neighbors(world, x, y, z);
                                if n <= *neighbor_threshold {
                                    to_remove.push((x, y, z, block));
                                }
                            }
                        }
                    }
                }
                for (x, y, z, old) in to_remove {
                    if world.set_block(x, y, z, BlockId::AIR) {
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
        FilterOp::Dilate {
            neighbor_threshold,
            iterations,
            block,
        } => {
            for _ in 0..*iterations {
                let mut to_fill = Vec::new();
                for y in min.y..=max.y {
                    for z in min.z..=max.z {
                        for x in min.x..=max.x {
                            let b = world.get_block(x, y, z);
                            if b.is_air() {
                                let n = solid_neighbors(world, x, y, z);
                                if n >= *neighbor_threshold {
                                    to_fill.push((x, y, z));
                                }
                            }
                        }
                    }
                }
                for (x, y, z) in to_fill {
                    let old = world.get_block(x, y, z);
                    if world.set_block(x, y, z, *block) {
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
        FilterOp::Smooth { iterations, radius } => {
            for _ in 0..*iterations {
                let mut replacements = Vec::new();
                for y in min.y..=max.y {
                    for z in min.z..=max.z {
                        for x in min.x..=max.x {
                            let old = world.get_block(x, y, z);
                            if !old.is_air() {
                                let majority = majority_neighbor(world, x, y, z, *radius);
                                if majority != BlockId::AIR && majority != old {
                                    replacements.push((x, y, z, old, majority));
                                }
                            }
                        }
                    }
                }
                for (x, y, z, old, new) in replacements {
                    if world.set_block(x, y, z, new) {
                        let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                            x,
                            y,
                            z,
                            old_block: old.0,
                            new_block: new.0,
                        });
                    }
                }
            }
        }
    }
}

/// Apply the entire filter stack to a region.
pub fn apply_filters(
    stack: &FilterStack,
    world: &Arc<World>,
    min: IVec3,
    max: IVec3,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    undo_redo.begin_batch("Filter Stack");
    for op in &stack.filters {
        apply_filter(op, world, min, max, undo_redo);
    }
    undo_redo.commit_batch();
    super::brush::affected_chunks_range(min, max)
}
