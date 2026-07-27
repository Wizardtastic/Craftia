//! Brush application: applies brush volumes to the world, generates
//! wireframe previews, and handles block picking.

use glam::IVec3;
use std::sync::Arc;

use voxel_render::overlay::{OverlayData, OverlayLine};
use voxel_world::World;

use super::{BrushShape, EditState};
use voxel_core::BlockId;

/// Generate wireframe lines for the brush preview.
pub fn brush_wireframe(center: IVec3, shape: BrushShape, radius: f32, valid: bool) -> OverlayData {
    let color = if valid {
        [80, 255, 80, 200]
    } else {
        [255, 80, 80, 200]
    };
    let lines = match shape {
        BrushShape::Box => {
            let half = radius as i32;
            let min = center - IVec3::splat(half);
            let max = center + IVec3::splat(half) + IVec3::splat(1);
            cube_wireframe(min, max, color)
        }
        BrushShape::Sphere => {
            let r = radius.ceil() as i32;
            let min = center - IVec3::splat(r);
            let max = center + IVec3::splat(r) + IVec3::splat(1);
            cube_wireframe(min, max, color)
        }
        BrushShape::Cylinder => {
            let r = radius.ceil() as i32;
            let h = (radius * 2.0).ceil() as i32;
            let min = center - IVec3::new(r, 0, r);
            let max = center + IVec3::new(r, h - 1, r) + IVec3::splat(1);
            cube_wireframe(min, max, color)
        }
    };
    OverlayData { lines }
}

pub fn cube_wireframe(min: IVec3, max: IVec3, color: [u8; 4]) -> Vec<OverlayLine> {
    let (x0, y0, z0) = (min.x as f32, min.y as f32, min.z as f32);
    let (x1, y1, z1) = (max.x as f32, max.y as f32, max.z as f32);
    let corners = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y0, z1],
        [x0, y0, z1],
        [x0, y1, z0],
        [x1, y1, z0],
        [x1, y1, z1],
        [x0, y1, z1],
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    edges
        .iter()
        .map(|&(i, j)| OverlayLine {
            a: corners[i],
            b: corners[j],
            color,
        })
        .collect()
}

/// Apply the brush at the given block-space center.
///
/// Supports: replace mode, hollow mode, surface-only mode, multi-block palette.
pub fn apply_brush(
    edit: &mut EditState,
    world: &Arc<World>,
    center: IVec3,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let brush = match edit.brush_ref() {
        Some(b) => b.clone(),
        None => return Vec::new(),
    };

    let radius_i = brush.radius.ceil() as i32;

    undo_redo.begin_batch("Brush".to_string());

    let mut record = |change: voxel_world::volume::BlockChange| {
        let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
            x: change.x,
            y: change.y,
            z: change.z,
            old_block: change.old.0,
            new_block: change.new.0,
        });
    };

    let center_tuple = (center.x, center.y, center.z);

    // Determine the block to place: either from palette or single block.
    let pick_block = || -> BlockId {
        if brush.palette.enabled && !brush.palette.entries.is_empty() {
            brush.palette.pick()
        } else {
            brush.block
        }
    };

    if brush.replace {
        // Replace mode: only overwrite blocks matching target.
        match brush.shape {
            BrushShape::Sphere => {
                world.fill_sphere(center_tuple, brush.radius, pick_block(), |change| {
                    let should_replace = match brush.target {
                        Some(t) => change.old == t,
                        None => !change.old.is_air(),
                    };
                    if should_replace {
                        record(change);
                    }
                });
            }
            BrushShape::Cylinder => {
                let base = (center.x, center.y - radius_i, center.z);
                let height = (radius_i * 2) as f32;
                world.fill_cylinder(base, brush.radius, height, pick_block(), |change| {
                    let should_replace = match brush.target {
                        Some(t) => change.old == t,
                        None => !change.old.is_air(),
                    };
                    if should_replace {
                        record(change);
                    }
                });
            }
            BrushShape::Box => {
                let min = center - IVec3::splat(radius_i);
                let max = center + IVec3::splat(radius_i) + IVec3::splat(1);
                let bounds = voxel_core::Aabb::new(min.as_vec3(), max.as_vec3());
                world.fill_aabb(bounds, pick_block(), |change| {
                    let should_replace = match brush.target {
                        Some(t) => change.old == t,
                        None => !change.old.is_air(),
                    };
                    if should_replace {
                        record(change);
                    }
                });
            }
        }
    } else if brush.hollow {
        // Hollow mode: use hollow volume methods.
        let block = pick_block();
        match brush.shape {
            BrushShape::Sphere => {
                let shell = brush.radius.max(1.0);
                world.hollow_sphere(center_tuple, brush.radius, shell, block, &mut record);
            }
            BrushShape::Cylinder => {
                // Approximate: fill cylinder then carve interior.
                let base = (center.x, center.y - radius_i, center.z);
                let height = (radius_i * 2) as f32;
                let inner_r = (brush.radius - 1.0).max(0.0);
                world.fill_cylinder(base, brush.radius, height, block, &mut record);
                if inner_r > 0.0 {
                    world.fill_cylinder(
                        base,
                        inner_r,
                        height,
                        voxel_core::BlockId::AIR,
                        &mut record,
                    );
                }
            }
            BrushShape::Box => {
                let min = center - IVec3::splat(radius_i);
                let max = center + IVec3::splat(radius_i) + IVec3::splat(1);
                let bounds = voxel_core::Aabb::new(min.as_vec3(), max.as_vec3());
                world.hollow_aabb(bounds, block, 1, &mut record);
            }
        }
    } else {
        // Normal fill mode.
        let block = pick_block();
        if brush.surface_only {
            // Surface only: fill volume, but only overwrite air blocks.
            match brush.shape {
                BrushShape::Sphere => {
                    world.fill_sphere(center_tuple, brush.radius, block, |change| {
                        if change.old.is_air() {
                            record(change);
                        }
                    });
                }
                BrushShape::Cylinder => {
                    let base = (center.x, center.y - radius_i, center.z);
                    let height = (radius_i * 2) as f32;
                    world.fill_cylinder(base, brush.radius, height, block, |change| {
                        if change.old.is_air() {
                            record(change);
                        }
                    });
                }
                BrushShape::Box => {
                    let min = center - IVec3::splat(radius_i);
                    let max = center + IVec3::splat(radius_i) + IVec3::splat(1);
                    let bounds = voxel_core::Aabb::new(min.as_vec3(), max.as_vec3());
                    world.fill_aabb(bounds, block, |change| {
                        if change.old.is_air() {
                            record(change);
                        }
                    });
                }
            }
        } else {
            // Standard fill.
            match brush.shape {
                BrushShape::Sphere => {
                    world.fill_sphere(center_tuple, brush.radius, block, record);
                }
                BrushShape::Cylinder => {
                    let base = (center.x, center.y - radius_i, center.z);
                    let height = (radius_i * 2) as f32;
                    world.fill_cylinder(base, brush.radius, height, block, record);
                }
                BrushShape::Box => {
                    let min = center - IVec3::splat(radius_i);
                    let max = center + IVec3::splat(radius_i) + IVec3::splat(1);
                    let bounds = voxel_core::Aabb::new(min.as_vec3(), max.as_vec3());
                    world.fill_aabb(bounds, block, record);
                }
            }
        }
    }

    undo_redo.commit_batch();
    let placed = pick_block();
    edit.add_recent(placed);
    affected_chunks(center, radius_i)
}

/// Pick a block from the world and set it as the brush block.
pub fn pick_brush_block(edit: &mut EditState, world: &World, hit: IVec3) {
    let block = world.get_block(hit.x, hit.y, hit.z);
    if !block.is_air() {
        if let Some(brush) = edit.brush_mut() {
            brush.block = block;
        }
        edit.add_recent(block);
    }
}

/// Pick a block from the world and set it as the replace target.
pub fn pick_replace_target(edit: &mut EditState, world: &World, hit: IVec3) {
    let block = world.get_block(hit.x, hit.y, hit.z);
    if !block.is_air() {
        if let Some(brush) = edit.brush_mut() {
            brush.target = Some(block);
        }
    }
}

fn affected_chunks(center: IVec3, radius: i32) -> Vec<voxel_core::math::ChunkPos> {
    let min_block = center - IVec3::splat(radius);
    let max_block = center + IVec3::splat(radius);
    affected_chunks_range(min_block, max_block)
}

/// Compute affected chunks from a min/max block range. Public for terrain/paint/filter use.
pub fn affected_chunks_range(min: IVec3, max: IVec3) -> Vec<voxel_core::math::ChunkPos> {
    let min_chunk = voxel_core::math::block_to_chunk(min);
    let max_chunk = voxel_core::math::block_to_chunk(max);
    let mut chunks = Vec::new();
    for cx in min_chunk.x()..=max_chunk.x() {
        for cy in min_chunk.y()..=max_chunk.y() {
            for cz in min_chunk.z()..=max_chunk.z() {
                chunks.push(voxel_core::math::ChunkPos::new(cx, cy, cz));
            }
        }
    }
    chunks
}
