//! Brush application: applies brush volumes to the world, generates
//! wireframe previews, and handles block picking.

use glam::IVec3;
use std::sync::Arc;

use voxel_render::overlay::{OverlayData, OverlayLine};
use voxel_world::volume::BlockPredicate;
use voxel_world::World;

use super::{BrushShape, EditState};

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

    // Do not let a failed/nested batch capture these edits or commit another
    // operation's pending batch.
    if !undo_redo.begin_batch("Brush") {
        return Vec::new();
    }

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

    // Choose once per stroke so the recent-block state reflects the actual
    // block written (and does not make an independent random palette pick).
    let block = if brush.palette.enabled && !brush.palette.entries.is_empty() {
        brush.palette.pick()
    } else {
        brush.block
    };

    if brush.replace || (!brush.hollow && brush.surface_only) {
        // These modes gate the write itself; the rasterizer evaluates the
        // predicate against the current block before recording a real change.
        let predicate = if brush.replace {
            brush
                .target
                .map_or(BlockPredicate::NonAir, BlockPredicate::Equals)
        } else {
            BlockPredicate::Air
        };
        match brush.shape {
            BrushShape::Sphere => {
                world.fill_sphere_where(center_tuple, brush.radius, block, predicate, &mut record);
            }
            BrushShape::Cylinder => {
                let base = (center.x, center.y - radius_i, center.z);
                let height = (radius_i * 2) as f32;
                world.fill_cylinder_where(
                    base,
                    brush.radius,
                    height,
                    block,
                    predicate,
                    &mut record,
                );
            }
            BrushShape::Box => {
                let min = center - IVec3::splat(radius_i);
                let max = center + IVec3::splat(radius_i) + IVec3::splat(1);
                let bounds = voxel_core::Aabb::new(min.as_vec3(), max.as_vec3());
                world.fill_aabb_where(bounds, block, predicate, &mut record);
            }
        }
    } else if brush.hollow {
        // Hollow mode: use hollow volume methods.
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

    undo_redo.commit_batch();
    edit.add_recent(block);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::{BrushTool, EditModeState, EditTool};
    use voxel_core::{BlockId, ChunkPos};
    use voxel_world::Chunk;

    const CENTER: IVec3 = IVec3::new(5, 5, 5);
    const OUTPUT: BlockId = BlockId(3);

    fn world_with_brush_fixture() -> Arc<World> {
        let world = World::new(42);
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(CENTER.x, CENTER.y, CENTER.z, BlockId(2));
        chunk.set(CENTER.x + 1, CENTER.y, CENTER.z, BlockId(4));
        world.insert_chunk(ChunkPos::new(0, 0, 0), chunk);
        world
    }

    fn edit_state(
        shape: BrushShape,
        replace: bool,
        target: Option<BlockId>,
        surface_only: bool,
    ) -> EditState {
        let mut brush = BrushTool::default();
        brush.shape = shape;
        brush.radius = 1.0;
        brush.block = OUTPUT;
        brush.replace = replace;
        brush.target = target;
        brush.surface_only = surface_only;
        let mut edit = EditState::default();
        edit.mode = EditModeState::Active {
            tool: EditTool::Brush(brush),
        };
        edit
    }

    fn shape_cells(shape: BrushShape) -> Vec<IVec3> {
        let mut cells = Vec::new();
        for y in CENTER.y - 1..=CENTER.y + 1 {
            for z in CENTER.z - 1..=CENTER.z + 1 {
                for x in CENTER.x - 1..=CENTER.x + 1 {
                    let p = IVec3::new(x, y, z);
                    let included = match shape {
                        BrushShape::Sphere => p.distance_squared(CENTER) <= 1,
                        BrushShape::Cylinder => {
                            y >= CENTER.y - 1
                                && y < CENTER.y + 1
                                && (x - CENTER.x).pow(2) + (z - CENTER.z).pow(2) <= 1
                        }
                        BrushShape::Box => true,
                    };
                    if included {
                        cells.push(p);
                    }
                }
            }
        }
        cells
    }

    fn sorted_edits(edits: &[voxel_game::BlockEdit]) -> Vec<(i32, i32, i32, u16, u16)> {
        let mut result: Vec<_> = edits
            .iter()
            .map(|e| (e.x, e.y, e.z, e.old_block, e.new_block))
            .collect();
        result.sort_unstable();
        result
    }

    fn exercise_brush(
        shape: BrushShape,
        replace: bool,
        target: Option<BlockId>,
        surface_only: bool,
    ) {
        let world = world_with_brush_fixture();
        let mut edit_state = edit_state(shape, replace, target, surface_only);
        let cells = shape_cells(shape);
        let before: Vec<_> = cells
            .iter()
            .map(|p| (p, world.get_block(p.x, p.y, p.z)))
            .collect();
        let expected: Vec<_> = before
            .iter()
            .map(|(p, old)| {
                let should_write = if replace {
                    target.map_or(!old.is_air(), |target| *old == target)
                } else {
                    old.is_air()
                };
                let new = if should_write && *old != OUTPUT {
                    OUTPUT
                } else {
                    *old
                };
                ((p.x, p.y, p.z, old.0, new.0), *old, new)
            })
            .collect();
        let mut undo = voxel_game::UndoRedoState::default();

        apply_brush(&mut edit_state, &world, CENTER, &mut undo);

        let after: Vec<_> = cells
            .iter()
            .map(|p| world.get_block(p.x, p.y, p.z))
            .collect();
        for ((p, old), (_, expected_old, new)) in before.iter().zip(expected.iter()) {
            assert_eq!(*old, *expected_old);
            assert_eq!(
                world.get_block(p.x, p.y, p.z),
                *new,
                "shape {shape:?} at {p:?}"
            );
        }

        let actual_edits: Vec<_> = expected
            .iter()
            .filter_map(|(edit, old, new)| (old != new).then_some(*edit))
            .collect();
        let recorded = undo
            .peek_undo()
            .map(|a| sorted_edits(&a.edits))
            .unwrap_or_default();
        let mut actual_sorted = actual_edits.clone();
        actual_sorted.sort_unstable();
        assert_eq!(
            recorded, actual_sorted,
            "undo records must exactly match writes"
        );

        // Undo and redo must restore the exact block snapshot, not merely the
        // count of recorded changes.
        if let Some(action) = undo.pop_undo() {
            for e in action.edits.iter().rev() {
                assert!(world.set_block(e.x, e.y, e.z, BlockId(e.old_block)));
            }
        }
        let restored: Vec<_> = cells
            .iter()
            .map(|p| world.get_block(p.x, p.y, p.z))
            .collect();
        assert_eq!(
            restored,
            before.iter().map(|(_, block)| *block).collect::<Vec<_>>()
        );

        if let Some(action) = undo.pop_redo() {
            for e in &action.edits {
                assert!(world.set_block(e.x, e.y, e.z, BlockId(e.new_block)));
            }
        }
        let redone: Vec<_> = cells
            .iter()
            .map(|p| world.get_block(p.x, p.y, p.z))
            .collect();
        assert_eq!(redone, after);
    }

    #[test]
    fn replace_target_defaults_and_surface_only_gate_writes_for_every_shape() {
        for shape in [BrushShape::Sphere, BrushShape::Cylinder, BrushShape::Box] {
            // Selecting air explicitly is distinct from the no-target
            // non-air default: only air cells should be replaced.
            exercise_brush(shape, true, Some(BlockId::AIR), false);
            // Explicit target: preserve the non-target solid and replace only
            // matching blocks; no-target default replaces all non-air.
            exercise_brush(shape, true, Some(BlockId(2)), false);
            exercise_brush(shape, true, None, false);
            // Surface-only fills air without overwriting either solid id.
            exercise_brush(shape, false, None, true);
        }
    }

    #[test]
    fn replace_noop_and_unloaded_chunks_produce_no_undo_edits() {
        for shape in [BrushShape::Sphere, BrushShape::Cylinder, BrushShape::Box] {
            let world = world_with_brush_fixture();
            let mut edit = edit_state(shape, true, Some(BlockId(2)), false);
            if let Some(brush) = edit.brush_mut() {
                brush.block = BlockId(2);
            }
            let mut undo = voxel_game::UndoRedoState::default();
            apply_brush(&mut edit, &world, CENTER, &mut undo);
            assert!(!undo.can_undo(), "same-block replacement is a no-op");

            let unloaded = World::new(42);
            let mut edit = edit_state(shape, true, None, false);
            apply_brush(&mut edit, &unloaded, CENTER, &mut undo);
            assert!(!undo.is_batching());
            assert!(!undo.can_undo(), "unloaded blocks must not be recorded");
        }
    }

    #[test]
    fn replace_at_loaded_chunk_edge_records_only_actual_loaded_writes() {
        for shape in [BrushShape::Sphere, BrushShape::Cylinder, BrushShape::Box] {
            let world = World::new(42);
            let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
            chunk.set(14, CENTER.y, CENTER.z, BlockId(2));
            chunk.set(15, CENTER.y, CENTER.z, BlockId(2));
            world.insert_chunk(ChunkPos::new(0, 0, 0), chunk);

            let center = IVec3::new(15, CENTER.y, CENTER.z);
            let mut edit = edit_state(shape, true, Some(BlockId(2)), false);
            let mut undo = voxel_game::UndoRedoState::default();
            apply_brush(&mut edit, &world, center, &mut undo);

            assert_eq!(world.get_block(14, CENTER.y, CENTER.z), OUTPUT);
            assert_eq!(world.get_block(15, CENTER.y, CENTER.z), OUTPUT);
            // x=16 belongs to an unloaded neighbor chunk. Reads report air,
            // and no undo edit may claim an unloaded write.
            let action = undo.peek_undo().expect("loaded writes are undoable");
            assert_eq!(action.edits.len(), 2, "shape {shape:?}");
            assert!(action.edits.iter().all(|e| e.x < 16));
        }
    }

    #[test]
    fn brush_does_not_record_out_of_world_y_candidates() {
        for shape in [BrushShape::Sphere, BrushShape::Cylinder, BrushShape::Box] {
            let world = World::new(42);
            let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
            chunk.set(CENTER.x, 0, CENTER.z, BlockId(2));
            world.insert_chunk(ChunkPos::new(0, 0, 0), chunk);

            let center = IVec3::new(CENTER.x, 0, CENTER.z);
            let mut edit = edit_state(shape, true, Some(BlockId(2)), false);
            let mut undo = voxel_game::UndoRedoState::default();
            apply_brush(&mut edit, &world, center, &mut undo);

            assert_eq!(world.get_block(center.x, 0, center.z), OUTPUT);
            let action = undo.peek_undo().expect("center write is recorded");
            assert_eq!(action.edits.len(), 1, "shape {shape:?}");
            assert!(action.edits.iter().all(|e| e.y >= 0));
        }
    }

    #[test]
    fn palette_choice_and_recent_block_stay_in_sync() {
        let world = world_with_brush_fixture();
        let mut edit = edit_state(BrushShape::Sphere, false, None, false);
        let palette_block = BlockId(7);
        let brush = edit.brush_mut().unwrap();
        brush.palette.enabled = true;
        brush.palette.add(palette_block, 1.0);
        let mut undo = voxel_game::UndoRedoState::default();

        apply_brush(&mut edit, &world, CENTER, &mut undo);

        assert_eq!(world.get_block(CENTER.x, CENTER.y, CENTER.z), palette_block);
        assert_eq!(edit.recently_used.first(), Some(&palette_block));
        let action = undo.peek_undo().unwrap();
        assert!(action.edits.iter().all(|e| e.new_block == palette_block.0));
    }

    #[test]
    fn active_undo_batch_is_not_stolen_by_brush_application() {
        let world = world_with_brush_fixture();
        let mut edit = edit_state(BrushShape::Box, true, None, false);
        let mut undo = voxel_game::UndoRedoState::default();
        assert!(undo.begin_batch("outer"));
        assert!(undo.push_edit_batched(voxel_game::BlockEdit {
            x: 1,
            y: 1,
            z: 1,
            old_block: 0,
            new_block: 1,
        }));
        let before = world.get_block(CENTER.x, CENTER.y, CENTER.z);

        assert!(apply_brush(&mut edit, &world, CENTER, &mut undo).is_empty());
        assert_eq!(world.get_block(CENTER.x, CENTER.y, CENTER.z), before);
        assert_eq!(undo.batch_name(), Some("outer"));
        assert_eq!(undo.batch_size(), 1);
        assert_eq!(undo.commit_batch().unwrap().edits.len(), 1);
    }
}
