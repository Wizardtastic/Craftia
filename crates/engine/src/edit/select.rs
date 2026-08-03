//! Selection tool: box select with drag, wireframe overlay, clipboard ops.

use glam::IVec3;
use voxel_core::BlockId;
use voxel_render::overlay::{OverlayData, OverlayLine};
use voxel_world::World;

/// Selection tool state.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SelectTool {
    /// First corner (set on drag start).
    pub corner_a: Option<IVec3>,
    /// Second corner (updated during drag, finalized on release).
    pub corner_b: Option<IVec3>,
    /// Whether the mouse is currently being dragged.
    pub dragging: bool,
    /// Confirmed selection (set after drag release, cleared on new drag or Escape).
    pub active_selection: Option<(IVec3, IVec3)>, // min, max
}

impl SelectTool {
    /// Start a new drag at the given block position.
    pub fn start_drag(&mut self, pos: IVec3) {
        self.corner_a = Some(pos);
        self.corner_b = Some(pos);
        self.dragging = true;
        self.active_selection = None;
    }

    /// Update the drag endpoint.
    pub fn update_drag(&mut self, pos: IVec3) {
        if self.dragging {
            self.corner_b = Some(pos);
        }
    }

    /// Finalize the selection on mouse release.
    pub fn end_drag(&mut self) {
        if self.dragging {
            self.dragging = false;
            if let (Some(a), Some(b)) = (self.corner_a, self.corner_b) {
                let min = a.min(b);
                let max = a.max(b);
                if min != max {
                    self.active_selection = Some((min, max));
                }
            }
            self.corner_a = None;
            self.corner_b = None;
        }
    }

    /// Clear the active selection.
    pub fn clear(&mut self) {
        self.corner_a = None;
        self.corner_b = None;
        self.dragging = false;
        self.active_selection = None;
    }

    /// Get the current selection bounds (min, max), whether dragging or finalized.
    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        if let (Some(a), Some(b)) = (self.corner_a, self.corner_b) {
            Some((a.min(b), a.max(b)))
        } else {
            self.active_selection
        }
    }

    /// Number of blocks in the selection.
    pub fn block_count(&self) -> usize {
        if let Some((min, max)) = self.bounds() {
            let size = max - min + IVec3::splat(1);
            (size.x.max(0) as usize) * (size.y.max(0) as usize) * (size.z.max(0) as usize)
        } else {
            0
        }
    }

    /// Selection dimensions as a string like "5 x 3 x 7".
    pub fn dimensions_str(&self) -> String {
        if let Some((min, max)) = self.bounds() {
            let s = max - min + IVec3::splat(1);
            format!("{} x {} x {}", s.x, s.y, s.z)
        } else {
            "--".to_string()
        }
    }
}

/// Generate wireframe overlay for the selection.
pub fn selection_wireframe(sel: &SelectTool) -> OverlayData {
    let mut lines = Vec::new();

    // Drag preview (yellow, dashed feel via thinner lines).
    if sel.dragging {
        if let (Some(a), Some(b)) = (sel.corner_a, sel.corner_b) {
            let min = a.min(b);
            let max = a.max(b) + IVec3::splat(1);
            lines.extend(cube_wireframe(min, max, [200, 200, 80, 180]));
        }
    }

    // Finalized selection (blue).
    if let Some((min, max)) = sel.active_selection {
        let max_exclusive = max + IVec3::splat(1);
        lines.extend(cube_wireframe(min, max_exclusive, [80, 200, 255, 220]));
    }

    OverlayData { lines }
}

fn cube_wireframe(min: IVec3, max: IVec3, color: [u8; 4]) -> Vec<OverlayLine> {
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

/// Copy blocks from selection into a flat Vec<BlockId> with corners.
pub fn copy_selection(world: &World, min: IVec3, max: IVec3) -> crate::Clipboard {
    let mut blocks = Vec::new();
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                blocks.push(world.get_block(x, y, z));
            }
        }
    }
    ((min.x, min.y, min.z), (max.x, max.y, max.z), blocks)
}

/// Delete blocks in selection (fill with air). Returns affected chunks.
pub fn delete_selection(
    world: &std::sync::Arc<World>,
    min: IVec3,
    max: IVec3,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    undo_redo.begin_batch("Delete Selection");
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                let old = world.get_block(x, y, z);
                if !old.is_air() && world.set_block(x, y, z, BlockId::AIR) {
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
    affected_chunks(min, max)
}

fn affected_chunks(min: IVec3, max: IVec3) -> Vec<voxel_core::math::ChunkPos> {
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
