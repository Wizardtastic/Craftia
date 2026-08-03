//! Gradient paint tool: paint blocks with a gradient blend between two blocks.
//!
//! Supports linear, radial, and radial-Y gradient shapes with
//! linear, smoothstep, and cosine interpolation.

use glam::IVec3;
use std::sync::Arc;
use voxel_core::BlockId;
use voxel_world::World;

use super::BrushShape;

/// Gradient shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientShape {
    LinearX,
    LinearY,
    LinearZ,
    Radial,
    RadialY,
}

impl GradientShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::LinearX => "Linear X",
            Self::LinearY => "Linear Y",
            Self::LinearZ => "Linear Z",
            Self::Radial => "Radial",
            Self::RadialY => "Radial Y",
        }
    }

    pub const ALL: &[GradientShape] = &[
        Self::LinearX,
        Self::LinearY,
        Self::LinearZ,
        Self::Radial,
        Self::RadialY,
    ];
}

/// Interpolation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    Linear,
    Smoothstep,
    Cosine,
}

impl InterpolationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Smoothstep => "Smoothstep",
            Self::Cosine => "Cosine",
        }
    }

    pub const ALL: &[InterpolationMode] = &[Self::Linear, Self::Smoothstep, Self::Cosine];
}

/// Paint tool state.
#[derive(Clone, Debug, PartialEq)]
pub struct PaintTool {
    pub block_a: BlockId,
    pub block_b: BlockId,
    pub gradient: GradientShape,
    pub radius: f32,
    pub shape: BrushShape,
    pub interpolation: InterpolationMode,
}

impl Default for PaintTool {
    fn default() -> Self {
        Self {
            block_a: BlockId::new(1), // stone
            block_b: BlockId::new(2), // dirt
            gradient: GradientShape::Radial,
            radius: 5.0,
            shape: BrushShape::Sphere,
            interpolation: InterpolationMode::Smoothstep,
        }
    }
}

/// Compute gradient t value (0.0..1.0) for a position relative to center and radius.
fn gradient_t(pos: IVec3, center: IVec3, radius: f32, shape: GradientShape) -> f32 {
    let d = (pos - center).as_vec3();
    match shape {
        GradientShape::LinearX => {
            let r = radius.max(1.0);
            ((d.x / r) * 0.5 + 0.5).clamp(0.0, 1.0)
        }
        GradientShape::LinearY => {
            let r = radius.max(1.0);
            ((d.y / r) * 0.5 + 0.5).clamp(0.0, 1.0)
        }
        GradientShape::LinearZ => {
            let r = radius.max(1.0);
            ((d.z / r) * 0.5 + 0.5).clamp(0.0, 1.0)
        }
        GradientShape::Radial => {
            let dist = d.length();
            (dist / radius).clamp(0.0, 1.0)
        }
        GradientShape::RadialY => {
            let dist = (d.x * d.x + d.z * d.z).sqrt();
            (dist / radius).clamp(0.0, 1.0)
        }
    }
}

/// Apply interpolation to t value.
fn interpolate(t: f32, mode: InterpolationMode) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match mode {
        InterpolationMode::Linear => t,
        InterpolationMode::Smoothstep => t * t * (3.0 - 2.0 * t),
        InterpolationMode::Cosine => (1.0 - (t * std::f32::consts::PI).cos()) * 0.5,
    }
}

/// Apply gradient paint.
pub fn apply_gradient(
    tool: &PaintTool,
    world: &Arc<World>,
    center: IVec3,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let r = tool.radius.ceil() as i32;

    undo_redo.begin_batch("Gradient Paint");

    match tool.shape {
        BrushShape::Sphere => {
            for y in (center.y - r)..=(center.y + r) {
                for z in (center.z - r)..=(center.z + r) {
                    for x in (center.x - r)..=(center.x + r) {
                        let pos = IVec3::new(x, y, z);
                        let d = pos - center;
                        if d.x * d.x + d.y * d.y + d.z * d.z > r * r {
                            continue;
                        }
                        apply_at(pos, tool, center, world, undo_redo);
                    }
                }
            }
        }
        BrushShape::Cylinder => {
            for y in center.y..=(center.y + r) {
                for z in (center.z - r)..=(center.z + r) {
                    for x in (center.x - r)..=(center.x + r) {
                        let pos = IVec3::new(x, y, z);
                        let dx = x - center.x;
                        let dz = z - center.z;
                        if dx * dx + dz * dz > r * r {
                            continue;
                        }
                        apply_at(pos, tool, center, world, undo_redo);
                    }
                }
            }
        }
        BrushShape::Box => {
            for y in (center.y - r)..=(center.y + r) {
                for z in (center.z - r)..=(center.z + r) {
                    for x in (center.x - r)..=(center.x + r) {
                        let pos = IVec3::new(x, y, z);
                        apply_at(pos, tool, center, world, undo_redo);
                    }
                }
            }
        }
    }

    undo_redo.commit_batch();
    super::brush::affected_chunks_range(center - IVec3::splat(r), center + IVec3::splat(r))
}

fn apply_at(
    pos: IVec3,
    tool: &PaintTool,
    center: IVec3,
    world: &Arc<World>,
    undo_redo: &mut voxel_game::UndoRedoState,
) {
    let t_raw = gradient_t(pos, center, tool.radius, tool.gradient);
    let t = interpolate(t_raw, tool.interpolation);

    let block = if t < 0.5 { tool.block_a } else { tool.block_b };
    let old = world.get_block(pos.x, pos.y, pos.z);
    if old != block
        && world.set_block(pos.x, pos.y, pos.z, block) {
            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                x: pos.x,
                y: pos.y,
                z: pos.z,
                old_block: old.0,
                new_block: block.0,
            });
        }
}
