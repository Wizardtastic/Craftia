//! Held item system: bridges the engine's Hotbar state → ECS HeldBlock
//! component on the player entity. Runs each tick so the renderer can
//! read HeldBlock.tile for first-person held item rendering.

use voxel_core::BlockId;
use voxel_ecs::World;

use crate::components::{HeldBlock, PlayerEntity};

/// Resource: hotbar slot → tile index mapping. Inserted by the engine
/// each frame before running the schedule.
#[derive(Clone, Copy, Debug, Default)]
pub struct HotbarResource {
    /// Atlas tile index of the currently selected hotbar slot (0 = air).
    pub tile: u32,
    /// BlockId of the currently selected hotbar slot.
    pub selected_block: BlockId,
    /// Tier of the currently selected tool. Zero means hand/no tool.
    /// The hotbar currently stores block IDs, so the engine supplies this
    /// separately until item/tool definitions are fully data-driven.
    pub selected_tool_tier: u8,
}

/// System: writes HeldBlock.tile on the player entity from HotbarResource.
pub fn held_item_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let tile = world
        .resource::<HotbarResource>()
        .map(|h| h.tile)
        .unwrap_or(0);

    if let Some(held) = world.get_mut::<HeldBlock>(player_entity) {
        held.tile = tile;
        held.in_first_person = tile != 0;
    }
}
