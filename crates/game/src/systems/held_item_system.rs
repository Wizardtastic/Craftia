//! Held item system: writes the `HeldBlock` component on the player entity
//! from the player's [`SurvivalInventory`] (selected hotbar slot + registry
//! tile lookup). Runs each tick so the renderer can read `HeldBlock.tile` for
//! first-person held item rendering.

use voxel_ecs::World;

use crate::components::{HeldBlock, PlayerEntity};
use crate::inventory::SurvivalInventory;
use crate::systems::PhysicsWorldRes;

/// System: writes `HeldBlock.tile` on the player entity from their inventory's
/// selected hotbar slot. The registry lookup uses the top face (PosY) tile.
pub fn held_item_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    // Selected block from the player's inventory (empty slot = no held item).
    let selected = world
        .get::<SurvivalInventory>(player_entity)
        .and_then(|inv| inv.selected_block());

    // Registry access for the tile index; skip if the physics world isn't
    // wired yet (nothing to look up).
    let tile = match selected {
        Some(id) if !id.is_air() => world
            .resource::<PhysicsWorldRes>()
            .map(|phys| phys.0.registry().get(id).textures.tiles[3] as u32)
            .unwrap_or(0),
        _ => 0,
    };

    if let Some(held) = world.get_mut::<HeldBlock>(player_entity) {
        held.tile = tile;
        held.in_first_person = tile != 0;
    }
}
