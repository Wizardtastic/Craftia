//! Item pickup system: handles picking up item entities.
//!
//! This system:
//! 1. Checks for item entities near the player
//! 2. If within vacuum range and pickup delay is 0, inserts the item into
//!    the player's [`SurvivalInventory`]
//! 3. If the inventory is full, the item stays on the ground

use glam::Vec3;
use voxel_ecs::{Entity, World};

use crate::components::{PlayerEntity, Transform};
use crate::inventory::SurvivalInventory;
use crate::item_entity::ItemEntity;

/// Vacuum range in blocks (items within this range are attracted to the player).
const VACUUM_RANGE: f32 = 1.5;
/// Pickup range in blocks (items within this range are picked up).
const PICKUP_RANGE: f32 = 0.8;

/// Item pickup system entry point. Called each fixed timestep.
pub fn item_pickup_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let player_pos = match world.get::<Transform>(player_entity) {
        Some(t) => t.pos,
        None => return,
    };

    // Collect all item entities and their positions.
    let items: Vec<(Entity, ItemEntity, Vec3)> = {
        let mut result = Vec::new();
        let entities: Vec<Entity> = world.query::<&ItemEntity>().map(|(e, _)| e).collect();
        for entity in entities {
            if let (Some(item), Some(transform)) = (
                world.get::<ItemEntity>(entity),
                world.get::<Transform>(entity),
            ) {
                result.push((entity, *item, transform.pos));
            }
        }
        result
    };

    // Process each item.
    let mut to_pickup = Vec::new();
    let mut to_attract = Vec::new();

    for (entity, item, item_pos) in items {
        let distance = (item_pos - player_pos).length();

        // Check if within pickup range and can be picked up.
        if distance <= PICKUP_RANGE && item.can_pickup() {
            to_pickup.push((entity, item));
        }
        // Check if within vacuum range (attract towards player).
        else if distance <= VACUUM_RANGE && item.can_pickup() {
            to_attract.push((entity, item_pos));
        }
    }

    // Attract items towards player.
    for (entity, item_pos) in to_attract {
        let direction = (player_pos - item_pos).normalize_or_zero();
        let speed = 5.0; // Attraction speed.
        let new_pos = item_pos + direction * speed * _dt;

        if let Some(mut transform) = world.get::<Transform>(entity).copied() {
            transform.pos = new_pos;
            world.set(entity, transform);
        }
    }

    // Pick up items.
    for (entity, item) in to_pickup {
        // Try to add to the player's inventory.
        if let Some(inventory) = world.get_mut::<SurvivalInventory>(player_entity) {
            let block_id = item.block_id();
            let stack = crate::items::ItemStack::single(block_id);

            // Merge into an existing stack or take the first empty slot;
            // if nothing fits, the item stays on the ground.
            let remainder = inventory.insert(stack);
            if remainder.is_some() {
                continue;
            }
        }

        // Remove the item entity.
        world.despawn(entity);
    }
}
