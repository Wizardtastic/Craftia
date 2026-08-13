//! Item pickup system: handles picking up item entities.
//!
//! This system:
//! 1. Checks for item entities near the player
//! 2. If within vacuum range and pickup delay is 0, try to merge into inventory
//! 3. For now, adds to the hotbar (simplified inventory)

use glam::Vec3;
use voxel_ecs::World;

use crate::components::{PlayerEntity, Transform};
use crate::inv::Hotbar;
use crate::item_entity::ItemEntity;
use voxel_core::BlockId;

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
        // Try to add to hotbar.
        if let Some(hotbar) = world.resource_mut::<Hotbar>() {
            let block_id = BlockId::new(item.item_id);

            // Try to find a slot with the same block that isn't full.
            let mut added = false;
            for i in 0..9 {
                if let Some(slot_id) = hotbar.slot(i) {
                    if slot_id == block_id {
                        // Found matching slot. For now, just keep it there.
                        // In a full inventory, we'd increment the count.
                        added = true;
                        break;
                    }
                }
            }

            // If no matching slot, try to find an empty slot.
            if !added {
                for i in 0..9 {
                    if let Some(slot_id) = hotbar.slot(i) {
                        if slot_id.is_air() {
                            hotbar.set_slot(i, block_id);
                            added = true;
                            break;
                        }
                    }
                }
            }

            // If still not added, the item stays on the ground.
            if !added {
                continue;
            }
        }

        // Remove the item entity.
        world.despawn(entity);
    }
}

use voxel_ecs::Entity;
