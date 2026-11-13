//! Item pickup system: handles picking up item entities.
//!
//! Vanilla-like ranges: items within one block of the player are collected,
//! and nearby stacks drift toward the player. The full stack count is
//! inserted, and leftovers stay on the ground.

use glam::Vec3;
use voxel_ecs::{Entity, World};

use crate::components::{PlayerEntity, Transform};
use crate::inventory::SurvivalInventory;
use crate::item_entity::ItemEntity;

/// Vacuum range in blocks (items within this range are attracted to the player).
const VACUUM_RANGE: f32 = 1.5;
/// Pickup range in blocks (items within this range are picked up).
const PICKUP_RANGE: f32 = 1.2;

/// Item pickup system entry point. Called each fixed timestep.
pub fn item_pickup_system(world: &mut World, dt: f32) {
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

    let mut to_pickup = Vec::new();
    let mut to_attract = Vec::new();

    for (entity, item, item_pos) in items {
        let distance = (item_pos - player_pos).length();
        if !item.can_pickup() {
            continue;
        }

        if distance <= PICKUP_RANGE {
            to_pickup.push((entity, item));
        } else if distance <= VACUUM_RANGE {
            to_attract.push((entity, item_pos));
        }
    }

    // Attract items towards player.
    for (entity, item_pos) in to_attract {
        let direction = (player_pos - item_pos).normalize_or_zero();
        let speed = 5.0;
        let new_pos = item_pos + direction * speed * dt;

        if let Some(mut transform) = world.get::<Transform>(entity).copied() {
            transform.pos = new_pos;
            world.set(entity, transform);
        }
    }

    // Pick up items.
    for (entity, item) in to_pickup {
        if let Some(inventory) = world.get_mut::<SurvivalInventory>(player_entity) {
            let stack = crate::items::ItemStack::new(item.block_id(), item.count.max(1));
            if inventory.insert(stack).is_some() {
                // Inventory full: leave the item on the ground.
                continue;
            }
        }
        world.despawn(entity);
    }
}
