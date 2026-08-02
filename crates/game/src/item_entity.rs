//! Item entity: a dropped item in the world.
//!
//! Item entities are spawned when blocks are broken or items are dropped.
//! They have physics, can merge with nearby items of the same type,
//! and despawn after a timeout.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use voxel_core::BlockId;
use voxel_ecs::{Entity, World};

use crate::components::Transform;

/// ECS component for a dropped item entity in the world.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ItemEntity {
    /// The item ID (block ID for block items).
    pub item_id: u16,
    /// Number of items in this stack.
    pub count: u16,
    /// Ticks before this item can be picked up (prevents instant pickup).
    pub pickup_delay: u32,
    /// Ticks since spawn (despawns after 6000 ticks = 5 minutes).
    pub age: u64,
    /// Velocity for physics simulation.
    pub velocity: Vec3,
    /// Whether this item is on the ground (no more physics).
    pub on_ground: bool,
}

impl ItemEntity {
    /// Maximum age before despawning (5 minutes at 20 TPS).
    pub const MAX_AGE: u64 = 6000;
    /// Pickup delay in ticks (1 second at 20 TPS).
    pub const PICKUP_DELAY: u32 = 20;
    /// Merge radius in blocks.
    pub const MERGE_RADIUS: f32 = 1.5;

    /// Create a new item entity at a position with initial velocity.
    pub fn new(item_id: BlockId, count: u16) -> Self {
        Self {
            item_id: item_id.raw(),
            count,
            pickup_delay: Self::PICKUP_DELAY,
            age: 0,
            velocity: Vec3::ZERO,
            on_ground: false,
        }
    }

    /// Create a new item entity with a specific velocity (for dropping).
    pub fn with_velocity(item_id: BlockId, count: u16, velocity: Vec3) -> Self {
        Self {
            item_id: item_id.raw(),
            count,
            pickup_delay: Self::PICKUP_DELAY,
            age: 0,
            velocity,
            on_ground: false,
        }
    }

    /// Get the BlockId for this item.
    pub fn block_id(&self) -> BlockId {
        BlockId::new(self.item_id)
    }

    /// Whether this item can be picked up.
    pub fn can_pickup(&self) -> bool {
        self.pickup_delay == 0
    }

    /// Whether this item should despawn.
    pub fn should_despawn(&self) -> bool {
        self.age >= Self::MAX_AGE
    }

    /// Tick the item entity (age, pickup delay).
    pub fn tick(&mut self) {
        self.age += 1;
        if self.pickup_delay > 0 {
            self.pickup_delay -= 1;
        }
    }

    /// Whether this item can merge with another item.
    pub fn can_merge_with(&self, other: &ItemEntity) -> bool {
        self.item_id == other.item_id
            && self.count + other.count <= 64
            && self.pickup_delay == 0
            && other.pickup_delay == 0
    }
}

/// System to update item entities (age, despawn, merge, physics).
pub fn item_entity_system(world: &mut World, dt: f32) {
    // Collect all item entities and their positions.
    let mut to_despawn = Vec::new();
    let mut updates = Vec::new();

    // Iterate through all entities with ItemEntity component.
    let entities: Vec<Entity> = world.query::<&ItemEntity>().map(|(e, _)| e).collect();

    for entity in entities {
        let Some(mut item) = world.get::<ItemEntity>(entity).copied() else {
            continue;
        };
        let Some(mut transform) = world.get::<Transform>(entity).copied() else {
            continue;
        };

        // Tick the item (age, pickup delay).
        item.tick();

        // Check for despawn.
        if item.should_despawn() {
            to_despawn.push(entity);
            continue;
        }

        // Apply physics if not on ground.
        if !item.on_ground {
            // Apply gravity.
            item.velocity.y -= 9.8 * dt;

            // Update position.
            transform.pos += item.velocity * dt;

            // Simple ground collision (assume ground at y=0 for now).
            if transform.pos.y < 0.5 {
                transform.pos.y = 0.5;
                item.velocity.y = 0.0;
                item.on_ground = true;
            }

            // Apply friction when on ground.
            if item.on_ground {
                item.velocity.x *= 0.9;
                item.velocity.z *= 0.9;
                if item.velocity.x.abs() < 0.01 {
                    item.velocity.x = 0.0;
                }
                if item.velocity.z.abs() < 0.01 {
                    item.velocity.z = 0.0;
                }
            }
        }

        updates.push((entity, item, transform));
    }

    // Apply updates.
    for (entity, item, transform) in updates {
        world.set(entity, item);
        world.set(entity, transform);
    }

    // Despawn expired items.
    for entity in to_despawn {
        world.despawn(entity);
    }

    // Merge nearby items of the same type.
    merge_nearby_items(world);
}

/// Merge nearby item entities of the same type.
fn merge_nearby_items(world: &mut World) {
    // Collect all items with their positions.
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

    // Find pairs that can merge.
    let mut to_merge: Vec<(Entity, Entity)> = Vec::new();
    let mut already_merged = std::collections::HashSet::new();

    for i in 0..items.len() {
        let (entity_a, item_a, pos_a) = items[i];
        if already_merged.contains(&entity_a) {
            continue;
        }

        for j in (i + 1)..items.len() {
            let (entity_b, item_b, pos_b) = items[j];
            if already_merged.contains(&entity_b) {
                continue;
            }

            // Check distance.
            let dist = (pos_a - pos_b).length();
            if dist > ItemEntity::MERGE_RADIUS {
                continue;
            }

            // Check if can merge.
            if item_a.can_merge_with(&item_b) {
                to_merge.push((entity_a, entity_b));
                already_merged.insert(entity_b);
                break; // Only merge one pair per item.
            }
        }
    }

    // Perform merges.
    for (target, source) in to_merge {
        let Some(mut target_item) = world.get::<ItemEntity>(target).copied() else {
            continue;
        };
        let Some(source_item) = world.get::<ItemEntity>(source).copied() else {
            continue;
        };

        // Merge counts.
        target_item.count += source_item.count;

        // Update target.
        world.set(target, target_item);

        // Remove source.
        world.despawn(source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_entity_age() {
        let mut item = ItemEntity::new(BlockId(1), 1);
        assert_eq!(item.age, 0);
        assert!(!item.can_pickup()); // pickup_delay starts at 20, can't pickup yet

        item.tick();
        assert_eq!(item.age, 1);
        assert_eq!(item.pickup_delay, 19);

        // After 20 ticks, can pickup.
        for _ in 0..19 {
            item.tick();
        }
        assert!(item.can_pickup());
    }

    #[test]
    fn item_entity_despawn() {
        let mut item = ItemEntity::new(BlockId(1), 1);
        item.age = ItemEntity::MAX_AGE - 1;
        assert!(!item.should_despawn());

        item.tick();
        assert!(item.should_despawn());
    }

    #[test]
    fn item_entity_merge() {
        let mut a = ItemEntity::new(BlockId(1), 10);
        let mut b = ItemEntity::new(BlockId(1), 5);
        // Need to wait for pickup delay before merging.
        a.pickup_delay = 0;
        b.pickup_delay = 0;
        assert!(a.can_merge_with(&b));

        let c = ItemEntity::new(BlockId(2), 5);
        assert!(!a.can_merge_with(&c));
    }
}
