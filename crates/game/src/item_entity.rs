//! Item entity: a dropped item in the world.
//!
//! Items fall under 16 blocks/s² of gravity, collide with the voxel world
//! (0.25-block cube), merge with nearby stacks and despawn after five minutes.
//! A ten-tick pickup delay prevents an item from being vacuumed the instant it
//! spawns. All timers are expressed in ticks and advanced from frame time.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use voxel_core::BlockId;
use voxel_ecs::{Entity, World};

use crate::components::Transform;
use crate::systems::PhysicsWorldRes;

/// Item gravity in blocks/s² (0.04 blocks/tick²).
const ITEM_GRAVITY: f32 = 16.0;
/// Per-tick air drag.
const ITEM_DRAG: f32 = 0.98;
/// Half-extent of the item collision cube.
const ITEM_HALF: f32 = 0.125;
/// Upward acceleration of a fully submerged item, blocks/s². Larger than
/// gravity so dropped items bob back to the surface instead of sinking.
const ITEM_BUOYANCY: f32 = 40.0;
/// How strongly a surface current pushes a floating item, blocks/s².
const ITEM_CURRENT_ACCEL: f32 = 3.0;

/// ECS component for a dropped item entity in the world.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ItemEntity {
    /// The item ID (block ID for block items).
    pub item_id: u16,
    /// Number of items in this stack.
    pub count: u16,
    /// Ticks before this item can be picked up (vanilla: 10).
    pub pickup_delay: f32,
    /// Ticks since spawn (despawns after 6000 ticks = 5 minutes).
    pub age: f32,
    /// Velocity for physics simulation.
    pub velocity: Vec3,
    /// Whether this item is on the ground (no more physics).
    pub on_ground: bool,
}

impl ItemEntity {
    /// Maximum age before despawning (5 minutes at 20 TPS).
    pub const MAX_AGE: f32 = 6000.0;
    /// Pickup delay in ticks (0.5 second at 20 TPS).
    pub const PICKUP_DELAY: f32 = 10.0;
    /// Merge radius in blocks.
    pub const MERGE_RADIUS: f32 = 1.0;

    /// Create a new item entity at a position with initial velocity.
    pub fn new(item_id: BlockId, count: u16) -> Self {
        Self {
            item_id: item_id.raw(),
            count,
            pickup_delay: Self::PICKUP_DELAY,
            age: 0.0,
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
            age: 0.0,
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
        self.pickup_delay <= 0.0
    }

    /// Whether this item should despawn.
    pub fn should_despawn(&self) -> bool {
        self.age >= Self::MAX_AGE
    }

    /// Advance age and pickup delay by `dt` seconds (20 ticks per second).
    pub fn tick(&mut self, dt: f32) {
        let ticks = dt * 20.0;
        self.age += ticks.max(0.0);
        if self.pickup_delay > 0.0 {
            self.pickup_delay = (self.pickup_delay - ticks).max(0.0);
        }
    }

    /// Whether this item can merge with another item.
    pub fn can_merge_with(&self, other: &ItemEntity) -> bool {
        self.item_id == other.item_id
            && self.count + other.count <= 64
            && self.can_pickup()
            && other.can_pickup()
    }
}

/// System to update item entities (age, despawn, merge, physics).
pub fn item_entity_system(world: &mut World, dt: f32) {
    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let mut to_despawn = Vec::new();
    let mut updates = Vec::new();

    let entities: Vec<Entity> = world.query::<&ItemEntity>().map(|(e, _)| e).collect();

    for entity in entities {
        let Some(mut item) = world.get::<ItemEntity>(entity).copied() else {
            continue;
        };
        let Some(mut transform) = world.get::<Transform>(entity).copied() else {
            continue;
        };

        item.tick(dt);

        if item.should_despawn() {
            to_despawn.push(entity);
            continue;
        }

        // Integrate with voxel collision. Items slide on the floor and stop
        // instead of falling through terrain.
        let drag = ITEM_DRAG.powf(dt * 20.0);
        // Probe slightly below the centre: an item bobbing with its centre just
        // above the surface must still see the water, otherwise a gravity /
        // buoyancy sawtooth makes it chatter instead of floating smoothly.
        let water_probe = transform.pos - Vec3::Y * (ITEM_HALF * 0.8);
        let water = physics
            .as_ref()
            .and_then(|p| p.0.water_sample(water_probe));
        if water.is_some() {
            // Water contact overrides ground state: a floating item must be
            // free to drift and bob even if it was resting on a bank.
            item.on_ground = false;
        }
        if !item.on_ground {
            if let Some(sample) = water {
                // Fraction of the item cube that is below the (wave-adjusted)
                // surface. Buoyancy equals gravity at ~40% coverage.
                let submerged = (sample.surface_y - (transform.pos.y - ITEM_HALF))
                    .clamp(0.0, ITEM_HALF * 2.0)
                    / (ITEM_HALF * 2.0);
                item.velocity.y +=
                    (ITEM_BUOYANCY * submerged - ITEM_GRAVITY) * dt;
                item.velocity.y = item.velocity.y.clamp(-6.0, 2.5);
                // Items are carried by the current and rapids.
                let water_drag = 0.82f32.powf(dt * 20.0);
                item.velocity.x = item.velocity.x * water_drag
                    + sample.flow.x * ITEM_CURRENT_ACCEL * dt;
                item.velocity.z = item.velocity.z * water_drag
                    + sample.flow.y * ITEM_CURRENT_ACCEL * dt;
                item.velocity.y *= water_drag;
            } else {
                item.velocity.y -= ITEM_GRAVITY * dt;
                item.velocity.x *= drag;
                item.velocity.z *= drag;
            }
        } else {
            let ground_friction = 0.6f32.powf(dt * 20.0);
            item.velocity.x *= ground_friction;
            item.velocity.z *= ground_friction;
            if item.velocity.x.abs() < 0.01 {
                item.velocity.x = 0.0;
            }
            if item.velocity.z.abs() < 0.01 {
                item.velocity.z = 0.0;
            }
        }

        if let Some(phys) = &physics {
            let res = voxel_physics::swept_aabb(
                &phys.0,
                transform.pos,
                Vec3::splat(ITEM_HALF),
                item.velocity * dt,
            );
            transform.pos = res.new_pos;
            if res.hit[3] {
                item.on_ground = true;
                item.velocity.y = 0.0;
            } else if res.hit[2] {
                item.velocity.y = 0.0;
            }
            if res.hit[0] || res.hit[1] {
                item.velocity.x = 0.0;
            }
            if res.hit[4] || res.hit[5] {
                item.velocity.z = 0.0;
            }
        } else {
            transform.pos += item.velocity * dt;
        }

        // Keep items inside the world's vertical bounds.
        if transform.pos.y < -64.0 {
            to_despawn.push(entity);
            continue;
        }

        updates.push((entity, item, transform));
    }

    for (entity, item, transform) in updates {
        world.set(entity, item);
        world.set(entity, transform);
    }

    for entity in to_despawn {
        world.despawn(entity);
    }

    merge_nearby_items(world);
}

/// Merge nearby item entities of the same type.
fn merge_nearby_items(world: &mut World) {
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

    let mut to_merge: Vec<(Entity, Entity)> = Vec::new();
    let mut already_merged = std::collections::HashSet::new();

    for i in 0..items.len() {
        let (entity_a, item_a, pos_a) = items[i];
        if already_merged.contains(&entity_a) {
            continue;
        }

        for (entity_b, item_b, pos_b) in items[(i + 1)..].iter().copied() {
            if already_merged.contains(&entity_b) {
                continue;
            }
            if (pos_a - pos_b).length() > ItemEntity::MERGE_RADIUS {
                continue;
            }
            if item_a.can_merge_with(&item_b) {
                to_merge.push((entity_a, entity_b));
                already_merged.insert(entity_b);
                break;
            }
        }
    }

    for (target, source) in to_merge {
        let Some(mut target_item) = world.get::<ItemEntity>(target).copied() else {
            continue;
        };
        let Some(source_item) = world.get::<ItemEntity>(source).copied() else {
            continue;
        };
        target_item.count += source_item.count;
        world.set(target, target_item);
        world.despawn(source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_entity_age() {
        let mut item = ItemEntity::new(BlockId(1), 1);
        assert_eq!(item.age, 0.0);
        assert!(!item.can_pickup()); // pickup delay starts at 10 ticks

        // 0.1s = 2 ticks.
        item.tick(0.1);
        assert!((item.age - 2.0).abs() < 1e-4);
        assert!((item.pickup_delay - 8.0).abs() < 1e-4);

        // Another 0.4s = 8 ticks, delay is satisfied.
        item.tick(0.4);
        assert!(item.can_pickup());
    }

    #[test]
    fn item_entity_despawn() {
        let mut item = ItemEntity::new(BlockId(1), 1);
        item.age = ItemEntity::MAX_AGE - 1.0;
        assert!(!item.should_despawn());
        item.tick(0.1);
        assert!(item.should_despawn());
    }

    #[test]
    fn item_entity_merge() {
        let mut a = ItemEntity::new(BlockId(1), 10);
        let mut b = ItemEntity::new(BlockId(1), 5);
        a.pickup_delay = 0.0;
        b.pickup_delay = 0.0;
        assert!(a.can_merge_with(&b));

        let c = ItemEntity::new(BlockId(2), 5);
        assert!(!a.can_merge_with(&c));
    }
}
