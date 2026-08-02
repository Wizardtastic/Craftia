//! Environmental damage system: fire, lava, void, suffocation.
//!
//! This system runs each tick and applies damage from environmental hazards.

use voxel_combat::{DamageEvent, DamageQueue, DamageSource};
use voxel_ecs::World;
use voxel_gamemode::GameMode;

use crate::components::{PlayerEntity, Transform};
use crate::systems::PhysicsWorldRes;

/// Environmental damage system entry point. Called each fixed timestep.
pub fn environmental_damage_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);
    if !game_mode.can_take_damage() {
        return;
    }

    // Skip if player is dead.
    let health = world
        .get::<voxel_combat::Health>(player_entity)
        .copied()
        .unwrap_or_default();
    if health.dead {
        return;
    }

    let transform = world
        .get::<Transform>(player_entity)
        .copied()
        .unwrap_or_default();
    let physics = world.resource::<PhysicsWorldRes>().cloned();

    if let Some(phys) = physics {
        let feet_pos = transform.pos;
        let head_pos = transform.pos + glam::Vec3::new(0.0, 0.8, 0.0);

        let feet_block = voxel_core::math::world_to_block(feet_pos);
        let head_block = voxel_core::math::world_to_block(head_pos);

        let registry = phys.0.registry();

        // Lava damage: only if the block is specifically lava (not water).
        let feet_def = registry.get(phys.0.get_block(feet_block.x, feet_block.y, feet_block.z));
        let in_lava = feet_def.kind == voxel_world::registry::BlockKind::Liquid
            && feet_def.name.as_ref() == "lava";

        if in_lava {
            if let Some(dq) = world.resource_mut::<DamageQueue>() {
                dq.push(
                    player_entity,
                    DamageEvent {
                        source: DamageSource::Lava { ticks: 1 },
                        amount: 4.0 / 20.0,
                    },
                );
            }
        }

        // Suffocation: head inside solid block (not liquid).
        let head_block_id = phys.0.get_block(head_block.x, head_block.y, head_block.z);
        let head_def = registry.get(head_block_id);
        if head_def.solid
            && !head_block_id.is_air()
            && head_def.kind != voxel_world::registry::BlockKind::Liquid
        {
            if let Some(dq) = world.resource_mut::<DamageQueue>() {
                dq.push(
                    player_entity,
                    DamageEvent {
                        source: DamageSource::Suffocation,
                        amount: 1.0 / 20.0,
                    },
                );
            }
        }

        // Void damage: y < -64.
        if transform.pos.y < -64.0 {
            if let Some(dq) = world.resource_mut::<DamageQueue>() {
                dq.push(
                    player_entity,
                    DamageEvent {
                        source: DamageSource::Void,
                        amount: 1000.0,
                    },
                );
            }
        }
    }
}
