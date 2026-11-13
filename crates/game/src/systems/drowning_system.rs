//! Drowning system: vanilla air supply and drowning damage.
//!
//! * Air lasts 300 ticks (15 s) underwater and refills when the head is out.
//! * Once air is gone the player takes 2 HP per second (1 HP on Easy).
//! * Damage goes through the normal damage queue so death and armor rules
//!   stay in one place.

use voxel_combat::{AirSupply, DamageEvent, DamageQueue, DamageSource};
use voxel_ecs::World;
use voxel_gamemode::GameMode;
use voxel_hunger::Difficulty;

use crate::components::{PlayerEntity, PlayerState, Transform};
use crate::systems::hunger_system::DifficultyResource;
use crate::systems::PhysicsWorldRes;

/// Tracks drowning damage timing per entity.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrowningState {
    /// Seconds since the last drowning damage tick.
    pub damage_timer: f32,
}

/// Drowning system entry point. Called each fixed timestep.
pub fn drowning_system(world: &mut World, dt: f32) {
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

    let transform = world
        .get::<Transform>(player_entity)
        .copied()
        .unwrap_or_default();
    let eye_offset = world
        .get::<PlayerState>(player_entity)
        .map(|s| s.eye_offset)
        .unwrap_or(0.72);
    let physics = world.resource::<PhysicsWorldRes>().cloned();

    let head_underwater = physics
        .as_ref()
        .map(|phys| {
            let head = transform.pos + glam::Vec3::new(0.0, eye_offset, 0.0);
            let block = voxel_core::math::world_to_block(head);
            phys.0.is_liquid(block.x, block.y, block.z)
        })
        .unwrap_or(false);

    let mut air = world
        .get::<AirSupply>(player_entity)
        .copied()
        .unwrap_or_default();
    let mut drowning = world
        .get::<DrowningState>(player_entity)
        .copied()
        .unwrap_or_default();

    if head_underwater {
        // 300 air ticks drain at 20 ticks per second.
        air.current = (air.current - 20.0 * dt).max(0.0);
        if air.is_drowning() {
            drowning.damage_timer += dt;
            if drowning.damage_timer >= 1.0 {
                drowning.damage_timer = 0.0;
                let difficulty = world
                    .resource::<DifficultyResource>()
                    .map(|d| d.0)
                    .unwrap_or(Difficulty::Normal);
                let damage = difficulty.drowning_damage();
                if damage > 0.0 {
                    if let Some(dq) = world.resource_mut::<DamageQueue>() {
                        dq.push(
                            player_entity,
                            DamageEvent {
                                source: DamageSource::Drowning,
                                amount: damage,
                            },
                        );
                    }
                }
            }
        }
    } else {
        // Vanilla refills the air bar almost immediately at the surface.
        air.current = air.max;
        drowning.damage_timer = 0.0;
    }

    world.set(player_entity, air);
    world.set(player_entity, drowning);
}
