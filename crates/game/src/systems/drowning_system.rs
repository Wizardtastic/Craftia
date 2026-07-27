//! Drowning system: air supply drain and drowning damage.
//!
//! This system runs each tick and:
//! 1. Checks if the player's head is underwater
//! 2. Drains air supply when submerged
//! 3. Applies drowning damage directly when air is depleted (no invulnerability)

use voxel_ecs::World;
use voxel_combat::{AirSupply, Health};
use voxel_gamemode::GameMode;
use voxel_hunger::Difficulty;

use crate::components::{PlayerEntity, Transform};
use crate::systems::hunger_system::DifficultyResource;
use crate::systems::PhysicsWorldRes;

/// Tracks drowning damage timing per entity.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrowningState {
    /// Ticks since air reached 0. Used to apply damage at intervals.
    pub ticks_without_air: u32,
}

/// Drowning system entry point. Called each fixed timestep.
pub fn drowning_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world.get::<GameMode>(player_entity).copied().unwrap_or(GameMode::Survival);

    // Creative/Spectator don't drown.
    if !game_mode.can_take_damage() {
        return;
    }

    let difficulty = world.resource::<DifficultyResource>().map(|d| d.0).unwrap_or(Difficulty::Normal);

    // Check if player's head is underwater.
    let transform = world.get::<Transform>(player_entity).copied().unwrap_or_default();
    let physics = world.resource::<PhysicsWorldRes>().cloned();

    let head_underwater = if let Some(phys) = physics {
        let head_pos = transform.pos + glam::Vec3::new(0.0, 0.8, 0.0);
        let block_pos = voxel_core::math::world_to_block(head_pos);
        phys.0.is_liquid(block_pos.x, block_pos.y, block_pos.z)
    } else {
        false
    };

    // Get air supply and drowning state.
    let mut air = world.get::<AirSupply>(player_entity).copied().unwrap_or_default();
    let mut drowning = world.get::<DrowningState>(player_entity).copied().unwrap_or_default();

    if head_underwater {
        // Drain air at ~0.333/tick → 300 / 0.333 = ~900 ticks = 15 seconds at 60 tps.
        air.current = (air.current - 0.333).max(0.0);

        if air.is_drowning() {
            // Track ticks without air.
            drowning.ticks_without_air += 1;

            // Apply drowning damage once per second (every 60 ticks).
            // Bypasses invulnerability — applied directly to Health.
            if drowning.ticks_without_air % 60 == 0 {
                let damage = difficulty.drowning_damage();
                if damage > 0.0 {
                    if let Some(health) = world.get_mut::<Health>(player_entity) {
                        if !health.dead {
                            health.current = (health.current - damage).max(0.0);
                            if health.current <= 0.0 {
                                health.current = 0.0;
                                health.dead = true;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Refill air when head is above water.
        // 50/tick → 300/50 = 6 ticks to fully refill (nearly instant).
        air.current = (air.current + 50.0).min(air.max);
        drowning.ticks_without_air = 0;
    }

    world.set(player_entity, air);
    world.set(player_entity, drowning);
}
