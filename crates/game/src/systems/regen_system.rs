//! Regeneration system: vanilla natural regeneration.
//!
//! * Natural regen requires food ≥ 18 and is available on every difficulty.
//! * With saturation left: 1 HP every 10 ticks (0.5 s).
//! * Without saturation: 1 HP every 80 ticks (4 s).
//! * Each health point costs 6 exhaustion.
//! * Peaceful heals 1 HP every 20 ticks regardless of hunger.
//! * Creative / spectator heal 1 HP every 20 ticks.

use voxel_combat::{Health, RegenState};
use voxel_ecs::World;
use voxel_gamemode::{GameMode, HealthRegenMode};
use voxel_hunger::{exhaustion, Difficulty, Hunger};

use crate::components::PlayerEntity;
use crate::systems::hunger_system::DifficultyResource;

/// Health regeneration is gated on the vanilla food threshold.
const NATURAL_REGEN_FOOD: f32 = 18.0;
/// 10 ticks = 0.5 s (saturation-based regen).
const SATURATION_INTERVAL_TICKS: f32 = 10.0;
/// 80 ticks = 4 s (normal natural regen).
const NATURAL_INTERVAL_TICKS: f32 = 80.0;
/// Peaceful / creative interval.
const ALWAYS_INTERVAL_TICKS: f32 = 20.0;

/// Regeneration system entry point. Called each fixed timestep.
pub fn regeneration_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);
    let regen_mode = game_mode.health_regeneration();

    if regen_mode == HealthRegenMode::None {
        return;
    }

    let mut health = match world.get::<Health>(player_entity) {
        Some(h) => *h,
        None => return,
    };
    let mut regen = world
        .get::<RegenState>(player_entity)
        .copied()
        .unwrap_or_default();
    let mut hunger = world
        .get::<Hunger>(player_entity)
        .copied()
        .unwrap_or_default();

    if health.dead {
        return;
    }

    if health.current >= health.max {
        regen.heal_timer = 0.0;
        world.set(player_entity, regen);
        return;
    }

    let difficulty = world
        .resource::<DifficultyResource>()
        .map(|d| d.0)
        .unwrap_or(Difficulty::Normal);

    let interval = if regen_mode == HealthRegenMode::Always {
        ALWAYS_INTERVAL_TICKS
    } else if difficulty == Difficulty::Peaceful {
        // Peaceful regenerates health quickly without consuming food.
        ALWAYS_INTERVAL_TICKS
    } else if game_mode == GameMode::Survival || game_mode == GameMode::Adventure {
        if hunger.food < NATURAL_REGEN_FOOD {
            regen.heal_timer = 0.0;
            world.set(player_entity, regen);
            return;
        }
        if hunger.saturation > 0.0 {
            SATURATION_INTERVAL_TICKS
        } else {
            NATURAL_INTERVAL_TICKS
        }
    } else {
        ALWAYS_INTERVAL_TICKS
    };

    regen.heal_timer += dt * 20.0;
    if regen.heal_timer >= interval {
        regen.heal_timer = 0.0;
        health.heal(1.0);
        // Vanilla charges 6 exhaustion per healed health point on all
        // difficulties except Peaceful.
        if difficulty != Difficulty::Peaceful {
            hunger.add_exhaustion(exhaustion::REGEN);
        }
        world.set(player_entity, health);
        world.set(player_entity, hunger);
    }
    world.set(player_entity, regen);
}
