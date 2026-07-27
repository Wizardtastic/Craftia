//! Regeneration system: natural health regen and saturation-based regen.
//!
//! This system runs each tick and:
//! 1. Tracks ticks since last damage
//! 2. Applies saturation-based regeneration (burst heal after eating)
//! 3. Applies natural regeneration based on food level and difficulty

use voxel_ecs::World;
use voxel_combat::{Health, RegenState};
use voxel_gamemode::{GameMode, HealthRegenMode};
use voxel_hunger::{Difficulty, Hunger};

use crate::components::PlayerEntity;
use crate::systems::hunger_system::DifficultyResource;

/// Regeneration system entry point. Called each fixed timestep.
pub fn regeneration_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world.get::<GameMode>(player_entity).copied().unwrap_or(GameMode::Survival);
    let regen_mode = game_mode.health_regeneration();

    // Creative/Spectator always regenerate (slowly).
    if regen_mode == HealthRegenMode::Always {
        let mut health = match world.get::<Health>(player_entity) {
            Some(h) => *h,
            None => return,
        };
        if health.current < health.max {
            // Heal 1 HP every 20 ticks (0.5 sec) in creative.
            let mut regen = world.get::<RegenState>(player_entity).copied().unwrap_or_default();
            regen.ticks_since_damage += 1;
            if regen.ticks_since_damage % 20 == 0 {
                health.heal(1.0);
                world.set(player_entity, health);
            }
            world.set(player_entity, regen);
        }
        return;
    }

    // No regen for modes that don't have it.
    if regen_mode == HealthRegenMode::None {
        return;
    }

    let difficulty = world.resource::<DifficultyResource>().map(|d| d.0).unwrap_or(Difficulty::Normal);

    // Get components.
    let mut health = match world.get::<Health>(player_entity) {
        Some(h) => *h,
        None => return,
    };
    let mut hunger = world.get::<Hunger>(player_entity).copied().unwrap_or_default();
    let mut regen = world.get::<RegenState>(player_entity).copied().unwrap_or_default();

    // Update ticks since damage.
    if health.invulnerability_ticks > 0 {
        regen.ticks_since_damage = 0;
    } else {
        regen.ticks_since_damage += 1;
    }

    // Skip if at full health.
    if health.current >= health.max {
        world.set(player_entity, regen);
        world.set(player_entity, hunger);
        return;
    }

    // Saturation-based regeneration (burst heal after eating).
    // In MC: heals 1 HP per tick, costs 6.0 exhaustion per heal.
    // With 5 starting saturation, that's ~3 HP healed before saturation runs out.
    if hunger.saturation > 0.0 && health.current < health.max {
        health.heal(1.0);
        hunger.add_exhaustion(6.0);
        world.set(player_entity, health);
        world.set(player_entity, hunger);
        world.set(player_entity, regen);
        return;
    }

    // Natural regeneration (when food >= 18 and regen delay elapsed).
    // In MC Normal: 1 HP every 80 ticks (1.33s), costs 6.0 exhaustion.
    if hunger.food >= 18.0 && regen.can_regen_natural() {
        if let Some(interval) = difficulty.regen_interval() {
            if regen.ticks_since_damage % interval == 0 {
                health.heal(1.0);
                hunger.add_exhaustion(6.0);
                world.set(player_entity, health);
                world.set(player_entity, hunger);
            }
        }
    }

    world.set(player_entity, health);
    world.set(player_entity, regen);
}
