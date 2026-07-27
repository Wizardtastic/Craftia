//! Hunger system: exhaustion → saturation → food decay, starvation damage.
//!
//! This system runs each tick and:
//! 1. Accumulates exhaustion from player actions
//! 2. When exhaustion reaches 4, decays saturation then food
//! 3. Deals starvation damage when food == 0

use voxel_ecs::World;
use voxel_combat::{DamageEvent, DamageQueue, DamageSource};
use voxel_gamemode::GameMode;
use voxel_hunger::{Difficulty, Hunger, exhaustion};

use crate::components::{PlayerEntity, PlayerInput, PlayerState, Velocity};

/// Resource tracking the world's difficulty level.
#[derive(Clone, Copy, Debug, Default)]
pub struct DifficultyResource(pub Difficulty);

/// Resource tracking the current game time in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct GameTimeResource(pub f64);

/// Hunger system entry point. Called each fixed timestep.
pub fn hunger_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world.get::<GameMode>(player_entity).copied().unwrap_or(GameMode::Survival);

    // Only apply hunger in modes that have it.
    if !game_mode.has_hunger() {
        return;
    }

    let difficulty = world.resource::<DifficultyResource>().map(|d| d.0).unwrap_or(Difficulty::Normal);

    // Get player state for movement tracking.
    let input = world.get::<PlayerInput>(player_entity).copied().unwrap_or_default();
    let state = world.get::<PlayerState>(player_entity).copied().unwrap_or_default();
    let velocity = world.get::<Velocity>(player_entity).copied().unwrap_or_default();

    // Get or create hunger component.
    let mut hunger = world.get::<Hunger>(player_entity).copied().unwrap_or_default();

    // Accumulate exhaustion from movement.
    if input.sprinting && state.on_ground {
        // Sprint exhaustion per meter (approximate from velocity)
        let speed = velocity.lin.length();
        if speed > 0.1 {
            hunger.add_exhaustion(exhaustion::SPRINT_PER_METER * speed * _dt);
        }
    }

    // Jump exhaustion
    if input.jump && state.on_ground {
        if input.sprinting {
            hunger.add_exhaustion(exhaustion::SPRINT_JUMP);
        } else {
            hunger.add_exhaustion(exhaustion::JUMP);
        }
    }

    // Swimming exhaustion
    if state.in_water {
        let speed = velocity.lin.length();
        if speed > 0.1 {
            hunger.add_exhaustion(exhaustion::SWIM_PER_METER * speed * _dt);
        }
    }

    // Tick hunger (exhaustion → saturation → food decay)
    let should_starve = if difficulty.has_hunger_depletion() {
        hunger.tick()
    } else {
        false
    };

    // Sprint gating: force sprint off when food <= 6
    if !hunger.can_sprint() && input.sprinting {
        // We can't directly modify input here, but we set a flag
        // The movement system will check hunger.can_sprint()
    }

    // Starvation damage
    if should_starve && difficulty.has_starvation() {
        // Starvation damage: 1 heart per 4 seconds (every 80 ticks at 20 tps)
        // We use a simple approach: damage every tick at a low rate
        if let Some(dq) = world.resource_mut::<DamageQueue>() {
            dq.push(player_entity, DamageEvent {
                source: DamageSource::Starvation,
                amount: 0.025, // 0.5 hearts per second / 20 ticks
            });
        }
    }

    world.set(player_entity, hunger);
}
