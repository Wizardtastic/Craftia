//! Hunger system: exhaustion → saturation → food decay, starvation damage.
//!
//! Vanilla accounting:
//! * Every 4 exhaustion points consume 1 saturation, then 1 food point.
//! * Walking costs 0.01 per block, sprinting 0.1, swimming 0.01, jumping 0.05
//!   (sprint-jumping 0.2), mining 0.005, taking damage 0.1 per point.
//! * Sprinting is disabled at food ≤ 6 (enforced by the movement system).
//! * Starvation deals 1 HP every 4 s (2 s on Hard); Easy stops at 10 HP.

use voxel_combat::{DamageEvent, DamageQueue, DamageSource, Health};
use voxel_ecs::World;
use voxel_gamemode::GameMode;
use voxel_hunger::{exhaustion, Difficulty, Hunger};

use crate::components::{PlayerEntity, PlayerInput, PlayerState, Velocity};

/// Resource tracking the world's difficulty level.
#[derive(Clone, Copy, Debug, Default)]
pub struct DifficultyResource(pub Difficulty);

/// Resource tracking the current game time in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct GameTimeResource(pub f64);

/// Hunger system entry point. Called each fixed timestep.
pub fn hunger_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);

    // Only apply hunger in modes that have it.
    if !game_mode.has_hunger() {
        return;
    }

    let difficulty = world
        .resource::<DifficultyResource>()
        .map(|d| d.0)
        .unwrap_or(Difficulty::Normal);

    let input = world
        .get::<PlayerInput>(player_entity)
        .copied()
        .unwrap_or_default();
    let state = world
        .get::<PlayerState>(player_entity)
        .copied()
        .unwrap_or_default();
    let velocity = world
        .get::<Velocity>(player_entity)
        .copied()
        .unwrap_or_default();

    let mut hunger = world
        .get::<Hunger>(player_entity)
        .copied()
        .unwrap_or_default();

    // Distance travelled this step drives movement exhaustion.
    let horizontal_speed = glam::Vec3::new(velocity.lin.x, 0.0, velocity.lin.z).length();
    let distance = horizontal_speed * dt;

    if state.in_water {
        hunger.add_exhaustion(exhaustion::SWIM_PER_METER * distance);
    } else if state.on_ground && horizontal_speed > 0.05 {
        let sprinting = input.sprinting && input.wish.z < -0.1 && hunger.can_sprint();
        let rate = if sprinting {
            exhaustion::SPRINT_PER_METER
        } else {
            exhaustion::WALK_PER_METER
        };
        hunger.add_exhaustion(rate * distance);
    }

    // Spend accumulated exhaustion. Peaceful never drains hunger.
    if difficulty.has_hunger_depletion() {
        hunger.tick();
    }

    // --- starvation --------------------------------------------------------
    if difficulty.has_starvation() && hunger.food <= 0.0 {
        hunger.starvation_timer += dt;
        let interval = if difficulty == Difficulty::Hard {
            2.0
        } else {
            4.0
        };
        if hunger.starvation_timer >= interval {
            hunger.starvation_timer = 0.0;

            let health = world
                .get::<Health>(player_entity)
                .copied()
                .unwrap_or_default();
            // Vanilla Easy difficulty stops starvation at 10 HP.
            let may_damage = !health.dead
                && (difficulty != Difficulty::Easy || health.current > 10.0);
            if may_damage {
                if let Some(dq) = world.resource_mut::<DamageQueue>() {
                    dq.push(
                        player_entity,
                        DamageEvent {
                            source: DamageSource::Starvation,
                            amount: 1.0,
                        },
                    );
                }
            }
        }
    } else {
        hunger.starvation_timer = 0.0;
    }

    world.set(player_entity, hunger);
}
