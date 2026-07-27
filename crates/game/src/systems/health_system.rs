//! Health system: processes damage events, manages invulnerability, checks death.
//!
//! This system runs each tick and:
//! 1. Drains the `DamageQueue` resource
//! 2. Applies damage to entities with `Health` components
//! 3. Ticks down invulnerability timers
//! 4. Fires `DeathEvent` when health reaches zero

use voxel_ecs::World;
use voxel_combat::{DamageQueue, Health, DeathEvent};
use voxel_gamemode::GameMode;
use crate::systems::hunger_system::GameTimeResource;

/// Health system entry point. Called each fixed timestep.
pub fn health_system(world: &mut World, _dt: f32) {
    // Read current game time from ECS resource.
    let game_time = world.resource::<GameTimeResource>().map(|r| r.0).unwrap_or(0.0);

    // Drain all pending damage events.
    let events = world.resource_mut::<DamageQueue>().map(|q| q.drain()).unwrap_or_default();

    for (entity_opt, event) in events {
        let Some(entity) = entity_opt else { continue };

        // Get the game mode for this entity (defaults to Survival if missing).
        let game_mode = world.get::<GameMode>(entity).copied().unwrap_or(GameMode::Survival);

        // Skip damage if the game mode doesn't allow it.
        if !game_mode.can_take_damage() {
            continue;
        }

        // Get the health component.
        let Some(health) = world.get_mut::<Health>(entity) else { continue };

        // Check invulnerability (unless the source ignores it).
        if health.is_invulnerable() && !event.source.ignores_invulnerability() {
            continue;
        }

        // Apply the damage.
        let _actual = health.apply_damage(event.amount, game_time);

        // Check for death.
        if health.dead {
            let message = event.source.death_message("Player");
            log::info!("Player died: {}", message);

            // Store the death event on the entity for the engine to pick up.
            // We'll use a simple approach: set a DeathEvent component.
            world.set(entity, DeathEvent {
                source: event.source,
                message,
            });
        }
    }

    // Tick invulnerability timers for all entities with Health.
    // We collect entities first to avoid borrow conflicts.
    let entities: Vec<_> = world.query::<&Health>().map(|(e, _)| e).collect();
    for entity in entities {
        if let Some(health) = world.get_mut::<Health>(entity) {
            health.tick();
        }
    }
}
