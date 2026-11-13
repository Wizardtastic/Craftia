//! Environmental damage: lava, cactus, suffocation and the void.
//!
//! Vanilla rates, on independent cooldowns so each hazard keeps its own
//! cadence instead of relying on the damage invulnerability window:
//! * Lava: 4 HP every 0.5 s.
//! * Cactus: 1 HP every 0.5 s.
//! * Suffocation: 1 HP per second while the head is inside a solid block.
//! * Void: lethal damage below y = −64.

use voxel_combat::{DamageEvent, DamageQueue, DamageSource};
use voxel_ecs::{Entity, World};
use voxel_gamemode::GameMode;

use crate::components::{PlayerEntity, PlayerState, Transform};
use crate::systems::PhysicsWorldRes;

/// Per-entity hazard cooldowns (seconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct HazardTimers {
    pub lava: f32,
    pub cactus: f32,
    pub suffocation: f32,
    pub void: f32,
}

const LAVA_INTERVAL: f32 = 0.5;
const CACTUS_INTERVAL: f32 = 0.5;
const SUFFOCATION_INTERVAL: f32 = 1.0;
const VOID_INTERVAL: f32 = 0.5;

/// Environmental damage system entry point. Called each fixed timestep.
pub fn environmental_damage_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);
    let health = world
        .get::<voxel_combat::Health>(player_entity)
        .copied()
        .unwrap_or_default();
    if !game_mode.can_take_damage() || health.dead {
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
    let Some(phys) = physics else { return };

    let registry = phys.0.registry();
    let feet = voxel_core::math::world_to_block(transform.pos);
    let head =
        voxel_core::math::world_to_block(transform.pos + glam::Vec3::new(0.0, eye_offset, 0.0));
    let name_at = |p: glam::IVec3| {
        registry
            .get(phys.0.get_block(p.x, p.y, p.z))
            .name
            .to_string()
    };

    let feet_name = name_at(feet);
    let head_name = name_at(head);
    let in_lava = feet_name == "lava" || head_name == "lava";

    let head_id = phys.0.get_block(head.x, head.y, head.z);
    let head_def = registry.get(head_id);
    let head_solid = head_def.solid
        && !head_id.is_air()
        && head_def.kind != voxel_world::registry::BlockKind::Liquid
        && !head_def.name.starts_with("light_");

    // Sample feet, legs and torso so the full 1.8-block body triggers cactus.
    let torso =
        voxel_core::math::world_to_block(transform.pos + glam::Vec3::new(0.0, 0.4, 0.0));
    let in_cactus = [feet, torso, head]
        .iter()
        .any(|p| name_at(*p) == "cactus");

    let in_void = transform.pos.y < -64.0;

    let mut timers = world
        .get::<HazardTimers>(player_entity)
        .copied()
        .unwrap_or_default();

    apply_periodic(
        world,
        player_entity,
        &mut timers.lava,
        in_lava,
        LAVA_INTERVAL,
        dt,
        DamageEvent {
            source: DamageSource::Lava { ticks: 1 },
            amount: 4.0,
        },
    );
    apply_periodic(
        world,
        player_entity,
        &mut timers.cactus,
        in_cactus,
        CACTUS_INTERVAL,
        dt,
        DamageEvent {
            source: DamageSource::Cactus,
            amount: 1.0,
        },
    );
    apply_periodic(
        world,
        player_entity,
        &mut timers.suffocation,
        head_solid,
        SUFFOCATION_INTERVAL,
        dt,
        DamageEvent {
            source: DamageSource::Suffocation,
            amount: 1.0,
        },
    );
    apply_periodic(
        world,
        player_entity,
        &mut timers.void,
        in_void,
        VOID_INTERVAL,
        dt,
        DamageEvent {
            source: DamageSource::Void,
            amount: 4.0,
        },
    );

    world.set(player_entity, timers);
}

/// Apply `amount` damage each time `interval` elapses while `active`.
/// Inactive hazards reset to a ready cooldown so contact hurts immediately.
fn apply_periodic(
    world: &mut World,
    entity: Entity,
    timer: &mut f32,
    active: bool,
    interval: f32,
    dt: f32,
    event: DamageEvent,
) {
    if !active {
        *timer = interval;
        return;
    }
    *timer -= dt;
    if *timer <= 0.0 {
        *timer = interval;
        if let Some(dq) = world.resource_mut::<DamageQueue>() {
            dq.push(entity, event);
        }
    }
}
