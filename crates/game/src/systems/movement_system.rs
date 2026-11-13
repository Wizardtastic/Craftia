//! Player movement, modelled on vanilla Minecraft's 20 Hz entity physics.
//!
//! The engine may step at any rate; this system consumes elapsed time into
//! fixed 20-tick physics ticks, exactly like the vanilla server. That makes
//! speeds, acceleration, jump height and fall damage independent of the
//! render/simulation refresh rate. The camera interpolates between the last
//! two ticks for smooth motion.
//!
//! Vanilla constants used here:
//! * walk 4.317, sprint 5.612, sneak 1.295, creative fly 10.92 blocks/s
//! * jump impulse 0.42 blocks/tick (8.4 blocks/s → 1.2522-block apex)
//! * gravity 0.08 blocks/tick², vertical drag 0.98/tick
//! * ground friction 0.546/tick, air drag 0.91/tick, air control 0.4/tick
//! * fall damage floor(distance − 3), sprint needs food > 6

use glam::Vec3;
use voxel_combat::{DamageEvent, DamageQueue, DamageSource};
use voxel_core::Camera;
use voxel_ecs::World;
use voxel_gamemode::GameMode;
use voxel_hunger::{exhaustion, Hunger};

use crate::components::{Aabb, PlayerEntity, PlayerInput, PlayerState, Transform, Velocity};

/// Resource: shared camera. The engine writes/reads it; the movement
/// system doesn't touch it directly (use `update_camera_from_transform`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraResource(pub Camera);

/// Resource: the voxel world, shared from the engine. The movement system
/// uses it for collision detection and water checks.
#[derive(Clone)]
pub struct PhysicsWorldRes(pub std::sync::Arc<voxel_world::world::World>);

// --- Timing ----------------------------------------------------------------
/// Vanilla simulates entities at 20 ticks per second.
pub const TICK_SECONDS: f32 = 1.0 / 20.0;
/// Cap on ticks processed in one call (guards against long stalls).
const MAX_TICKS_PER_UPDATE: u32 = 5;
/// Collision substep length, so fast falls cannot tunnel through a block.
const MAX_COLLISION_STEP: f32 = 0.45;
/// How long a jump press stays queued, so taps shorter than a 50 ms tick are
/// not dropped at high refresh rates.
const JUMP_BUFFER_SECONDS: f32 = 0.15;

// --- Horizontal speeds (blocks per second) --------------------------------
pub const WALK_SPEED: f32 = 4.317;
pub const SPRINT_SPEED: f32 = 5.612;
pub const SNEAK_SPEED: f32 = 1.295;
pub const FLY_SPEED: f32 = 10.92;
pub const SPRINT_FLY_SPEED: f32 = 21.78;
pub const SWIM_SPEED: f32 = 2.2;

// --- Vertical physics ------------------------------------------------------
/// 0.42 blocks/tick at 20 tps.
pub const JUMP_SPEED: f32 = 8.4;
/// Sprint jumps add 0.2 blocks/tick of forward momentum.
pub const SPRINT_JUMP_BOOST: f32 = 4.0;
/// 0.08 blocks/tick² at 20 tps.
pub const GRAVITY: f32 = 32.0;
/// Vanilla clamps falling at 78.4 blocks/s (3.92 blocks/tick).
pub const TERMINAL_VELOCITY: f32 = 78.4;
/// 0.02 blocks/tick of air acceleration, expressed per second.
pub const AIR_ACCEL: f32 = 8.0;
/// Water gravity: 0.02 blocks/tick².
pub const WATER_GRAVITY: f32 = 8.0;
pub const WATER_SWIM_UP: f32 = 2.2;
pub const WATER_SINK: f32 = 1.8;

// --- Per-tick drag ---------------------------------------------------------
/// Default block slipperiness (0.6) × global 0.91.
pub const GROUND_FRICTION: f32 = 0.546;
pub const AIR_FRICTION: f32 = 0.91;
/// Vertical air drag: gives jump the vanilla 1.2522-block apex.
pub const AIR_DRAG: f32 = 0.98;
pub const WATER_FRICTION: f32 = 0.8;

// --- Player dimensions -----------------------------------------------------
/// Eye offset above the AABB centre while standing: 1.62 − 0.9.
pub const EYE_HEIGHT: f32 = 0.72;
/// Eye offset above the AABB centre while sneaking: 1.27 − 0.75.
pub const EYE_HEIGHT_SNEAK: f32 = 0.52;
/// Half-height while sneaking (1.5-block-tall AABB).
const SNEAK_AABB_Y: f32 = 0.75;

pub const MOUSE_SENSITIVITY: f32 = 0.0022;

/// Inputs that stay constant for every physics tick in one system call.
struct TickContext<'a> {
    world: Option<&'a voxel_world::world::World>,
    flying: bool,
    in_water: bool,
    noclip: bool,
    sprinting: bool,
    speed: f32,
    jump: bool,
    sneak: bool,
    wish: Vec3,
    forward: Vec3,
}

#[derive(Default)]
struct TickOutcome {
    fall_distance: Option<f32>,
    jump_exhaustion: f32,
}

/// System: advances the player by `dt`, in fixed vanilla ticks.
pub fn movement_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let initial = {
        let transform = match world.get::<Transform>(player_entity) {
            Some(t) => *t,
            None => return,
        };
        let velocity = match world.get::<Velocity>(player_entity) {
            Some(v) => *v,
            None => return,
        };
        let aabb = match world.get::<Aabb>(player_entity) {
            Some(a) => *a,
            None => Aabb::default(),
        };
        let input = match world.get::<PlayerInput>(player_entity) {
            Some(i) => *i,
            None => return,
        };
        let state = match world.get::<PlayerState>(player_entity) {
            Some(s) => *s,
            None => return,
        };
        (transform, velocity, aabb, input, state)
    };

    let (mut transform, velocity, aabb_base, input, mut state) = initial;
    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);
    let hunger = world.get::<Hunger>(player_entity).copied();

    apply_look(&mut transform, &input);

    let (forward, _) = look_basis(&transform);
    let wish = wish_direction(&transform, &input);

    let food_allows_sprint = if game_mode.has_hunger() {
        hunger.map(|h| h.can_sprint()).unwrap_or(true)
    } else {
        true
    };
    let sprinting = input.sprinting
        && input.wish.z < -0.1
        && !input.sneaking
        && game_mode.can_sprint()
        && food_allows_sprint;
    let speed = if sprinting {
        SPRINT_SPEED
    } else if input.sneaking {
        SNEAK_SPEED
    } else {
        WALK_SPEED
    };
    let flying = (input.flying || game_mode == GameMode::Spectator) && game_mode.allows_flight();

    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let half = Vec3::new(
        aabb_base.half.x,
        if input.sneaking {
            SNEAK_AABB_Y
        } else {
            aabb_base.half.y
        },
        aabb_base.half.z,
    );

    let submersion = physics
        .as_ref()
        .map(|w| w.0.water_submersion(transform.pos - half, transform.pos + half))
        .unwrap_or(0.0);
    let in_water = submersion > 0.05 && !flying;

    state.eye_offset = if input.sneaking {
        EYE_HEIGHT_SNEAK
    } else {
        EYE_HEIGHT
    };
    if state.prev_eye_pos.length_squared() <= f32::EPSILON {
        state.prev_eye_pos = transform.pos + Vec3::new(0.0, state.eye_offset, 0.0);
    }

    let ctx = TickContext {
        world: physics.as_ref().map(|p| p.0.as_ref()),
        flying,
        in_water,
        noclip: game_mode == GameMode::Spectator,
        sprinting,
        speed,
        jump: input.jump,
        sneak: input.sneaking,
        wish,
        forward,
    };

    if input.jump {
        state.jump_buffer = JUMP_BUFFER_SECONDS;
    }

    let mut new_vel = velocity.lin;
    let mut fall_distance = None;
    let mut jump_exhaustion = 0.0;

    // Consume elapsed time into fixed physics ticks.
    state.physics_accum += dt;
    state.physics_accum = state.physics_accum.min(TICK_SECONDS * MAX_TICKS_PER_UPDATE as f32);
    let mut ticks = 0;
    while state.physics_accum >= TICK_SECONDS && ticks < MAX_TICKS_PER_UPDATE {
        let before = transform.pos;
        let outcome = step_tick(&ctx, &mut transform.pos, &mut new_vel, &mut state, half);
        state.prev_eye_pos = before + Vec3::new(0.0, state.eye_offset, 0.0);
        if outcome.fall_distance.is_some() {
            fall_distance = outcome.fall_distance;
        }
        if outcome.jump_exhaustion > 0.0 {
            jump_exhaustion = outcome.jump_exhaustion;
        }
        state.physics_accum -= TICK_SECONDS;
        ticks += 1;
    }
    state.interp_alpha = (state.physics_accum / TICK_SECONDS).clamp(0.0, 1.0);

    // Landing damage is queued after the tick loop so the ECS borrow is free.
    if let Some(distance) = fall_distance {
        if game_mode.takes_fall_damage() {
            let damage = (distance - 3.0).floor();
            if damage > 0.0 {
                log::debug!("Fall damage: distance={distance:.2}, damage={damage:.1}");
                if let Some(dq) = world.resource_mut::<DamageQueue>() {
                    dq.push(
                        player_entity,
                        DamageEvent {
                            source: DamageSource::Fall(distance),
                            amount: damage,
                        },
                    );
                }
            }
        }
    }
    if jump_exhaustion > 0.0 {
        if let Some(mut h) = hunger {
            h.add_exhaustion(jump_exhaustion);
            world.set(player_entity, h);
        }
    }

    state.in_water = in_water;
    update_view_state(&mut state, &new_vel, dt, input.sneaking, in_water, sprinting, speed);

    world.set(player_entity, transform);
    world.set(
        player_entity,
        Velocity {
            lin: new_vel,
            ang: Vec3::ZERO,
        },
    );
    world.set(player_entity, state);
}

/// Apply the frame's mouse delta to yaw/pitch.
fn apply_look(transform: &mut Transform, input: &PlayerInput) {
    if input.mouse_delta.0 == 0.0 && input.mouse_delta.1 == 0.0 {
        return;
    }
    let (mut yaw, mut pitch, _) = transform.rot.to_euler(glam::EulerRot::YXZ);
    yaw -= input.mouse_delta.0 * MOUSE_SENSITIVITY;
    pitch -= input.mouse_delta.1 * MOUSE_SENSITIVITY;
    let limit = std::f32::consts::FRAC_PI_2 - 0.01;
    pitch = pitch.clamp(-limit, limit);
    transform.rot = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
}

/// Flat forward/right vectors for the current look orientation.
fn look_basis(transform: &Transform) -> (Vec3, Vec3) {
    let (yaw, _, _) = transform.rot.to_euler(glam::EulerRot::YXZ);
    (
        Vec3::new(-yaw.sin(), 0.0, -yaw.cos()),
        Vec3::new(yaw.cos(), 0.0, -yaw.sin()),
    )
}

/// Horizontal wish direction from the input vector.
fn wish_direction(transform: &Transform, input: &PlayerInput) -> Vec3 {
    let (forward, right) = look_basis(transform);
    let wish = -forward * input.wish.z + right * input.wish.x;
    if wish.length_squared() > 0.0 {
        wish.normalize()
    } else {
        Vec3::ZERO
    }
}

/// One vanilla physics tick.
fn step_tick(
    ctx: &TickContext<'_>,
    pos: &mut Vec3,
    vel: &mut Vec3,
    state: &mut PlayerState,
    half: Vec3,
) -> TickOutcome {
    let mut out = TickOutcome::default();
    let was_on_ground = state.on_ground;

    // 1. Acceleration for this tick.
    if ctx.flying {
        let fly_speed = if ctx.sprinting { SPRINT_FLY_SPEED } else { FLY_SPEED };
        let target = ctx.wish * fly_speed;
        let target_y = if ctx.jump {
            fly_speed
        } else if ctx.sneak {
            -fly_speed
        } else {
            0.0
        };
        vel.x = target.x + (vel.x - target.x) * 0.5;
        vel.z = target.z + (vel.z - target.z) * 0.5;
        vel.y = target_y + (vel.y - target_y) * 0.5;
        state.fall_distance = 0.0;
    } else if ctx.in_water {
        let target = ctx.wish * SWIM_SPEED;
        vel.x = target.x + (vel.x - target.x) * WATER_FRICTION;
        vel.z = target.z + (vel.z - target.z) * WATER_FRICTION;
        if ctx.jump {
            vel.y = WATER_SWIM_UP;
        } else if ctx.sneak {
            vel.y = -WATER_SINK;
        } else {
            vel.y -= WATER_GRAVITY * TICK_SECONDS;
            vel.y *= WATER_FRICTION;
            vel.y = vel.y.max(-WATER_SINK);
        }
        state.fall_distance = 0.0;
    } else {
        if state.on_ground && state.jump_buffer > 0.0 {
            state.jump_buffer = 0.0;
            vel.y = JUMP_SPEED;
            state.on_ground = false;
            out.jump_exhaustion = if ctx.sprinting {
                vel.x += ctx.forward.x * SPRINT_JUMP_BOOST;
                vel.z += ctx.forward.z * SPRINT_JUMP_BOOST;
                exhaustion::SPRINT_JUMP
            } else {
                exhaustion::JUMP
            };
        } else if state.on_ground {
            let target = ctx.wish * ctx.speed;
            vel.x = target.x + (vel.x - target.x) * GROUND_FRICTION;
            vel.z = target.z + (vel.z - target.z) * GROUND_FRICTION;
        } else {
            vel.x += ctx.wish.x * AIR_ACCEL * TICK_SECONDS;
            vel.z += ctx.wish.z * AIR_ACCEL * TICK_SECONDS;
        }
    }

    // 2. Move, with collision substeps so fast movement cannot tunnel.
    let before = *pos;
    let delta = *vel * TICK_SECONDS;
    if ctx.noclip {
        *pos += delta;
        state.on_ground = false;
    } else if let Some(voxels) = ctx.world {
        let (new_pos, hits, on_ground) = move_with_collisions(voxels, *pos, half, delta);
        *pos = new_pos;
        if hits[0] || hits[1] {
            vel.x = 0.0;
        }
        if hits[2] || hits[3] {
            vel.y = 0.0;
        }
        if hits[4] || hits[5] {
            vel.z = 0.0;
        }
        state.on_ground = on_ground;

        let moved_xz = (new_pos.x - before.x).abs() > 1e-4
            || (new_pos.z - before.z).abs() > 1e-4;
        if ctx.sneak && state.on_ground && moved_xz && !has_ground_support(voxels, *pos, half) {
            pos.x = before.x;
            pos.z = before.z;
            vel.x = 0.0;
            vel.z = 0.0;
        }
    } else {
        *pos += delta;
    }

    // 3. Vanilla applies gravity and drag after movement. This ordering is
    // what produces the 1.2522-block jump apex and 78.4 blocks/s fall cap.
    if !ctx.flying && !ctx.in_water {
        vel.y = (vel.y - GRAVITY * TICK_SECONDS) * AIR_DRAG;
        vel.y = vel.y.max(-TERMINAL_VELOCITY);
        if !state.on_ground {
            vel.x *= AIR_FRICTION;
            vel.z *= AIR_FRICTION;
        }
    }

    // 4. Fall distance, landing and damage.
    if !ctx.flying && !ctx.in_water && vel.y < 0.0 {
        state.fall_distance += (before.y - pos.y).max(0.0);
    }
    if !was_on_ground && state.on_ground {
        out.fall_distance = Some(state.fall_distance);
        state.fall_distance = 0.0;
    } else if state.on_ground {
        state.fall_distance = 0.0;
    }

    state.jump_buffer = (state.jump_buffer - TICK_SECONDS).max(0.0);
    out
}

/// Move the AABB in substeps, returning the final position, hit mask and
/// whether the player ended on the ground.
fn move_with_collisions(
    world: &voxel_world::world::World,
    mut pos: Vec3,
    half: Vec3,
    delta: Vec3,
) -> (Vec3, [bool; 6], bool) {
    let steps = ((delta.length() / MAX_COLLISION_STEP).ceil() as u32).clamp(1, 16);
    let step = delta / steps as f32;
    let mut hits = [false; 6];
    let mut on_ground = false;

    for _ in 0..steps {
        let res = voxel_physics::swept_aabb(world, pos, half, step);
        for (i, hit) in res.hit.iter().enumerate() {
            hits[i] |= *hit;
        }
        pos = res.new_pos;
        if step.y < 0.0 {
            on_ground = res.on_ground;
        } else if step.y > 0.0 {
            on_ground = false;
        }
    }

    (pos, hits, on_ground)
}

/// Whether there is any solid ground under the AABB's footprint. Used by
/// sneaking to prevent walking off ledges.
fn has_ground_support(world: &voxel_world::world::World, pos: Vec3, half: Vec3) -> bool {
    let y = (pos.y - half.y - 0.05).floor() as i32;
    let eps = 0.01;
    for x in [pos.x - half.x + eps, pos.x + half.x - eps] {
        for z in [pos.z - half.z + eps, pos.z + half.z - eps] {
            if world.is_solid(x.floor() as i32, y, z.floor() as i32) {
                return true;
            }
        }
    }
    world.is_solid(pos.x.floor() as i32, y, pos.z.floor() as i32)
}

/// Update the visual-only state (bob phase, mining swing decay).
fn update_view_state(
    state: &mut PlayerState,
    velocity: &Vec3,
    dt: f32,
    sneaking: bool,
    in_water: bool,
    sprinting: bool,
    speed: f32,
) {
    let horiz_speed = Vec3::new(velocity.x, 0.0, velocity.z).length();
    if state.on_ground && horiz_speed > 0.1 && !sneaking && !in_water {
        let speed_factor = (horiz_speed / speed).min(1.0);
        let bob_freq = if sprinting { 18.0 } else { 12.0 };
        state.bob_phase += dt * speed_factor * bob_freq;
    } else {
        state.bob_phase *= 1.0 - 5.0 * dt;
    }

    if state.mining_swing > 0.0 {
        state.mining_swing -= dt / 0.4;
        if state.mining_swing < 0.0 {
            state.mining_swing = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vanilla_speed_constants() {
        assert!((WALK_SPEED - 4.317).abs() < 1e-3);
        assert!((SPRINT_SPEED - 5.612).abs() < 1e-3);
        assert!((SNEAK_SPEED - 1.295).abs() < 1e-3);
        assert!((JUMP_SPEED - 8.4).abs() < 1e-4);
        assert!((GRAVITY - 32.0).abs() < 1e-4);
        assert!((TICK_SECONDS - 0.05).abs() < 1e-6);
    }

    #[test]
    fn tick_sequence_reaches_vanilla_jump_height() {
        let mut velocity = JUMP_SPEED;
        let mut height = 0.0f32;
        while velocity > 0.0 {
            height += velocity * TICK_SECONDS;
            velocity = (velocity - GRAVITY * TICK_SECONDS) * AIR_DRAG;
        }
        assert!(
            (height - 1.2522).abs() < 1e-3,
            "jump height should be 1.2522 blocks, got {height}"
        );
    }

    #[test]
    fn eye_offsets_match_vanilla() {
        assert!((EYE_HEIGHT + 0.9 - 1.62).abs() < 1e-4);
        assert!((EYE_HEIGHT_SNEAK + SNEAK_AABB_Y - 1.27).abs() < 1e-4);
    }

    #[test]
    fn ground_friction_matches_vanilla() {
        // 0.6 slipperiness × 0.91 global drag.
        assert!((GROUND_FRICTION - 0.546).abs() < 1e-5);
    }
}
