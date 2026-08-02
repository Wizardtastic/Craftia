//! `Simulation` — headless ECS-driven gameplay step.
//!
//! Owns the ECS world, the compiled system schedule, and a clone of the
//! `Arc<World>` the simulation should query blocks from. EngineApp composes
//! a `Simulation` alongside its rendering / input machinery; the
//! `TestSim` test harness composes one without a window. The split lets
//! gameplay logic run standalone — no GPU, no winit, no event loop.
//!
//! Tick flow:
//!
//! ```text
//!   frame()                      tick(N) (TestSim)
//!     Γöé                              Γöé
//!     Γû╝                              Γû╝
//!   set_player_input(snap)        hold/release
//!     Γöé                              Γöé
//!     Γû╝                              Γû╝
//!   tick_fixed(frame_dt)          tick_fixed(FIXED_DT ├ù N)
//!     Γöé                              Γöé
//!     Γöö─Γû║  while acc >= FIXED_DT: ─Γöÿ
//!           step(FIXED_DT):
//!             sched.run(&mut ecs, FIXED_DT)
//!             world.tick_water(FIXED_DT)
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use glam::Vec3;
use voxel_core::ChunkPos;
use voxel_ecs::{Entity, FnSystem, SystemSchedule, World as EcsWorld};
use voxel_game::{
    animation_system,
    armor_system,
    drowning_system,
    environmental_damage_system,
    health_system,
    held_item_system,
    hierarchy_system,
    hunger_system,
    input_system,
    item_pickup_system,
    lifecycle_system,
    movement_system,
    progressive_mining_system,
    property_animation_system,
    regeneration_system,
    xp_collection_system,
    Aabb,
    AirSupply,
    AnimationDataResource,
    AnimationPlayer,
    BoneTransforms,
    CameraOwner,
    CameraResource,
    ChildMapResource,
    Children,
    DamageQueue,
    DifficultyResource,
    DrowningState,
    EatingState,
    Experience,
    GameMode,
    GameTimeResource,
    // Survival mode components
    Health,
    HeldBlock,
    HotbarResource,
    Hunger,
    InputResource,
    InputSnapshot,
    Mesh,
    MiningProgress,
    ModelRef,
    Parent,
    PhysicsWorldRes,
    PlayerEntity,
    PlayerInput,
    PlayerLookTarget,
    PlayerState,
    RegenState,
    Transform,
    Velocity,
    ViewMode,
};
use voxel_world::World;

/// Fixed-timestep simulation step used by both the ECS schedule and any
/// standalone physics updates inside the frame loop.
pub(crate) const FIXED_DT: f64 = 1.0 / 60.0;

/// Cap on the per-frame accumulator to avoid spiral-of-death after long
/// pauses (debugger attach, alt-tab, etc.).
const MAX_ACCUM: f64 = 0.25;

/// Maximum number of fixed steps the schedule will run per `tick_fixed`
/// call. Eight at 60Hz = 133ms of catch-up per frame, which is enough
/// to ride out a stutter without snowballing.
const MAX_STEPS_PER_FRAME: u32 = 8;

/// Headless gameplay step. Owns the ECS world, system schedule, and a
/// clone of the voxel `World` for collision queries.
///
/// `EngineApp` composes one for its `frame()` loop; the `TestSim`
/// harness composes one for unit tests. The struct is `pub` so the
/// engine can hand the `ecs_world` to debug overlays; the test harness
/// borrows the world, schedule, and player accessors through `&self`.
///
/// Note: the day/night clock (game time + day length) lives on
/// `GamePlayState` instead of here. The engine still drives the cycle
/// from `frame()`'s per-frame `dt`; this struct is intentionally
/// focused on ECS tick + water + input plumbing, not the broader
/// game-state clock.
pub struct Simulation {
    /// ECS world owning entities (player + debug entities), resources
    /// (camera, input, player handle), and archetype storage.
    pub ecs_world: EcsWorld,
    /// Compiled gameplay system schedule. `None` only during late-init
    /// scaffolding paths; populated by `Simulation::new`.
    pub schedule: Option<SystemSchedule>,
    /// Clone of the shared voxel `World` handle used for collision
    /// queries. The engine also holds its own `Arc<World>` for the
    /// chunk streamer; both reference the same underlying data.
    pub world: Arc<World>,
    /// Internal accumulator that the frame loop feeds via
    /// `tick_fixed`. Owned by `Simulation` so the engine no longer
    /// threads a per-frame double through `EngineInputState`.
    sim_accumulator: f64,
}

impl Simulation {
    /// Build a fresh simulation: ECS, player, schedule.
    ///
    /// Takes the **shared** `Arc<World>` by reference so the engine
    /// can wire its chunk streamer to the same data the simulation
    /// collision-queries. The `TestSim` harness owns its own
    /// `Arc<World>` and passes a clone.
    ///
    /// `spawn_pos` is the player's AABB centre in world space. The
    /// caller is responsible for picking a sensible Y (typically
    /// just above a stone floor).
    ///
    /// `with_timing` opts the schedule in to per-system wall-clock
    /// timing so the F6 profiler overlay has data to display.
    pub fn new(world: Arc<World>, spawn_pos: Vec3, with_timing: bool) -> Self {
        // ECS world + default camera + default input snapshot.
        let mut ecs_world = EcsWorld::new();
        let mut initial_camera = voxel_core::Camera::default();
        initial_camera.pos = spawn_pos + glam::Vec3::new(0.0, voxel_game::EYE_HEIGHT, 0.0);
        ecs_world.insert_resource(CameraResource(initial_camera));
        ecs_world.insert_resource(InputResource(InputSnapshot::default()));
        // Install the physics world once. The world Arc is constant for
        // a `Simulation`'s lifetime; `set_player_input` does not need
        // to re-insert it on every frame.
        ecs_world.insert_resource(PhysicsWorldRes(world.clone()));
        ecs_world.insert_resource(ChildMapResource::default());

        // Debug formatters — must run before any spawn so the
        // archetype-creation `ensure_registered` lookup has the
        // short-name fns ready.
        ecs_world.register_debug_formatter::<Transform>();
        ecs_world.register_debug_formatter::<Velocity>();
        ecs_world.register_debug_formatter::<Aabb>();
        ecs_world.register_debug_formatter::<PlayerInput>();
        ecs_world.register_debug_formatter::<PlayerState>();
        ecs_world.register_debug_formatter::<CameraOwner>();
        ecs_world.register_debug_formatter::<CameraResource>();
        ecs_world.register_debug_formatter::<InputResource>();
        ecs_world.register_debug_formatter::<PlayerEntity>();
        ecs_world.register_debug_formatter::<voxel_game::DebugEntityMarker>();
        ecs_world.register_debug_formatter::<Parent>();
        ecs_world.register_debug_formatter::<Children>();
        ecs_world.register_debug_formatter::<Mesh>();
        ecs_world.register_debug_formatter::<HeldBlock>();
        ecs_world.register_debug_formatter::<AnimationPlayer>();
        ecs_world.register_debug_formatter::<BoneTransforms>();
        ecs_world.register_debug_formatter::<ModelRef>();
        ecs_world.register_debug_formatter::<Health>();
        ecs_world.register_debug_formatter::<AirSupply>();
        ecs_world.register_debug_formatter::<RegenState>();
        ecs_world.register_debug_formatter::<GameMode>();
        ecs_world.register_debug_formatter::<Hunger>();
        ecs_world.register_debug_formatter::<EatingState>();
        ecs_world.register_debug_formatter::<Experience>();
        ecs_world.register_debug_formatter::<DrowningState>();
        ecs_world.insert_resource(AnimationDataResource::default());
        ecs_world.insert_resource(ViewMode::default());

        // Survival mode resources.
        ecs_world.insert_resource(DamageQueue::new());
        ecs_world.insert_resource(DifficultyResource::default());
        ecs_world.insert_resource(GameTimeResource::default());
        ecs_world.insert_resource(PlayerLookTarget::default());
        ecs_world.insert_resource(HotbarResource::default());

        // Spawn the player with the full component set the gameplay
        // systems expect.
        let player_entity = ecs_world.spawn((
            Transform {
                pos: spawn_pos,
                rot: glam::Quat::IDENTITY,
            },
            Velocity::default(),
            Aabb::default(),
            PlayerInput::default(),
            PlayerState::default(),
            CameraOwner,
            HeldBlock::default(),
        ));
        // Add survival mode components separately (Bundle supports max 8).
        ecs_world.set(player_entity, Health::default());
        ecs_world.set(player_entity, AirSupply::default());
        ecs_world.set(player_entity, RegenState::default());
        ecs_world.set(player_entity, GameMode::default());
        ecs_world.set(player_entity, MiningProgress::default());
        ecs_world.set(player_entity, Hunger::default());
        ecs_world.set(player_entity, EatingState::default());
        ecs_world.set(player_entity, Experience::default());
        ecs_world.set(player_entity, DrowningState::default());
        ecs_world.insert_resource(PlayerEntity(Some(player_entity)));

        // Two debug entities ┬▒5 blocks East/West so the inspector has
        // something to pin/cycle beyond the player.
        voxel_game::spawn_debug_entity(&mut ecs_world, spawn_pos + glam::Vec3::new(5.0, 2.0, 0.0));
        voxel_game::spawn_debug_entity(&mut ecs_world, spawn_pos + glam::Vec3::new(-5.0, 2.0, 0.0));

        // Build the gameplay system schedule.
        let mut schedule = SystemSchedule::new()
            .add_system(FnSystem::new("InputSystem", input_system))
            .add_system(FnSystem::new("MovementSystem", movement_system))
            .add_system(FnSystem::new("HungerSystem", hunger_system))
            .add_system(FnSystem::new("DrowningSystem", drowning_system))
            .add_system(FnSystem::new(
                "EnvironmentalDamageSystem",
                environmental_damage_system,
            ))
            .add_system(FnSystem::new("HealthSystem", health_system))
            .add_system(FnSystem::new("RegenerationSystem", regeneration_system))
            .add_system(FnSystem::new("ArmorSystem", armor_system))
            .add_system(FnSystem::new(
                "ProgressiveMiningSystem",
                progressive_mining_system,
            ))
            .add_system(FnSystem::new("ItemPickupSystem", item_pickup_system))
            .add_system(FnSystem::new("XpCollectionSystem", xp_collection_system))
            .add_system(FnSystem::new("HierarchySystem", hierarchy_system))
            .add_system(FnSystem::new("AnimationSystem", animation_system))
            .add_system(FnSystem::new(
                "PropertyAnimationSystem",
                property_animation_system,
            ))
            .add_system(FnSystem::new("HeldItemSystem", held_item_system))
            .add_system(FnSystem::new("LifecycleSystem", lifecycle_system));
        if with_timing {
            schedule = schedule.with_timing();
        }

        Self {
            ecs_world,
            schedule: Some(schedule),
            world,
            sim_accumulator: 0.0,
        }
    }

    /// True iff the simulation's wall-clock per-system timing is
    /// enabled. The engine uses this to decide whether to copy
    /// `last_frame_timings()` into the profiler state.
    pub fn timings_enabled(&self) -> bool {
        self.schedule.as_ref().is_some_and(|s| s.timings_enabled())
    }

    /// Owned copy of `(system_name, elapsed_micros)` for the most
    /// recent `step`. Returns an empty `Vec` when timing is disabled
    /// or no step has run yet.
    pub fn last_frame_timings(&self) -> Vec<(String, u64)> {
        self.schedule
            .as_ref()
            .map(|s| s.last_frame_timings())
            .unwrap_or_default()
    }

    // -----------------------------------------------------------------
    // Tick primitives
    // -----------------------------------------------------------------

    /// Project a per-frame input snapshot into the ECS so the next
    /// `step` (or every step in an upcoming `tick_fixed` loop) sees
    /// the same action state. The snap is latched onto
    /// `InputResource` for every subsequent fixed step in the
    /// accumulator loop, so a single `set_player_input` call covers
    /// an entire `tick_fixed` run.
    ///
    /// The `PhysicsWorldRes` is installed once in [`Simulation::new`]
    /// and never re-inserted — the world Arc is constant for a
    /// `Simulation`'s lifetime.
    pub fn set_player_input(&mut self, snap: InputSnapshot) {
        // Accumulate any unconsumed mouse delta from a previous frame that had
        // zero fixed steps (happens when FPS > 60). Without accumulation, the
        // delta would be silently dropped when `insert_resource` overwrites the
        // old `InputResource`.
        let mut snap = snap;
        let prev_delta = {
            self.ecs_world
                .resource::<InputResource>()
                .map(|r| r.0.mouse_delta)
                .unwrap_or((0.0, 0.0))
        };
        snap.mouse_delta.0 += prev_delta.0;
        snap.mouse_delta.1 += prev_delta.1;
        self.ecs_world.insert_resource(InputResource(snap));
    }

    /// Run one fixed-timestep step. Inserts nothing — the caller is
    /// expected to have called `set_player_input` once for the
    /// surrounding frame. Schedules the gameplay systems and ticks
    /// the water simulation.
    ///
    /// Returns the set of chunks whose blocks changed during the
    /// water tick, so the engine can hand them to the chunk streamer
    /// for remeshing. The test harness can ignore the return.
    pub fn step(&mut self, dt: f64) -> HashSet<ChunkPos> {
        if let Some(sched) = self.schedule.as_mut() {
            sched.run(&mut self.ecs_world, dt as f32);
        }
        // Drive incremental water flow. Internal accumulator means
        // the rate is governed by elapsed wall-clock seconds, not by
        // step frequency — calling per fixed step vs per frame is
        // equivalent for the simulation itself.
        self.world.tick_water(dt as f32)
    }

    /// Accumulate `frame_dt` and run as many fixed steps as needed
    /// to catch up. Caps both the accumulator (0.25s, anti-spiral)
    /// and the per-frame step count (8, anti-thrash).
    ///
    /// Returns the union of chunks affected by water flow across
    /// every step in the loop, so the engine can request remeshes
    /// once per frame.
    pub fn tick_fixed(&mut self, frame_dt: f64) -> HashSet<ChunkPos> {
        self.sim_accumulator += frame_dt;
        if self.sim_accumulator > MAX_ACCUM {
            self.sim_accumulator = MAX_ACCUM;
        }
        let mut affected = HashSet::new();
        let mut steps = 0;
        while self.sim_accumulator >= FIXED_DT && steps < MAX_STEPS_PER_FRAME {
            affected.extend(self.step(FIXED_DT));
            self.sim_accumulator -= FIXED_DT;
            steps += 1;
        }
        affected
    }

    // -----------------------------------------------------------------
    // Player accessors (mirror of the legacy `GamePlayState` helpers)
    // -----------------------------------------------------------------

    /// Resolve the player entity handle from the ECS resource.
    pub fn player_entity(&self) -> Option<Entity> {
        self.ecs_world.resource::<PlayerEntity>().and_then(|p| p.0)
    }

    /// Read the player's world-space position from the ECS, if any.
    pub fn player_pos(&self) -> Option<Vec3> {
        let e = self.player_entity()?;
        self.ecs_world.get::<Transform>(e).map(|t| t.pos)
    }

    /// Read the player's linear velocity from the ECS, if any.
    pub fn player_vel(&self) -> Option<Vec3> {
        let e = self.player_entity()?;
        self.ecs_world.get::<Velocity>(e).map(|v| v.lin)
    }

    /// Read the player's `PlayerState` component, if any.
    pub fn player_state(&self) -> Option<PlayerState> {
        let e = self.player_entity()?;
        self.ecs_world.get::<PlayerState>(e).copied()
    }

    /// Read whether the player is currently flying (ECS state).
    pub fn player_flying(&self) -> bool {
        self.player_entity()
            .and_then(|e| self.ecs_world.get::<PlayerInput>(e))
            .map(|i| i.flying)
            .unwrap_or(false)
    }

    /// Set the player's world-space position. Also clears vertical
    /// velocity so the player doesn't immediately fall after a
    /// teleport.
    pub fn set_player_pos(&mut self, pos: Vec3) {
        if let Some(e) = self.player_entity() {
            if let Some(t) = self.ecs_world.get_mut::<Transform>(e) {
                t.pos = pos;
            }
            if let Some(v) = self.ecs_world.get_mut::<Velocity>(e) {
                v.lin = Vec3::ZERO;
            }
        }
    }

    /// Set the player's flying flag in the ECS.
    pub fn set_player_flying(&mut self, flying: bool) {
        if let Some(e) = self.player_entity() {
            if let Some(input) = self.ecs_world.get_mut::<PlayerInput>(e) {
                input.flying = flying;
            }
        }
    }

    /// Compute the current eye position (head height) by reading the
    /// player's `PlayerState` eye offset from the ECS. Mirrors the
    /// legacy `GamePlayState::player_eye_pos` helper.
    pub fn player_eye_pos(&self) -> Option<Vec3> {
        let e = self.player_entity()?;
        let t = self.ecs_world.get::<Transform>(e)?;
        let s = self
            .ecs_world
            .get::<PlayerState>(e)
            .copied()
            .unwrap_or_default();
        Some(t.pos + glam::Vec3::new(0.0, s.eye_offset, 0.0))
    }

    /// Read the current `CameraResource` value. Mirrors the legacy
    /// `GamePlayState::player_camera` helper.
    pub fn player_camera(&self) -> Option<voxel_core::Camera> {
        self.ecs_world.resource::<CameraResource>().map(|c| c.0)
    }

    /// Overwrite the player's camera (position + yaw/pitch) in the ECS.
    /// Used by the auto-capture camera override for deterministic
    /// verification screenshots.
    pub fn set_player_camera(&mut self, cam: voxel_core::Camera) {
        if let Some(r) = self.ecs_world.resource_mut::<CameraResource>() {
            r.0 = cam;
        }
    }

    /// Cloned `(Transform, PlayerState)` for the player, used by the
    /// engine's per-frame camera projection (avoids two `&EcsWorld`
    /// borrows in a single expression).
    pub fn player_transform_state(&self) -> Option<(Transform, PlayerState)> {
        let e = self.player_entity()?;
        let t = self.ecs_world.get::<Transform>(e)?.clone();
        let s = self.ecs_world.get::<PlayerState>(e)?.clone();
        Some((t, s))
    }

    /// Reset the fixed-step accumulator to zero. The engine calls
    /// this from `enter_pause` so a long pause doesn't unleash a
    /// burst of catch-up steps on resume.
    pub fn reset_accumulator(&mut self) {
        self.sim_accumulator = 0.0;
    }

    /// Read-only borrow of the ECS world (for queries / inspectors).
    pub fn ecs_world(&self) -> &EcsWorld {
        &self.ecs_world
    }

    /// Mutable borrow of the ECS world. Use sparingly — the schedule
    /// owns most mutations; this is for tests and the inspector.
    pub fn ecs_world_mut(&mut self) -> &mut EcsWorld {
        &mut self.ecs_world
    }

    /// Borrow the shared voxel `World` (read-only) for block
    /// assertions in tests.
    pub fn voxel_world(&self) -> &Arc<World> {
        &self.world
    }
}
