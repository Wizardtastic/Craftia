//! `TestSim` — headless simulation harness for ECS integration tests.
//!
//! Wraps [`Simulation`] with ergonomic builders and assertion helpers
//! so `#[test]` functions can drive a fixed-timestep ECS world without
//! a GPU, a window, or an event loop. Each `tick(N)` call runs `N`
//! fixed steps (1/60s each), capped at 1000 to prevent runaway loops.
//!
//! Default world: a 3├ù3 grid of chunks centered at the origin, each
//! with a stone floor at local `y=0`. Players spawn at `[0.5, 2.0, 0.5]`
//! by default (just above the floor); `spawn_player_at(pos)` overrides.
//!
//! ```ignore
//! use voxel_engine::test_sim::TestSim;
//! use voxel_game::input::Action;
//!
//! #[test]
//! fn player_falls_with_gravity() {
//!     let mut sim = TestSim::new().spawn_player_at([0.0, 20.0, 0.0]);
//!     sim.tick(60); // 1 second at 60Hz
//!     assert!(sim.player_pos().unwrap().y < 19.9);
//! }
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use glam::Vec3;
use voxel_core::ChunkPos;
use voxel_ecs::{Component, Entity, World as EcsWorld};
use voxel_game::input::Action;
use voxel_game::PlayerState;
use voxel_world::{Chunk, World};

use crate::sim::Simulation;

/// Cap on per-tick step count to prevent runaway loops in tests.
const TICK_CAP: u32 = 1000;

/// Default spawn position: chunk centre, just above the stone floor.
const DEFAULT_SPAWN: [f32; 3] = [0.5, 2.0, 0.5];

/// Build a flat-stone 3├ù3 chunk grid centered at the origin, returning
/// the populated `Arc<World>`. Mirrors the `setup_world()` pattern from
/// `voxel-world/src/water.rs:setup_world()` so test worlds are
/// consistent across the project.
fn build_flat_world(seed: i32) -> Arc<World> {
    let world = World::new(seed);
    let reg = world.registry();
    let stone = reg
        .id_of("stone")
        .expect("stone must be in the default block registry");
    for cx in -1..=1 {
        for cz in -1..=1 {
            let cp = ChunkPos::new(cx, 0, cz);
            let mut chunk = Chunk::new(cp);
            for lx in 0..voxel_core::CHUNK_SIZE {
                for lz in 0..voxel_core::CHUNK_SIZE {
                    chunk.set(lx, 0, lz, stone);
                }
            }
            world.insert_chunk(cp, chunk);
        }
    }
    world
}

/// Headless simulation harness for ECS integration tests.
pub struct TestSim {
    sim: Simulation,
    /// Held input actions carried between `tick` calls. Projected into
    /// an `InputSnapshot` at the top of every `step` inside `tick`.
    held: HashSet<Action>,
}

impl Default for TestSim {
    fn default() -> Self {
        Self::new()
    }
}

impl TestSim {
    /// Create a fresh `TestSim` with a flat-stone 3├ù3 world and the
    /// player spawned at `[0.5, 2.0, 0.5]`. Use the chainable
    /// `seed`/`world`/`spawn_player_at` builders to customize.
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Override the world-generation seed (default 0). Convenience
    /// around `with_world` that builds a 3├ù3 flat-stone world with
    /// the requested seed.
    pub fn with_seed(seed: i32) -> Self {
        let world = build_flat_world(seed);
        Self::with_world(world)
    }

    /// Replace the auto-generated flat world with a caller-supplied
    /// `Arc<World>`. Useful for tests that need a specific terrain
    /// (caves, water, schematics) pre-loaded.
    pub fn with_world(world: Arc<World>) -> Self {
        let sim = Simulation::new(
            world,
            Vec3::from(DEFAULT_SPAWN),
            /* with_timing = */ false,
        );
        Self {
            sim,
            held: HashSet::new(),
        }
    }

    /// Override the player's spawn position. The Y component should
    /// be just above a solid surface (e.g. 2.0 for a stone floor at
    /// y=1) for the player to land and stand.
    pub fn spawn_player_at(mut self, pos: [f32; 3]) -> Self {
        self.sim.set_player_pos(Vec3::from(pos));
        self
    }

    /// Hold an input action. The action is projected into the
    /// `InputSnapshot` consumed by `input_system` for every subsequent
    /// `tick`.
    pub fn hold(&mut self, action: Action) {
        self.held.insert(action);
    }

    /// Release a previously held input action. No-op if the action
    /// wasn't held.
    pub fn release(&mut self, action: Action) {
        self.held.remove(&action);
    }

    /// Clear all held inputs. Mirrors the engine's pause-state reset.
    pub fn clear_input(&mut self) {
        self.held.clear();
    }

    /// Run `count` fixed-timestep ticks (1/60s each). Capped at
    /// `TICK_CAP` (1000) so a runaway test that creates an infinite
    /// loop still completes in finite time.
    pub fn tick(&mut self, count: u32) {
        let steps = count.min(TICK_CAP);
        for _ in 0..steps {
            let snap = build_snapshot(&self.held, self.sim.player_flying());
            self.sim.set_player_input(snap);
            self.sim.step(crate::sim::FIXED_DT);
        }
    }

    // -----------------------------------------------------------------
    // Player state accessors
    // -----------------------------------------------------------------

    /// Player world-space position, or `None` if the player entity is
    /// missing.
    pub fn player_pos(&self) -> Option<Vec3> {
        self.sim.player_pos()
    }

    /// Player linear velocity.
    pub fn player_vel(&self) -> Option<Vec3> {
        self.sim.player_vel()
    }

    /// Player `PlayerState` (eye offset, on-ground flags, etc.).
    pub fn player_state(&self) -> Option<PlayerState> {
        self.sim.player_state()
    }

    /// Whether the player is currently in fly mode.
    pub fn player_flying(&self) -> bool {
        self.sim.player_flying()
    }

    /// Player entity handle. `None` for an empty ECS.
    pub fn player_entity(&self) -> Option<Entity> {
        self.sim.player_entity()
    }

    // -----------------------------------------------------------------
    // ECS accessors
    // -----------------------------------------------------------------

    /// Read-only borrow of the underlying ECS world.
    pub fn ecs_world(&self) -> &EcsWorld {
        self.sim.ecs_world()
    }

    /// Mutable borrow of the underlying ECS world. Required for tests
    /// that toggle components directly (e.g. flip `PlayerInput::flying`).
    pub fn ecs_world_mut(&mut self) -> &mut EcsWorld {
        self.sim.ecs_world_mut()
    }

    /// True iff the entity has a component of type `T`.
    pub fn has_component<T: Component>(&self, entity: Entity) -> bool {
        self.ecs_world().has::<T>(entity)
    }

    /// Read-only borrow of component `T` on `entity`.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.ecs_world().get::<T>(entity)
    }

    // -----------------------------------------------------------------
    // Voxel world accessors
    // -----------------------------------------------------------------

    /// Read-only borrow of the shared voxel `World` for block / chunk
    /// assertions.
    pub fn voxel_world(&self) -> &Arc<World> {
        self.sim.voxel_world()
    }
}

/// Project the held-action set into the per-step `InputSnapshot` the
/// `input_system` reads. Mouse delta is zero (no mouse in tests).
fn build_snapshot(held: &HashSet<Action>, flying: bool) -> voxel_game::InputSnapshot {
    voxel_game::InputSnapshot {
        forward: held.contains(&Action::Forward),
        back: held.contains(&Action::Back),
        left: held.contains(&Action::Left),
        right: held.contains(&Action::Right),
        jump: held.contains(&Action::Jump),
        sneak: held.contains(&Action::Sneak),
        sprint: held.contains(&Action::Sprint),
        flying,
        mouse_delta: (0.0, 0.0),
        mining: held.contains(&Action::Attack),
        use_item: false,
    }
}
