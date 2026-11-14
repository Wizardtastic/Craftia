//! In-game water integration tests.
//!
//! These drive the real `Simulation` schedule (movement, drowning, item
//! physics) against a world with actual water columns, rather than the flat
//! stone floor used by the generic `TestSim` tests.

use std::sync::Arc;

use glam::Vec3;
use voxel_core::ChunkPos;
use voxel_engine::test_sim::TestSim;
use voxel_game::input::Action;
use voxel_game::{ItemEntity, Transform};
use voxel_world::gen::TerrainGenerator;
use voxel_world::water::{WaterColumnMeta, WaterKind};
use voxel_world::{Chunk, World};

/// Generate the real world column at the seeded lake near (-736, -252),
/// including both the bed chunk and the surface chunk.
fn generated_lake_world() -> Arc<World> {
    let world = World::new(1337);
    let gen = TerrainGenerator::new(1337);
    for cy in [3, 4] {
        let cp = ChunkPos::new(-46, cy, -16);
        let mut chunk = Chunk::new(cp);
        let columns = gen.column_table(voxel_core::math::chunk_origin(cp));
        {
            let reg = world.registry();
            gen.generate(&mut chunk, &reg, &columns).unwrap();
        }
        world.insert_chunk(cp, chunk);
    }
    world
}

const WATER_BOTTOM: i32 = 1;
const WATER_TOP: i32 = 4;
const WATER_SURFACE: f32 = (WATER_TOP + 1) as f32;

fn water_world(kind: WaterKind, flow_speed: u8) -> Arc<World> {
    let world = World::new(0);
    let reg = world.registry();
    let stone = reg.id_of("stone").unwrap();
    let water = reg.id_of("water").unwrap();

    for cx in -1..=1 {
        for cz in -1..=1 {
            let cp = ChunkPos::new(cx, 0, cz);
            let mut chunk = Chunk::new(cp);
            for lx in 0..voxel_core::CHUNK_SIZE {
                for lz in 0..voxel_core::CHUNK_SIZE {
                    chunk.set(lx, 0, lz, stone);
                }
            }
            if cx == 0 && cz == 0 {
                let meta = WaterColumnMeta::new(kind, 255, 0, flow_speed, 0);
                for lx in 0..voxel_core::CHUNK_SIZE {
                    for lz in 0..voxel_core::CHUNK_SIZE {
                        for y in WATER_BOTTOM..=WATER_TOP {
                            chunk.set(lx, y, lz, water);
                        }
                        chunk.set_water_meta(lx, lz, meta);
                    }
                }
            }
            world.insert_chunk(cp, chunk);
        }
    }
    world
}

/// A lake whose water spans two vertical chunks (world y 1..=21), with the
/// surface in the upper chunk. Its bed is deliberately below the surface
/// chunk's floor so buoyancy must use the full column.
fn deep_lake_world() -> Arc<World> {
    let world = World::new(0);
    let reg = world.registry();
    let stone = reg.id_of("stone").unwrap();
    let water = reg.id_of("water").unwrap();

    for cx in -1..=1 {
        for cz in -1..=1 {
            let cp = ChunkPos::new(cx, 0, cz);
            let mut chunk = Chunk::new(cp);
            for lx in 0..voxel_core::CHUNK_SIZE {
                for lz in 0..voxel_core::CHUNK_SIZE {
                    chunk.set(lx, 0, lz, stone);
                }
            }
            if cx == 0 && cz == 0 {
                let meta = WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0);
                for lx in 0..voxel_core::CHUNK_SIZE {
                    for lz in 0..voxel_core::CHUNK_SIZE {
                        for y in 1..=15 {
                            chunk.set(lx, y, lz, water);
                        }
                        chunk.set_water_meta(lx, lz, meta);
                    }
                }
            }
            world.insert_chunk(cp, chunk);
        }
    }

    // Upper slice (world y 16..20) holding the surface.
    let cp = ChunkPos::new(0, 1, 0);
    let mut upper = Chunk::new(cp);
    let meta = WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0);
    for lx in 0..voxel_core::CHUNK_SIZE {
        for lz in 0..voxel_core::CHUNK_SIZE {
            for y in 0..=4 {
                upper.set(lx, y, lz, water);
            }
            upper.set_water_meta(lx, lz, meta);
        }
    }
    world.insert_chunk(cp, upper);
    world
}

/// A lake in the surface chunk with a separate cave aquifer twelve blocks
/// below it. The cave pool must remain swimmable even though the lake body is
/// the topmost entry in the same world column.
fn lake_over_cave_world() -> Arc<World> {
    let world = World::new(0);
    let reg = world.registry();
    let stone = reg.id_of("stone").unwrap();
    let water = reg.id_of("water").unwrap();

    for cx in -1..=1 {
        for cz in -1..=1 {
            let cp = ChunkPos::new(cx, 0, cz);
            let mut chunk = Chunk::new(cp);
            for lx in 0..voxel_core::CHUNK_SIZE {
                for lz in 0..voxel_core::CHUNK_SIZE {
                    chunk.set(lx, 0, lz, stone);
                }
            }
            if cx == 0 && cz == 0 {
                let meta = WaterColumnMeta::new(WaterKind::Aquifer, 255, 0, 0, 0);
                for lx in 0..voxel_core::CHUNK_SIZE {
                    for lz in 0..voxel_core::CHUNK_SIZE {
                        for y in 1..=4 {
                            chunk.set(lx, y, lz, water);
                        }
                        chunk.set_water_meta(lx, lz, meta);
                    }
                }
            }
            world.insert_chunk(cp, chunk);
        }
    }

    let cp = ChunkPos::new(0, 1, 0);
    let mut upper = Chunk::new(cp);
    let meta = WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0);
    for lx in 0..voxel_core::CHUNK_SIZE {
        for lz in 0..voxel_core::CHUNK_SIZE {
            for y in 0..=4 {
                upper.set(lx, y, lz, water);
            }
            upper.set_water_meta(lx, lz, meta);
        }
    }
    world.insert_chunk(cp, upper);
    world
}

fn spawn_item(sim: &mut TestSim, pos: Vec3) -> voxel_ecs::Entity {
    let stone = sim.voxel_world().registry().id_of("stone").unwrap();
    sim.ecs_world_mut().spawn((
        Transform {
            pos,
            rot: glam::Quat::IDENTITY,
        },
        ItemEntity::new(stone, 1),
    ))
}

#[test]
fn player_floats_at_the_lake_surface() {
    let world = water_world(WaterKind::Lake, 0);
    let mut sim = TestSim::with_world(world).spawn_player_at([8.5, 6.5, 8.5]);
    sim.tick(180);

    let pos = sim.player_pos().unwrap();
    assert!(
        (pos.y - (WATER_SURFACE - 0.55)).abs() < 0.35,
        "player should settle just under the lake surface, y={}",
        pos.y
    );
    assert!(
        (pos.x - 8.5).abs() < 0.05 && (pos.z - 8.5).abs() < 0.05,
        "a still lake must not drift the player, pos={pos:?}"
    );
    assert!(sim.player_state().unwrap().in_water);
}

#[test]
fn player_floats_in_a_lake_spanning_two_vertical_chunks() {
    let world = deep_lake_world();
    let mut sim = TestSim::with_world(world).spawn_player_at([8.5, 10.0, 8.5]);
    // Slow water friction gives a ~1.6 m/s rise, so allow enough time to
    // traverse the 10 blocks to the surface.
    sim.tick(600);

    let pos = sim.player_pos().unwrap();
    assert!(
        (pos.y - (21.0 - 0.55)).abs() < 0.4,
        "deep-water bed below the surface chunk must still buoy the player, y={}",
        pos.y
    );
    assert!(sim.player_state().unwrap().in_water);
}

#[test]
fn player_floats_in_the_generated_seed_lake() {
    let world = generated_lake_world();
    let column = world
        .water_column_at(-735, -251)
        .expect("generated lake column");
    assert_eq!(column.kind, WaterKind::Lake);
    assert!((column.surface_y - 72.0).abs() < 0.01);

    // The renderer's caustics/wet-edge grid must carry the lake's surface,
    // not the global sea level.
    let mut grid = vec![
        voxel_world::water::WATER_SURFACE_NONE;
        81 * voxel_world::water::WATER_SURFACE_FLOATS_PER_CELL
    ];
    world.fill_water_surface_grid(-738, -254, 9, &mut grid);
    assert!(
        grid.iter().any(|v| (*v - 72.0).abs() < 0.01),
        "lake surface missing from grid: {grid:?}"
    );
    assert!(
        grid.iter().any(|v| *v < -1.0),
        "expected a wet-edge fringe around the lake: {grid:?}"
    );

    let mut sim = TestSim::with_world(world)
        .spawn_player_at([-734.5, 60.0, -250.5]);
    sim.tick(600);

    let pos = sim.player_pos().unwrap();
    assert!(
        (pos.y - (72.0 - 0.55)).abs() < 0.5,
        "player should float at the generated lake surface, y={}",
        pos.y
    );
    assert!(sim.player_state().unwrap().in_water);
}

#[test]
fn player_floats_in_a_cave_aquifer_under_a_lake() {
    let world = lake_over_cave_world();
    let bodies = world.water_bodies_at(8, 8);
    assert_eq!(bodies.len(), 2, "lake + cave bodies: {bodies:?}");

    let mut sim = TestSim::with_world(world).spawn_player_at([8.5, 2.5, 8.5]);
    sim.tick(180);

    let pos = sim.player_pos().unwrap();
    assert!(
        (pos.y - (5.0 - 0.55)).abs() < 0.4,
        "cave aquifer must buoy the player despite the lake above, y={}",
        pos.y
    );
    assert!(sim.player_state().unwrap().in_water);
}

#[test]
fn player_swims_up_and_sinks_down() {
    let world = water_world(WaterKind::Lake, 0);
    let mut sim = TestSim::with_world(world).spawn_player_at([8.5, 2.5, 8.5]);
    sim.tick(30);

    let start = sim.player_pos().unwrap().y;
    sim.hold(Action::Jump);
    sim.tick(45);
    let risen = sim.player_pos().unwrap().y;
    assert!(
        risen > start + 0.6,
        "holding jump should swim upward, start={start} now={risen}"
    );

    sim.release(Action::Jump);
    sim.hold(Action::Sneak);
    sim.tick(45);
    let sunk = sim.player_pos().unwrap().y;
    assert!(
        sunk < risen - 0.5,
        "holding sneak should sink, from={risen} now={sunk}"
    );
}

#[test]
fn player_drowns_while_held_under_lake_water() {
    let world = water_world(WaterKind::Lake, 0);
    let mut sim = TestSim::with_world(world).spawn_player_at([8.5, 2.5, 8.5]);
    let player = sim.player_entity().unwrap();
    let mut air = *sim.get::<voxel_game::AirSupply>(player).unwrap();
    air.current = 1.0;
    sim.ecs_world_mut().set(player, air);

    // Stay under: sinking keeps the head below the still lake surface.
    sim.hold(Action::Sneak);
    sim.tick(180);

    let air = sim.get::<voxel_game::AirSupply>(player).unwrap();
    let health = sim.get::<voxel_game::Health>(player).unwrap();
    assert_eq!(air.current, 0.0, "air should drain underwater");
    assert!(
        health.current < 20.0,
        "drowning should deal damage, health={}",
        health.current
    );
}

#[test]
fn item_floats_on_the_lake_surface_without_drifting() {
    let world = water_world(WaterKind::Lake, 0);
    let mut sim = TestSim::with_world(world).spawn_player_at([1.5, 20.0, 14.5]);
    let item = spawn_item(&mut sim, Vec3::new(8.5, 2.5, 8.5));

    sim.tick(180);

    let pos = sim.get::<Transform>(item).unwrap().pos;
    assert!(
        pos.y > WATER_SURFACE - 0.5 && pos.y < WATER_SURFACE + 0.8,
        "item should float at the lake surface, y={}",
        pos.y
    );
    assert!(
        (pos.x - 8.5).abs() < 0.1 && (pos.z - 8.5).abs() < 0.1,
        "still lake must not push the item, pos={pos:?}"
    );
}

#[test]
fn player_rides_ocean_swells_without_falling_through() {
    let world = water_world(WaterKind::Ocean, 0);
    let mut sim = TestSim::with_world(world).spawn_player_at([8.5, 5.2, 8.5]);
    sim.tick(300);

    let pos = sim.player_pos().unwrap();
    // The 4-deep test ocean has ~0.85 blocks of swell. The player must stay
    // at the moving surface instead of dropping out when a crest forms.
    assert!(
        (pos.y - (WATER_SURFACE - 0.55)).abs() < 1.2,
        "player should ride the swell, y={}",
        pos.y
    );
    assert!(pos.y > WATER_BOTTOM as f32 + 1.5, "player sank, y={}", pos.y);
    assert!(sim.player_state().unwrap().in_water);
}

#[test]
fn item_bobs_on_ocean_swells() {
    let world = water_world(WaterKind::Ocean, 0);
    let mut sim = TestSim::with_world(world).spawn_player_at([1.5, 20.0, 14.5]);
    let item = spawn_item(&mut sim, Vec3::new(8.5, 2.5, 8.5));
    sim.tick(300);

    let pos = sim.get::<Transform>(item).unwrap().pos;
    assert!(
        (pos.y - WATER_SURFACE).abs() < 1.2,
        "item should bob on the swell, y={}",
        pos.y
    );
}

#[test]
fn river_current_carries_the_player_downstream() {
    let world = water_world(WaterKind::River, 255);
    let mut sim = TestSim::with_world(world).spawn_player_at([4.5, 4.2, 8.5]);
    sim.tick(120);

    let pos = sim.player_pos().unwrap();
    assert!(
        pos.x > 6.0,
        "river current should push the player east, x={}",
        pos.x
    );
    assert!(
        (pos.z - 8.5).abs() < 0.3,
        "river current should stay along +X, z={}",
        pos.z
    );
}

#[test]
fn river_current_carries_a_floating_item_downstream() {
    let world = water_world(WaterKind::River, 255);
    let mut sim = TestSim::with_world(world).spawn_player_at([1.5, 20.0, 14.5]);
    let item = spawn_item(&mut sim, Vec3::new(4.5, 3.6, 8.5));

    sim.tick(120);

    let pos = sim.get::<Transform>(item).unwrap().pos;
    assert!(
        pos.x > 6.0,
        "river current should carry the item east, x={}",
        pos.x
    );
    assert!(
        pos.y > WATER_SURFACE - 0.8 && pos.y < WATER_SURFACE + 0.8,
        "item should ride the river surface, y={}",
        pos.y
    );
}
