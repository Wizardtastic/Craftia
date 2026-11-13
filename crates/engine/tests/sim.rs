//! Integration tests for the headless `TestSim` harness.
//!
//! Each test drives a `TestSim` for a fixed number of fixed-timestep
//! ticks (1/60s each) and asserts on the resulting ECS state. No GPU,
//! no window, no event loop. Run with `cargo test -p voxel-engine
//! --test sim` or as part of `cargo test --workspace`.

use voxel_engine::test_sim::TestSim;
use voxel_game::input::Action;

#[test]
fn player_falls_with_gravity() {
    let mut sim = TestSim::new().spawn_player_at([0.0, 20.0, 0.0]);
    sim.tick(60); // 1 second at 60 Hz
    let pos = sim.player_pos().expect("player should be alive");
    assert!(
        pos.y < 19.9,
        "player should fall under gravity, y={}",
        pos.y
    );
}

#[test]
fn player_lands_and_stands_still_on_ground() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    // Stone floor is at y=0 (block at y=0, top at y=1), so the player
    // AABB centre should settle around y = 1.0 + PLAYER_HALF.y = 1.9.
    sim.tick(60);
    let pos = sim.player_pos().expect("player should be alive");
    assert!(
        (pos.y - 1.9).abs() < 0.5,
        "player should land and stand on ground, y={}",
        pos.y
    );
}

#[test]
fn player_moves_forward_when_w_held() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.tick(30); // settle onto the floor
    sim.hold(Action::Forward);
    sim.tick(30); // 0.5 seconds at 60 Hz
    let pos = sim.player_pos().expect("player should be alive");
    // Walk speed is ~4.3 m/s, so half a second is ~1.9 m. Forward direction
    // is -Z (camera-relative; the spawn default looks down -Z).
    assert!(
        pos.z < -1.2,
        "player should move in -Z direction when W held, z={}",
        pos.z
    );
}

#[test]
fn player_jumps_when_space_held() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.tick(30); // settle on ground first
    let pre = sim.player_pos().expect("player should be alive");
    sim.hold(Action::Jump);
    sim.tick(4); // one 20 Hz tick of jump impulse + ascent
    let post = sim.player_pos().expect("player should be alive");
    assert!(
        post.y > pre.y,
        "player should rise after jump, pre.y={} post.y={}",
        pre.y,
        post.y
    );
}

#[test]
fn fly_mode_keeps_player_aloft() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 20.0, 0.5]);
    let player = sim.player_entity().expect("player should be alive");
    // Flight is a creative/spectator ability in vanilla; survival ignores the
    // flag. Put the player in creative before flipping it.
    sim.ecs_world_mut().set(player, voxel_game::GameMode::Creative);
    sim.ecs_world_mut()
        .get_mut::<voxel_game::PlayerInput>(player)
        .expect("player has PlayerInput")
        .flying = true;
    sim.tick(60); // 1 second with no input
    let pos = sim.player_pos().expect("player should be alive");
    assert!(
        (pos.y - 20.0).abs() < 0.5,
        "fly mode should keep player aloft, y={}",
        pos.y
    );
}

#[test]
fn held_action_survives_multiple_ticks() {
    // Holding Forward across many ticks should keep moving the player.
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.hold(Action::Forward);
    sim.tick(15);
    let p1 = sim.player_pos().unwrap();
    sim.tick(15);
    let p2 = sim.player_pos().unwrap();
    assert!(
        p2.z < p1.z,
        "player should keep moving while Forward is held, p1.z={} p2.z={}",
        p1.z,
        p2.z
    );
}

#[test]
fn release_action_stops_movement() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.hold(Action::Forward);
    sim.tick(10);
    sim.release(Action::Forward);
    sim.tick(30); // 0.5s without input
    let pos = sim.player_pos().unwrap();
    let vel = sim.player_vel().unwrap();
    // After release, horizontal velocity should be ~0 (no wish, only gravity
    // affects vel.y); the player should be standing on the floor.
    assert!(
        vel.x.abs() < 0.1 && vel.z.abs() < 0.1,
        "released Forward should zero horizontal velocity, vel={:?}",
        vel
    );
    assert!(
        (pos.y - 1.9).abs() < 0.5,
        "player should be on ground after release, y={}",
        pos.y
    );
}

#[test]
fn tick_cap_prevents_runaway() {
    // 5000 ticks should be capped to TICK_CAP (1000). 1000 ticks at
    // 60Hz = ~16.7s of sim time, enough for a free-falling player to
    // hit the ground from any reasonable starting Y.
    let mut sim = TestSim::new().spawn_player_at([0.5, 50.0, 0.5]);
    sim.tick(5000);
    let pos = sim.player_pos().unwrap();
    assert!(
        pos.y < 5.0,
        "player should have hit the ground within the tick cap, y={}",
        pos.y
    );
}

#[test]
fn jump_apex_matches_vanilla_height() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.tick(30); // settle on the floor
    let start = sim.player_pos().unwrap().y;

    sim.hold(Action::Jump);
    sim.tick(4);
    sim.release(Action::Jump);

    let mut apex = start;
    for _ in 0..40 {
        sim.tick(1);
        apex = apex.max(sim.player_pos().unwrap().y);
    }
    let height = apex - start;
    assert!(
        (1.20..=1.30).contains(&height),
        "vanilla jump apex is 1.2522 blocks, got {height}"
    );
}

#[test]
fn movement_is_independent_of_refresh_rate() {
    fn travel(frame_dt: f64, frames: u32) -> f32 {
        let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
        sim.tick(30);
        sim.hold(Action::Forward);
        sim.tick_frame(frame_dt, frames);
        sim.player_pos().unwrap().z
    }

    let at_30 = travel(1.0 / 30.0, 30);
    let at_60 = travel(1.0 / 60.0, 60);
    let at_144 = travel(1.0 / 144.0, 144);
    assert!(
        (at_30 - at_60).abs() < 0.3,
        "30 Hz ({at_30}) and 60 Hz ({at_60}) should walk the same distance"
    );
    assert!(
        (at_144 - at_60).abs() < 0.3,
        "144 Hz ({at_144}) and 60 Hz ({at_60}) should walk the same distance"
    );
}

#[test]
fn survival_mining_takes_vanilla_stone_time() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.tick(30);

    let stone = sim.voxel_world().registry().id_of("stone").unwrap();
    assert!(sim.voxel_world().set_block(2, 2, 0, stone));
    sim.ecs_world_mut()
        .insert_resource(voxel_game::PlayerLookTarget {
            block: Some([2, 2, 0]),
            block_id: stone,
        });
    sim.hold(Action::Attack);

    // Stone by hand takes 7.5 s in vanilla.
    sim.tick(420);
    assert_eq!(
        sim.voxel_world().get_block(2, 2, 0),
        stone,
        "stone should survive 7 s of mining"
    );
    sim.tick(60);
    assert!(
        sim.voxel_world().get_block(2, 2, 0).is_air(),
        "stone should break after 7.5 s"
    );
}

#[test]
fn sprinting_drains_hunger() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.tick(30);
    let player = sim.player_entity().unwrap();

    let mut hunger = *sim.get::<voxel_game::Hunger>(player).unwrap();
    hunger.food = 10.0;
    hunger.saturation = 0.0;
    hunger.exhaustion = 3.6; // one second of sprinting crosses 4.0
    sim.ecs_world_mut().set(player, hunger);

    sim.hold(Action::Forward);
    sim.hold(Action::Sprint);
    sim.tick(60);

    let hunger = sim.get::<voxel_game::Hunger>(player).unwrap();
    assert!(
        hunger.food < 10.0,
        "sprinting should consume food, got {}",
        hunger.food
    );
}

#[test]
fn drowning_drains_air_and_deals_damage() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    sim.tick(20);
    let water = sim.voxel_world().registry().id_of("water").unwrap();
    assert!(sim.voxel_world().set_block(0, 2, 0, water));

    let player = sim.player_entity().unwrap();
    let mut air = *sim.get::<voxel_game::AirSupply>(player).unwrap();
    air.current = 1.0;
    sim.ecs_world_mut().set(player, air);

    sim.tick(90); // 1.5 s submerged
    let air = sim.get::<voxel_game::AirSupply>(player).unwrap();
    let health = sim.get::<voxel_game::Health>(player).unwrap();
    assert_eq!(air.current, 0.0, "air should be exhausted");
    assert!(
        health.current < 20.0,
        "drowning should hurt, health={}",
        health.current
    );
}

#[test]
fn hard_landing_deals_vanilla_fall_damage() {
    let mut sim = TestSim::new().spawn_player_at([0.5, 20.0, 0.5]);
    sim.tick(90); // fall ~18 blocks and land
    let player = sim.player_entity().unwrap();
    let health = sim.get::<voxel_game::Health>(player).unwrap();
    // ~18 blocks of fall = floor(18 − 3) = 15 HP, leaving 5.
    assert!(
        (4.0..=6.0).contains(&health.current),
        "an 18-block fall should deal 15 HP (health 5), got {}",
        health.current
    );
}

#[test]
fn sprint_requires_food_above_six() {
    // Full hunger: sprint moves farther than a walk over the same time.
    let mut full = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    full.tick(30);
    full.hold(Action::Forward);
    full.hold(Action::Sprint);
    full.tick(30);
    let full_distance = full.player_pos().unwrap().z.abs();

    // Food at 6 disables sprinting.
    let mut starving = TestSim::new().spawn_player_at([0.5, 2.0, 0.5]);
    let player = starving.player_entity().unwrap();
    let mut hunger = *starving.get::<voxel_game::Hunger>(player).unwrap();
    hunger.food = 6.0;
    hunger.saturation = 0.0;
    starving.ecs_world_mut().set(player, hunger);
    starving.tick(30);
    starving.hold(Action::Forward);
    starving.hold(Action::Sprint);
    starving.tick(30);
    let slow_distance = starving.player_pos().unwrap().z.abs();

    assert!(
        full_distance > slow_distance * 1.15,
        "sprint should be faster with food above 6 (full={full_distance}, low={slow_distance})"
    );
}

#[test]
fn player_has_expected_components() {
    let sim = TestSim::new();
    let player = sim.player_entity().expect("player should be alive");
    assert!(sim.has_component::<voxel_game::Transform>(player));
    assert!(sim.has_component::<voxel_game::Velocity>(player));
    assert!(sim.has_component::<voxel_game::PlayerInput>(player));
    assert!(sim.has_component::<voxel_game::PlayerState>(player));
    assert!(sim.has_component::<voxel_game::Aabb>(player));
    assert!(sim.has_component::<voxel_game::CameraOwner>(player));
}
