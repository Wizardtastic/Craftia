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
    sim.hold(Action::Forward);
    sim.tick(30); // 0.5 seconds at 60 Hz
    let pos = sim.player_pos().expect("player should be alive");
    // Walk speed is ~4.3 m/s, so half a second is ~2m. Forward direction
    // is -Z (camera-relative; the spawn default looks down -Z).
    assert!(
        pos.z < -1.0,
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
    sim.tick(2); // one jump impulse + a tick of ascent
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
    // Flip the flying flag directly on PlayerInput.
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
