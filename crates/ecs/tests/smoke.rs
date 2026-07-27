//! Integration smoke tests for `voxel-ecs`.

use voxel_ecs::*;

// --- Test component types --------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct Velocity {
    dx: f32,
    dy: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct Health(u32);

#[derive(Debug, Clone, PartialEq)]
struct Name(String);

// --- Basic spawn / get / set -----------------------------------------------

#[test]
fn spawn_single_component() {
    let mut world = World::new();
    let e = world.spawn((Position { x: 1.0, y: 2.0 },));
    assert!(world.is_alive(e));
    assert_eq!(world.get::<Position>(e), Some(&Position { x: 1.0, y: 2.0 }));
    assert_eq!(world.entity_count(), 1);
}

#[test]
fn spawn_bundle() {
    let mut world = World::new();
    let e = world.spawn((
        Position { x: 0.0, y: 0.0 },
        Velocity { dx: 1.0, dy: 0.0 },
        Health(100),
    ));
    assert!(world.has::<Position>(e));
    assert!(world.has::<Velocity>(e));
    assert!(world.has::<Health>(e));
    assert_eq!(world.get::<Health>(e), Some(&Health(100)));
}

#[test]
fn set_replaces_existing() {
    let mut world = World::new();
    let e = world.spawn((Position { x: 1.0, y: 2.0 },));
    world.set(e, Position { x: 9.0, y: 9.0 });
    assert_eq!(world.get::<Position>(e), Some(&Position { x: 9.0, y: 9.0 }));
}

#[test]
fn set_adds_new_component() {
    let mut world = World::new();
    let e = world.spawn((Position { x: 1.0, y: 2.0 },));
    assert!(!world.has::<Velocity>(e));
    world.set(e, Velocity { dx: 1.0, dy: 0.0 });
    assert!(world.has::<Position>(e));
    assert!(world.has::<Velocity>(e));
    assert_eq!(
        world.get::<Velocity>(e),
        Some(&Velocity { dx: 1.0, dy: 0.0 })
    );
}

#[test]
fn remove_returns_value() {
    let mut world = World::new();
    let e = world.spawn((Position { x: 1.0, y: 2.0 }, Health(50)));
    let removed = world.remove::<Health>(e);
    assert_eq!(removed, Some(Health(50)));
    assert!(world.has::<Position>(e));
    assert!(!world.has::<Health>(e));
}

#[test]
fn get_mut_modifies_in_place() {
    let mut world = World::new();
    let e = world.spawn((Health(10),));
    if let Some(h) = world.get_mut::<Health>(e) {
        h.0 += 5;
    }
    assert_eq!(world.get::<Health>(e), Some(&Health(15)));
}

// --- Despawn & entity recycling --------------------------------------------

#[test]
fn despawn_frees_slot() {
    let mut world = World::new();
    let e1 = world.spawn((Health(1),));
    assert!(world.despawn(e1));
    assert!(!world.is_alive(e1));
    // The same handle is now stale.
    assert!(world.get::<Health>(e1).is_none());
}

#[test]
fn entity_count_tracks_live() {
    let mut world = World::new();
    let e1 = world.spawn((Health(1),));
    let _e2 = world.spawn((Health(2),));
    let _e3 = world.spawn((Health(3),));
    assert_eq!(world.entity_count(), 3);
    world.despawn(e1);
    assert_eq!(world.entity_count(), 2);
}

// --- Queries ---------------------------------------------------------------

#[test]
fn query_single_component() {
    let mut world = World::new();
    world.spawn((Position { x: 1.0, y: 0.0 },));
    world.spawn((Position { x: 2.0, y: 0.0 },));
    world.spawn((Health(99),));

    let mut xs: Vec<f32> = world.query::<&Position>().map(|(_e, p)| p.x).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(xs, vec![1.0, 2.0]);
}

#[test]
fn query_tuple_of_two_components() {
    let mut world = World::new();
    world.spawn((Position { x: 1.0, y: 0.0 }, Velocity { dx: 0.0, dy: 1.0 }));
    world.spawn((Position { x: 2.0, y: 0.0 }, Velocity { dx: 0.0, dy: 2.0 }));
    world.spawn((Position { x: 3.0, y: 0.0 },)); // no velocity, should not match

    let mut count = 0;
    for (_e, (p, _v)) in world.query::<(&Position, &Velocity)>() {
        assert!(p.x > 0.0);
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn query_mut_allows_mutation() {
    let mut world = World::new();
    world.spawn((Position { x: 0.0, y: 0.0 },));
    world.spawn((Position { x: 0.0, y: 0.0 },));
    for (_e, p) in world.query::<&mut Position>() {
        p.x += 1.0;
    }
    let mut total = 0.0;
    for (_e, p) in world.query::<&Position>() {
        total += p.x;
    }
    assert_eq!(total, 2.0);
}

// --- Archetypes ------------------------------------------------------------

#[test]
fn archetypes_are_reused_for_same_composition() {
    let mut world = World::new();
    let a = world.spawn((Position { x: 0.0, y: 0.0 }, Health(1)));
    let _b = world.spawn((Position { x: 0.0, y: 0.0 }, Health(2)));
    // Replacing an existing component should not grow the archetype count.
    world.set(a, Position { x: 1.0, y: 1.0 });
    // The world should have at most two archetypes ({Position} and
    // {Position, Health}).
    assert!(world.archetype_count() <= 2);
}

// --- Resources -------------------------------------------------------------

#[test]
fn resources_round_trip() {
    let mut world = World::new();
    assert!(world.resource::<u64>().is_none());
    assert!(world.insert_resource(42u64).is_none());
    assert_eq!(world.resource::<u64>().copied(), Some(42));
    *world.resource_mut::<u64>().unwrap() += 1;
    assert_eq!(world.resource::<u64>().copied(), Some(43));
    assert_eq!(world.remove_resource::<u64>(), Some(43));
    assert!(world.resource::<u64>().is_none());
}

// --- Schedule --------------------------------------------------------------

#[test]
fn schedule_runs_systems_in_order() {
    let mut world = World::new();
    world.spawn((Health(0),));

    let mut schedule = SystemSchedule::new()
        .add_fn("heal", |world, _dt| {
            for (_e, h) in world.query::<&mut Health>() {
                h.0 += 10;
            }
        })
        .add_fn("heal again", |world, _dt| {
            for (_e, h) in world.query::<&mut Health>() {
                h.0 += 5;
            }
        });

    schedule.run(&mut world, 1.0 / 60.0);

    for (_e, h) in world.query::<&Health>() {
        assert_eq!(h.0, 15);
    }
}

// --- Per-system timing ---------------------------------------------------

#[test]
fn system_schedule_timing_disabled_is_zero_cost() {
    let mut world = World::new();
    let mut schedule = SystemSchedule::new()
        .add_fn("a", |_w, _dt| {})
        .add_fn("b", |_w, _dt| {});
    assert!(!schedule.timings_enabled());
    schedule.run(&mut world, 1.0 / 60.0);
    assert!(
        schedule.last_frame_timings().is_empty(),
        "timing-disabled schedule must report zero timings"
    );
}

#[test]
fn system_schedule_records_per_system_timings() {
    let mut world = World::new();
    world.spawn((Health(0),));
    let mut schedule = SystemSchedule::new()
        .with_timing()
        .add_fn("sleep_a", |_w, _dt| {
            // ~2ms is comfortably above the lower-bound assertion and
            // still well under the 50ms upper bound on slow CI runners.
            std::thread::sleep(std::time::Duration::from_millis(2));
        })
        .add_fn("sleep_b", |_w, _dt| {
            std::thread::sleep(std::time::Duration::from_millis(2));
        })
        .add_fn("noop", |_w, _dt| {});
    assert!(schedule.timings_enabled());
    schedule.run(&mut world, 1.0 / 60.0);

    let timings = schedule.last_frame_timings();
    assert_eq!(timings.len(), 3, "must have one entry per system");
    assert_eq!(timings[0].0, "sleep_a");
    assert!(
        timings[0].1 >= 1_000,
        "sleep_a should be at least 1ms ({}µs)",
        timings[0].1
    );
    assert!(
        timings[0].1 <= 50_000,
        "sleep_a should be under 50ms ({}µs)",
        timings[0].1
    );
    assert_eq!(timings[1].0, "sleep_b");
    assert!(timings[1].1 >= 1_000);
    assert_eq!(timings[2].0, "noop");
    // `noop` should still record a non-negative elapsed duration; the
    // exact value is platform-dependent so we only assert non-negative.
    assert!(timings[2].1 < 50_000);
}

#[test]
fn system_schedule_timings_overwrite_per_run() {
    let mut world = World::new();
    let mut schedule = SystemSchedule::new()
        .with_timing()
        .add_fn("first", |_w, _dt| {})
        .add_fn("second", |_w, _dt| {});
    schedule.run(&mut world, 1.0 / 60.0);
    let first_run = schedule.last_frame_timings();
    assert_eq!(first_run.len(), 2);
    // Re-run with a no-op; buffer should still hold exactly two entries
    // (overwritten, not appended).
    schedule.run(&mut world, 1.0 / 60.0);
    let second_run = schedule.last_frame_timings();
    assert_eq!(second_run.len(), 2, "timings must not append across runs");
}

// --- Bundle naming sanity check -------------------------------------------

#[test]
fn tuple_bundle_with_8_components() {
    let mut world = World::new();
    let e = world.spawn((
        Position { x: 0.0, y: 0.0 },
        Velocity { dx: 1.0, dy: 1.0 },
        Health(1),
        Name("a".to_string()),
        Position { x: 0.0, y: 0.0 }, // duplicate — second wins via set()
        Velocity { dx: 2.0, dy: 2.0 },
        Health(2),
        Name("b".to_string()),
    ));
    assert!(world.is_alive(e));
    assert_eq!(world.get::<Position>(e), Some(&Position { x: 0.0, y: 0.0 }));
    assert_eq!(world.get::<Health>(e), Some(&Health(2)));
}
