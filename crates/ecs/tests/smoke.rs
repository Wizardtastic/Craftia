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

#[test]
#[should_panic(expected = "aliased")]
fn query_rejects_shared_mutable_duplicate_component() {
    let mut world = World::new();
    world.spawn((Position { x: 0.0, y: 0.0 },));
    let _ = world.query::<(&Position, &mut Position)>();
}

#[test]
#[should_panic(expected = "duplicate mutable")]
fn query_rejects_duplicate_mutable_component() {
    let mut world = World::new();
    world.spawn((Position { x: 0.0, y: 0.0 },));
    let _ = world.query::<(&mut Position, &mut Position)>();
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

// --- Archetypes beyond the 16-component stack buffer ----------------------
//
// `get_or_create_archetype`/`transition` use a MAX_STACK=16 stack buffer
// for the archetype key. Regression: sets larger than 16 must fall back to
// the heap instead of silently dropping types (which panics later with
// "missing column in new archetype after get_or_create").

macro_rules! count_comp {
    ($name:ident, $val:expr) => {
        #[derive(Debug, Clone, PartialEq)]
        struct $name(u32);
        const _: () = ();
    };
}

count_comp!(C0, 0);
count_comp!(C1, 1);
count_comp!(C2, 2);
count_comp!(C3, 3);
count_comp!(C4, 4);
count_comp!(C5, 5);
count_comp!(C6, 6);
count_comp!(C7, 7);
count_comp!(C8, 8);
count_comp!(C9, 9);
count_comp!(C10, 10);
count_comp!(C11, 11);
count_comp!(C12, 12);
count_comp!(C13, 13);
count_comp!(C14, 14);
count_comp!(C15, 15);
count_comp!(C16, 16);
count_comp!(C17, 17);

#[test]
fn archetype_supports_more_than_16_components() {
    let mut world = World::new();
    let e = world.spawn((C0(0), C1(1), C2(2), C3(3), C4(4), C5(5), C6(6), C7(7)));
    // Grow the archetype past the 16-component stack-buffer cap.
    world.set(e, C8(8));
    world.set(e, C9(9));
    world.set(e, C10(10));
    world.set(e, C11(11));
    world.set(e, C12(12));
    world.set(e, C13(13));
    world.set(e, C14(14));
    world.set(e, C15(15));
    world.set(e, C16(16)); // 17th component — crosses the cap
    world.set(e, C17(17)); // 18th
    assert_eq!(world.get::<C16>(e), Some(&C16(16)));
    assert_eq!(world.get::<C17>(e), Some(&C17(17)));
    assert_eq!(world.get::<C0>(e), Some(&C0(0)));
    // And back down again: removing a component below the cap must also
    // transition cleanly.
    let removed = world.remove::<C0>(e);
    assert_eq!(removed, Some(C0(0)));
    assert!(!world.has::<C0>(e));
    assert!(world.has::<C17>(e));
}
