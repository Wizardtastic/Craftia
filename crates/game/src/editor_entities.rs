//! Spawn helpers for ad-hoc debug entities so the runtime ECS Inspector
//! has more than the player entity to look at. These entities are
//! intended for development testing — they're not gameplay-visible.
//!
//! The inspector surfaces them as `[DebugEntityMarker]` rows; the
//! pin-cycle keybinds rotate through them as well as the player.

use glam::{Quat, Vec3};
use voxel_ecs::{Entity, World};

use crate::components::{Transform, Velocity};

/// Marker component attached to debug entities so the inspector can
/// distinguish them from gameplay entities. Has no fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugEntityMarker;

/// Spawn a debug entity with `Transform` at `pos` and zero velocity,
/// tagged with [`DebugEntityMarker`]. Returns the freshly-allocated
/// entity handle.
pub fn spawn_debug_entity(ecs: &mut World, pos: Vec3) -> Entity {
    ecs.spawn((
        Transform { pos, rot: Quat::IDENTITY },
        Velocity::default(),
        DebugEntityMarker,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_marker_is_default_and_unit() {
        // Default value constructs; Copy + Default + Debug hold.
        let m = DebugEntityMarker;
        let _copy = m;
        assert_eq!(format!("{:?}", m), "DebugEntityMarker");
    }

    #[test]
    fn spawn_debug_entity_sets_components() {
        use voxel_ecs::World;
        let mut w = World::new();
        let e = spawn_debug_entity(&mut w, Vec3::new(1.0, 2.0, 3.0));
        assert!(w.is_alive(e));
        let t = w.get::<Transform>(e).copied().unwrap();
        assert!((t.pos.x - 1.0).abs() < f32::EPSILON);
        assert!((t.pos.y - 2.0).abs() < f32::EPSILON);
        assert!((t.pos.z - 3.0).abs() < f32::EPSILON);
        assert!(w.has::<DebugEntityMarker>(e));
        assert!(w.has::<Velocity>(e));
    }

    #[test]
    fn spawn_two_distinct_entities() {
        use voxel_ecs::World;
        let mut w = World::new();
        let a = spawn_debug_entity(&mut w, Vec3::ZERO);
        let b = spawn_debug_entity(&mut w, Vec3::ZERO);
        assert_ne!(a.index, b.index, "two spawns must get different slots");
    }
}
