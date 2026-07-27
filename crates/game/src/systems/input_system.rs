//! Reads from the engine's input snapshot (via resource) and writes a
//! `PlayerInput` component on the player entity.
//!
//! The engine is expected to update the `InputResource` each frame before
//! the gameplay schedule runs.

use voxel_ecs::World;

use crate::components::{PlayerEntity, PlayerInput};

/// Minimal input data the system needs. The engine's full `InputState`
/// projects these booleans (and the mouse delta) from raw window events.
#[derive(Clone, Copy, Debug, Default)]
pub struct InputSnapshot {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub sneak: bool,
    pub sprint: bool,
    pub flying: bool,
    pub mouse_delta: (f32, f32),
    /// Left mouse button held (for continuous mining).
    pub mining: bool,
    /// Right mouse button pressed this frame (for placing/using).
    pub use_item: bool,
}

/// Resource: the current input snapshot. The engine writes this every
/// frame just before running gameplay systems.
#[derive(Clone, Copy, Debug, Default)]
pub struct InputResource(pub InputSnapshot);

/// System: translate `InputSnapshot` into a `PlayerInput` component on the
/// player entity. The `PlayerInput` is then consumed by `movement_system`.
///
/// After copying the snapshot, the `mouse_delta` in `InputResource` is zeroed
/// so that subsequent fixed steps in the same frame don't re-apply the same
/// delta (which would cause the camera to snap on frame spikes where
/// `tick_fixed` runs multiple catch-up steps).
pub fn input_system(world: &mut World, _dt: f32) {
    let input = match world.resource_mut::<InputResource>() {
        Some(r) => {
            let snap = r.0;
            // Consume the mouse delta so it's only applied once per frame.
            r.0.mouse_delta = (0.0, 0.0);
            snap
        }
        None => return,
    };

    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let mut wish = glam::Vec3::ZERO;
    if input.forward {
        wish.z -= 1.0;
    }
    if input.back {
        wish.z += 1.0;
    }
    if input.left {
        wish.x -= 1.0;
    }
    if input.right {
        wish.x += 1.0;
    }
    if wish.length_squared() > 0.0 {
        wish = wish.normalize();
    }

    let pi = PlayerInput {
        wish,
        jump: input.jump,
        sneaking: input.sneak,
        sprinting: input.sprint,
        flying: input.flying,
        mouse_delta: input.mouse_delta,
        mining: input.mining,
        use_item: input.use_item,
    };

    world.set(player_entity, pi);
}
