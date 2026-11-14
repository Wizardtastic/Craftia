//! Camera ownership marker + the resource that points at the local player.

use voxel_core::Camera;
use voxel_ecs::Entity;

use crate::components::player_state::PlayerState;
use crate::components::transform::Transform;

/// Marker component: this entity owns the main camera. Only one entity
/// should have this at a time.
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraOwner;

/// Resource holding the entity ID of the local player. `None` until the
/// player has been spawned.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerEntity(pub Option<Entity>);

/// Compute the camera's position and orientation from the player's
/// transform + current eye offset. The position is interpolated between the
/// last two 20 Hz physics ticks (entities without physics state, such as
/// debug pins, simply use their current position). Includes camera bobbing.
pub fn update_camera_from_transform(
    camera: &mut Camera,
    transform: &Transform,
    state: &PlayerState,
) {
    let current_eye = transform.pos + glam::Vec3::new(0.0, state.eye_offset, 0.0);
    let base_pos = if state.prev_eye_pos.length_squared() > f32::EPSILON {
        state
            .prev_eye_pos
            .lerp(current_eye, state.interp_alpha.clamp(0.0, 1.0))
    } else {
        current_eye
    };

    // Compute bob offset (figure-8 pattern: up/down + slight left/right).
    // Vanilla view bobbing is roughly 5cm at the extreme; use a slightly
    // gentler 3cm so it reads the same without becoming distracting. The
    // envelope fades to zero on a stop, so the offset shrinks smoothly along
    // the current sine instead of vibrating through a decaying phase.
    let bob_amp = 0.03 * state.bob_amplitude.clamp(0.0, 1.0);
    let y_bob = state.bob_phase.sin() * bob_amp;
    let x_bob = (state.bob_phase * 2.0 + std::f32::consts::PI).sin() * bob_amp * 0.3;

    camera.pos = base_pos + glam::Vec3::new(x_bob, y_bob, 0.0);
    let (yaw, pitch, _roll) = transform.rot.to_euler(glam::EulerRot::YXZ);
    camera.yaw = yaw;
    camera.pitch = pitch;
}
