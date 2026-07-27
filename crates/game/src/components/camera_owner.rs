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
/// transform + current eye offset. Includes camera bobbing.
pub fn update_camera_from_transform(
    camera: &mut Camera,
    transform: &Transform,
    state: &PlayerState,
) {
    let base_pos = transform.pos + glam::Vec3::new(0.0, state.eye_offset, 0.0);

    // Compute bob offset (figure-8 pattern: up/down + slight left/right).
    let bob_amp = 0.012; // ~1.2cm (MC is ~1cm)
    let y_bob = state.bob_phase.sin() * bob_amp;
    let x_bob = (state.bob_phase * 2.0 + std::f32::consts::PI).sin() * bob_amp * 0.3;

    camera.pos = base_pos + glam::Vec3::new(x_bob, y_bob, 0.0);
    let (yaw, pitch, _roll) = transform.rot.to_euler(glam::EulerRot::YXZ);
    camera.yaw = yaw;
    camera.pitch = pitch;
}
