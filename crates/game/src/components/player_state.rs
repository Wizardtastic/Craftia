//! Persistent player state across frames. Written by `movement_system`,
//! read by other systems (camera, audio, particles, ...).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PlayerState {
    pub on_ground: bool,
    pub in_water: bool,
    pub was_in_water: bool,
    pub fall_speed_peak: f32,
    /// Y offset for the camera (0.8 standing, 0.55 sneaking).
    pub eye_offset: f32,
    /// Accumulated bob phase (radians), driven by horizontal speed.
    /// Used for camera bobbing and held-item bob.
    pub bob_phase: f32,
    /// Mining swing progress (1.0 = just started, 0.0 = finished).
    /// Decremented each frame over ~400ms.
    pub mining_swing: f32,
}
