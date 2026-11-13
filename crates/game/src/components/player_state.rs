//! Persistent player state across frames. Written by `movement_system`,
//! read by other systems (camera, audio, particles, ...).
//!
//! Player physics runs on a fixed 20 Hz vanilla tick. `physics_accum` carries
//! the leftover time between engine steps and `prev_eye_pos` / `interp_alpha`
//! let the camera interpolate between the last two physics ticks, so the game
//! is smooth and identical at any render or simulation refresh rate.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PlayerState {
    pub on_ground: bool,
    pub in_water: bool,
    /// Distance fallen since leaving the ground, in blocks. Vanilla computes
    /// fall damage from this, not from the impact speed.
    #[serde(default)]
    pub fall_distance: f32,
    /// Y offset for the camera above the AABB centre (vanilla 1.62−0.9
    /// standing, 1.27−0.75 sneaking).
    pub eye_offset: f32,
    /// Accumulated bob phase (radians), driven by horizontal speed.
    /// Used for camera bobbing and held-item bob.
    pub bob_phase: f32,
    /// Mining swing progress (1.0 = just started, 0.0 = finished).
    /// Decremented each frame over ~400ms.
    pub mining_swing: f32,
    /// Leftover time (seconds) toward the next 20 Hz physics tick.
    #[serde(default)]
    pub physics_accum: f32,
    /// Eye position at the previous physics tick, for render interpolation.
    #[serde(default)]
    pub prev_eye_pos: Vec3,
    /// Interpolation factor in `0..=1` between `prev_eye_pos` and the
    /// current eye position.
    #[serde(default)]
    pub interp_alpha: f32,
    /// Seconds left in which a queued jump is still valid. Buffering the
    /// press means a tap shorter than one 50 ms tick is never dropped.
    #[serde(default)]
    pub jump_buffer: f32,
}
