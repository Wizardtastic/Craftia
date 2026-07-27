//! Property animator component: animates named properties via keyframes.

use serde::{Deserialize, Serialize};
use voxel_animation::data::AnimationClip;

/// Handle referencing a loaded animation clip by name.
#[derive(Clone, Debug)]
pub struct AnimationClipHandle {
    /// Name of the clip in the registry.
    pub clip_name: String,
}

/// What property on the entity to animate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PropertyTarget {
    /// Animate Transform.position.
    Position,
    /// Animate Transform.rotation (as euler angles in degrees).
    RotationEuler,
    /// Animate Transform.scale.
    Scale,
}

/// ECS component that animates a single named property on this entity.
#[derive(Clone, Debug)]
pub struct PropertyAnimator {
    /// The animation clip to play.
    pub clip: Option<AnimationClip>,
    /// Current playback time in seconds.
    pub time: f32,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
    /// Whether the animation loops.
    pub loop_: bool,
    /// Whether the animation is currently playing.
    pub playing: bool,
    /// Which property this animator targets.
    pub target: PropertyTarget,
}

impl Default for PropertyAnimator {
    fn default() -> Self {
        Self {
            clip: None,
            time: 0.0,
            speed: 1.0,
            loop_: false,
            playing: false,
            target: PropertyTarget::Position,
        }
    }
}

impl PropertyAnimator {
    /// Create a new PropertyAnimator with the given clip and target.
    pub fn new(clip: AnimationClip, target: PropertyTarget) -> Self {
        Self {
            clip: Some(clip),
            time: 0.0,
            speed: 1.0,
            loop_: true,
            playing: true,
            target,
        }
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }

    /// Advance time by dt. Returns true if the animation completed (non-looping).
    pub fn tick(&mut self, dt: f32) -> bool {
        if !self.playing || self.clip.is_none() {
            return false;
        }
        self.time += dt * self.speed;
        let duration = self.clip.as_ref().map(|c| c.duration).unwrap_or(0.0);
        if self.time >= duration {
            if self.loop_ {
                self.time %= duration;
            } else {
                self.time = duration;
                self.playing = false;
                return true;
            }
        }
        false
    }
}
