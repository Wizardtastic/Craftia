//! Core animation data types.

use serde::{Deserialize, Serialize};

/// A single keyframe in a channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keyframe {
    /// Time in seconds from clip start.
    pub time: f32,
    /// Value at this keyframe (up to 4 components).
    pub value: [f32; 4],
}

/// Interpolation mode between keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    /// Hold previous value until next keyframe.
    Step,
    /// Linear interpolation.
    Linear,
    /// Smooth cubic spline (with tangents).
    CubicSpline,
    /// Bezier curve.
    Bezier,
}

impl Default for Interpolation {
    fn default() -> Self {
        Self::Linear
    }
}

/// What property a channel targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TargetedProperty {
    /// Animate translation (Vec3).
    Translation,
    /// Animate rotation (Quat stored as [x, y, z, w]).
    Rotation,
    /// Animate scale (Vec3).
    Scale,
    /// Morph target weight.
    Weight,
    /// Named float property (e.g., emission, alpha).
    Float { name: String },
    /// Named color property (RGBA).
    Color { name: String },
    /// Named texture tile index.
    TextureTile { name: String },
}

/// A single animation channel targeting one property on one node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationChannel {
    /// Node index in the hierarchy (0 = root).
    pub target_node: usize,
    /// Which property this channel animates.
    pub property: TargetedProperty,
    /// Interpolation mode.
    pub interpolation: Interpolation,
    /// Keyframe times (sorted ascending).
    pub keyframe_times: Vec<f32>,
    /// Keyframe values (same length as keyframe_times).
    /// Each value is [f32; 4] — 1 component for float, 3 for translation/scale,
    /// 4 for rotation/quaternion/color.
    pub keyframe_values: Vec<[f32; 4]>,
}

/// A named animation clip (typically one per glTF animation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationClip {
    /// Clip name (e.g., "idle", "walk", "attack").
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// Animation channels.
    pub channels: Vec<AnimationChannel>,
}

/// A complete set of clips for one model/scene.
#[derive(Clone, Debug, Default)]
pub struct AnimationLibrary {
    /// All available clips.
    pub clips: Vec<AnimationClip>,
    /// Index of the default clip to play.
    pub default_clip: usize,
}

impl AnimationLibrary {
    /// Get a clip by name.
    pub fn find_clip(&self, name: &str) -> Option<&AnimationClip> {
        self.clips.iter().find(|c| c.name == name)
    }

    /// Get a clip by index.
    pub fn get_clip(&self, index: usize) -> Option<&AnimationClip> {
        self.clips.get(index)
    }
}
