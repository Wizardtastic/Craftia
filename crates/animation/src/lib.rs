//! Animation system for the voxel engine.
//!
//! Provides shared animation data types, keyframe sampling, interpolation,
//! and an animation clip registry for loading `.anim` files.

pub mod data;
pub mod interpolate;
pub mod registry;
pub mod sampling;

pub use data::{
    AnimationChannel, AnimationClip, AnimationLibrary, Interpolation, Keyframe, TargetedProperty,
};
pub use registry::AnimationClipRegistry;
pub use sampling::{sample_channel, sample_rotation, sample_scale, sample_translation, trs_to_matrix};
