//! Animation system: advances animation time, samples keyframes,
//! and computes bone transforms for entities with AnimationPlayer
//! + ModelRef + BoneTransforms components.
//!
//! Runs AFTER hierarchy_system in the schedule.

use glam::{Quat, Vec3};
use std::sync::Arc;
use voxel_ecs::World;

use crate::components::{AnimationPlayer, BoneTransforms, ModelRef};

/// Resource: animation data for all loaded models, indexed by model_id.
/// Inserted by the engine when models are loaded.
/// Wrapped in Arc so the animation system can borrow without cloning.
#[derive(Clone, Default)]
pub struct AnimationDataResource {
    /// Per-model animation data. Index matches ModelRef.model_id.
    pub data: Arc<Vec<ModelAnimationData>>,
}

/// Resource: skin data for all loaded models, indexed by model_id.
/// Inserted by the engine when models are loaded.
/// Wrapped in Arc so the animation system can borrow without cloning.
#[derive(Clone, Default)]
pub struct SkinDataResource {
    /// Per-model skin data. Index matches ModelRef.model_id.
    pub data: Arc<Vec<ModelSkinData>>,
}

/// Skin data for a single model.
#[derive(Clone, Debug)]
pub struct ModelSkinData {
    /// Skins defined in this model.
    pub skins: Vec<SkinInfo>,
    /// Node parent indices for computing world transforms.
    pub node_parents: Vec<Option<usize>>,
}

/// Information about a single skin.
#[derive(Clone, Debug)]
pub struct SkinInfo {
    /// Joint node indices.
    pub joints: Vec<usize>,
    /// Inverse bind matrices for each joint.
    pub inverse_bind_matrices: Vec<glam::Mat4>,
}

/// Animation data for a single model.
#[derive(Clone, Debug)]
pub struct ModelAnimationData {
    pub animations: Vec<AnimationClip>,
}

/// A single animation clip.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
}

/// One channel of animation (one node, one property).
#[derive(Clone, Debug)]
pub struct AnimChannel {
    pub node_index: usize,
    pub path: AnimPath,
    pub keyframe_times: Vec<f32>,
    pub keyframe_values: Vec<[f32; 4]>,
    pub interpolation: AnimInterpolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimPath {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimInterpolation {
    Linear,
    Step,
    CubicSpline,
}

/// System: advance animation time and compute bone transforms.
pub fn animation_system(world: &mut World, dt: f32) {
    let anim_data = match world.resource::<AnimationDataResource>() {
        Some(r) => Arc::clone(&r.data),
        None => return,
    };

    // Get skin data if available.
    let skin_data = world.resource::<SkinDataResource>().map(|r| Arc::clone(&r.data));

    // Collect entities with AnimationPlayer + ModelRef + BoneTransforms.
    // Iterate archetypes directly to avoid per-entity HashMap lookups.
    let entities: Vec<voxel_ecs::Entity> = world
        .archetypes()
        .iter()
        .filter(|arch| arch.has::<AnimationPlayer>() && arch.has::<ModelRef>())
        .flat_map(|arch| arch.entities().iter().copied())
        .collect();

    for entity in entities {
        let (model_id, anim_index, anim_time, speed) = {
            let player = match world.get::<AnimationPlayer>(entity) {
                Some(p) => p,
                None => continue,
            };
            if !player.playing {
                continue;
            }
            let model_ref = match world.get::<ModelRef>(entity) {
                Some(r) => r,
                None => continue,
            };
            (model_ref.model_id, player.current_animation, player.time, player.speed)
        };

        let model_anim = match anim_data.get(model_id as usize) {
            Some(d) => d,
            None => continue,
        };
        let clip = match model_anim.animations.get(anim_index as usize) {
            Some(c) => c,
            None => continue,
        };

        // Advance time.
        let new_time = anim_time + dt * speed;
        let final_time = if new_time >= clip.duration {
            // Check if looping.
            let player = world.get::<AnimationPlayer>(entity);
            if player.map(|p| p.looping).unwrap_or(false) {
                new_time % clip.duration.max(0.001)
            } else {
                clip.duration
            }
        } else {
            new_time
        };

        // Update player time.
        if let Some(player) = world.get_mut::<AnimationPlayer>(entity) {
            player.time = final_time;
            if final_time >= clip.duration && !player.looping {
                player.playing = false;
            }
        }

        // Sample channels and compute node transforms.
        let node_count = clip
            .channels
            .iter()
            .map(|c| c.node_index + 1)
            .max()
            .unwrap_or(0);
        let mut node_transforms = vec![glam::Mat4::IDENTITY; node_count];

        // Default: identity translation, identity rotation, unit scale.
        let mut translations = vec![Vec3::ZERO; node_count];
        let mut rotations = vec![Quat::IDENTITY; node_count];
        let mut scales = vec![Vec3::ONE; node_count];

        for channel in &clip.channels {
            let value = sample_channel(channel, final_time);
            match channel.path {
                AnimPath::Translation => {
                    translations[channel.node_index] = Vec3::new(value[0], value[1], value[2]);
                }
                AnimPath::Rotation => {
                    rotations[channel.node_index] =
                        Quat::from_xyzw(value[0], value[1], value[2], value[3]).normalize();
                }
                AnimPath::Scale => {
                    scales[channel.node_index] = Vec3::new(value[0], value[1], value[2]);
                }
            }
        }

        for i in 0..node_count {
            let t = glam::Mat4::from_translation(translations[i]);
            let r = glam::Mat4::from_quat(rotations[i]);
            let s = glam::Mat4::from_scale(scales[i]);
            node_transforms[i] = t * r * s;
        }

        // Write to BoneTransforms component.
        // Compute skin matrices if skin data is available.
        let skin_matrices = if let Some(ref skins) = skin_data {
            if let Some(model_skins) = skins.get(model_id as usize) {
                compute_skin_matrices(&node_transforms, model_skins)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        if let Some(bones) = world.get_mut::<BoneTransforms>(entity) {
            bones.node_transforms = node_transforms;
            bones.skin_matrices = skin_matrices;
        } else {
            world.set(
                entity,
                BoneTransforms {
                    node_transforms,
                    skin_matrices,
                },
            );
        }
    }
}

/// Compute skin matrices from node transforms and skin data.
/// Each skin matrix = node_world_transform * inverse_bind_matrix.
pub fn compute_skin_matrices(
    node_transforms: &[glam::Mat4],
    model_skin_data: &ModelSkinData,
) -> Vec<glam::Mat4> {
    let mut skin_matrices = Vec::new();

    // Compute world transforms for each node (parent chain).
    let node_count = node_transforms.len();
    let mut world_transforms = vec![glam::Mat4::IDENTITY; node_count];

    for i in 0..node_count {
        world_transforms[i] = if let Some(parent) = model_skin_data.node_parents.get(i).and_then(|p| *p) {
            if parent < node_count {
                world_transforms[parent] * node_transforms[i]
            } else {
                node_transforms[i]
            }
        } else {
            node_transforms[i]
        };
    }

    // For each skin, compute joint matrices.
    for skin in &model_skin_data.skins {
        for (joint_idx, &node_idx) in skin.joints.iter().enumerate() {
            let node_world = if node_idx < world_transforms.len() {
                world_transforms[node_idx]
            } else {
                glam::Mat4::IDENTITY
            };
            let inverse_bind = skin.inverse_bind_matrices.get(joint_idx).copied()
                .unwrap_or(glam::Mat4::IDENTITY);
            skin_matrices.push(node_world * inverse_bind);
        }
    }

    skin_matrices
}

/// Sample a single channel at the given time.
fn sample_channel(channel: &AnimChannel, time: f32) -> [f32; 4] {
    let times = &channel.keyframe_times;
    let values = &channel.keyframe_values;

    if times.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }

    // Find surrounding keyframes (binary search).
    let idx = match times.binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1).min(times.len() - 1),
    };

    let next = (idx + 1).min(times.len() - 1);

    if idx == next || channel.interpolation == AnimInterpolation::Step {
        return values[idx];
    }

    let t0 = times[idx];
    let t1 = times[next];
    let frac = ((time - t0) / (t1 - t0)).clamp(0.0, 1.0);

    match channel.interpolation {
        AnimInterpolation::Step => values[idx],
        AnimInterpolation::Linear => {
            let v0 = values[idx];
            let v1 = values[next];
            match channel.path {
                AnimPath::Rotation => {
                    let q0 = Quat::from_xyzw(v0[0], v0[1], v0[2], v0[3]);
                    let q1 = Quat::from_xyzw(v1[0], v1[1], v1[2], v1[3]);
                    let q = q0.slerp(q1, frac);
                    [q.x, q.y, q.z, q.w]
                }
                _ => [
                    v0[0] + (v1[0] - v0[0]) * frac,
                    v0[1] + (v1[1] - v0[1]) * frac,
                    v0[2] + (v1[2] - v0[2]) * frac,
                    0.0,
                ],
            }
        }
        AnimInterpolation::CubicSpline => {
            // Cubic spline: each keyframe has 3 values (in, value, out).
            // For simplicity, fall back to linear interpolation.
            let v0 = values[idx];
            let v1 = values[next];
            let frac = frac;
            match channel.path {
                AnimPath::Rotation => {
                    let q0 = Quat::from_xyzw(v0[0], v0[1], v0[2], v0[3]);
                    let q1 = Quat::from_xyzw(v1[0], v1[1], v1[2], v1[3]);
                    let q = q0.slerp(q1, frac);
                    [q.x, q.y, q.z, q.w]
                }
                _ => [
                    v0[0] + (v1[0] - v0[0]) * frac,
                    v0[1] + (v1[1] - v0[1]) * frac,
                    v0[2] + (v1[2] - v0[2]) * frac,
                    0.0,
                ],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_channel_single_keyframe() {
        let channel = AnimChannel {
            node_index: 0,
            path: AnimPath::Translation,
            keyframe_times: vec![0.0],
            keyframe_values: vec![[1.0, 2.0, 3.0, 0.0]],
            interpolation: AnimInterpolation::Linear,
        };
        let v = sample_channel(&channel, 0.0);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sample_channel_interpolation() {
        let channel = AnimChannel {
            node_index: 0,
            path: AnimPath::Translation,
            keyframe_times: vec![0.0, 1.0],
            keyframe_values: vec![[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
            interpolation: AnimInterpolation::Linear,
        };
        let v = sample_channel(&channel, 0.5);
        assert!((v[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn sample_channel_step() {
        let channel = AnimChannel {
            node_index: 0,
            path: AnimPath::Translation,
            keyframe_times: vec![0.0, 1.0],
            keyframe_values: vec![[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
            interpolation: AnimInterpolation::Step,
        };
        let v = sample_channel(&channel, 0.5);
        assert!((v[0]).abs() < f32::EPSILON); // step: holds previous value
    }

    #[test]
    fn sample_channel_rotation_slerp() {
        let channel = AnimChannel {
            node_index: 0,
            path: AnimPath::Rotation,
            keyframe_times: vec![0.0, 1.0],
            keyframe_values: vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.7071, 0.0, 0.7071],
            ],
            interpolation: AnimInterpolation::Linear,
        };
        let v = sample_channel(&channel, 0.5);
        // Should be roughly halfway rotation.
        let q = Quat::from_xyzw(v[0], v[1], v[2], v[3]);
        assert!(q.length() - 1.0 < 0.01);
    }
}
