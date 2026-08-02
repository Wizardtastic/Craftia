//! Keyframe sampling: evaluate animation channels at a given time.

use glam::{Mat4, Quat, Vec3};

use crate::data::{AnimationChannel, Interpolation, TargetedProperty};
use crate::interpolate::{cubic_hermite4, lerp4, slerp};

/// Sample a channel at the given time, returning the raw [f32; 4] value.
pub fn sample_channel(channel: &AnimationChannel, time: f32) -> [f32; 4] {
    if channel.keyframe_times.is_empty() {
        return [0.0; 4];
    }
    if channel.keyframe_times.len() == 1 {
        return channel.keyframe_values[0];
    }

    let duration = channel.keyframe_times.last().copied().unwrap_or(0.0);
    let clamped = time.clamp(0.0, duration);

    // Binary search for the keyframe index.
    let idx = channel
        .keyframe_times
        .partition_point(|&t| t <= clamped)
        .saturating_sub(1);
    let next_idx = (idx + 1).min(channel.keyframe_values.len() - 1);

    if idx == next_idx {
        return channel.keyframe_values[idx];
    }

    let t0 = channel.keyframe_times[idx];
    let t1 = channel.keyframe_times[next_idx];
    let segment_duration = t1 - t0;
    let t = if segment_duration > 0.0 {
        (clamped - t0) / segment_duration
    } else {
        0.0
    };

    match channel.interpolation {
        Interpolation::Step => channel.keyframe_values[idx],
        Interpolation::Linear => {
            // Use slerp for rotations, lerp for everything else.
            match channel.property {
                TargetedProperty::Rotation => slerp(
                    channel.keyframe_values[idx],
                    channel.keyframe_values[next_idx],
                    t,
                ),
                _ => lerp4(
                    channel.keyframe_values[idx],
                    channel.keyframe_values[next_idx],
                    t,
                ),
            }
        }
        Interpolation::CubicSpline => {
            // Cubic spline: keyframes store [in-tangent, value, out-tangent] triplets.
            // For simplicity, use Hermite with tangents from adjacent keyframes.
            let p0 = channel.keyframe_values[idx];
            let p1 = channel.keyframe_values[next_idx];
            // Approximate tangents from neighboring values.
            let m0 = if idx > 0 {
                let prev = channel.keyframe_values[idx - 1];
                [
                    p0[0] - prev[0],
                    p0[1] - prev[1],
                    p0[2] - prev[2],
                    p0[3] - prev[3],
                ]
            } else {
                [0.0; 4]
            };
            let m1 = if next_idx < channel.keyframe_values.len() - 1 {
                let next_next = channel.keyframe_values[next_idx + 1];
                [
                    next_next[0] - p1[0],
                    next_next[1] - p1[1],
                    next_next[2] - p1[2],
                    next_next[3] - p1[3],
                ]
            } else {
                [0.0; 4]
            };
            cubic_hermite4(p0, m0, p1, m1, t)
        }
        Interpolation::Bezier => {
            // Fallback to linear for now.
            lerp4(
                channel.keyframe_values[idx],
                channel.keyframe_values[next_idx],
                t,
            )
        }
    }
}

/// Sample a Translation channel, returning a Vec3.
pub fn sample_translation(channel: &AnimationChannel, time: f32) -> Vec3 {
    let v = sample_channel(channel, time);
    Vec3::new(v[0], v[1], v[2])
}

/// Sample a Rotation channel, returning a Quat.
pub fn sample_rotation(channel: &AnimationChannel, time: f32) -> Quat {
    let v = sample_channel(channel, time);
    Quat::from_xyzw(v[0], v[1], v[2], v[3])
}

/// Sample a Scale channel, returning a Vec3.
pub fn sample_scale(channel: &AnimationChannel, time: f32) -> Vec3 {
    let v = sample_channel(channel, time);
    Vec3::new(v[0], v[1], v[2])
}

/// Convert TRS (translation, rotation, scale) to a 4x4 matrix.
pub fn trs_to_matrix(translation: Vec3, rotation: Quat, scale: Vec3) -> Mat4 {
    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}

/// Compute node transforms for all nodes in a clip at the given time.
/// Returns a Vec of Mat4, one per node index.
pub fn evaluate_clip(clip: &crate::data::AnimationClip, time: f32, node_count: usize) -> Vec<Mat4> {
    let mut translations = vec![Vec3::ZERO; node_count];
    let mut rotations = vec![Quat::IDENTITY; node_count];
    let mut scales = vec![Vec3::ONE; node_count];

    // Apply channel values.
    for channel in &clip.channels {
        if channel.target_node >= node_count {
            continue;
        }
        match channel.property {
            TargetedProperty::Translation => {
                translations[channel.target_node] = sample_translation(channel, time);
            }
            TargetedProperty::Rotation => {
                rotations[channel.target_node] = sample_rotation(channel, time);
            }
            TargetedProperty::Scale => {
                scales[channel.target_node] = sample_scale(channel, time);
            }
            _ => {
                // Other property types handled by property animation system.
            }
        }
    }

    // Build TRS matrices.
    (0..node_count)
        .map(|i| trs_to_matrix(translations[i], rotations[i], scales[i]))
        .collect()
}
