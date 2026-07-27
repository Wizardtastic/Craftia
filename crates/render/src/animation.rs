//! Animation data structures parsed from glTF files.
//!
//! Stores keyframe channels and samplers for skeletal animation.

/// All animations from one glTF file.
pub struct AnimationData {
    pub animations: Vec<Animation>,
}

/// A single animation clip (e.g., "walk", "idle").
pub struct Animation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
    pub samplers: Vec<AnimationSampler>,
}

/// Which node and property a channel targets.
pub struct AnimationChannel {
    pub node_index: usize,
    pub path: AnimationPath,
    pub sampler_index: usize,
}

/// What property the animation drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationPath {
    Translation,
    Rotation,
    Scale,
}

/// Keyframe data for one channel.
pub struct AnimationSampler {
    /// Keyframe times in seconds.
    pub input: Vec<f32>,
    /// Keyframe values (quaternion for rotation, vec3 for translation/scale).
    pub output: Vec<[f32; 4]>,
    /// Interpolation method.
    pub interpolation: InterpolationMethod,
}

/// How to interpolate between keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMethod {
    Linear,
    Step,
    CubicSpline,
}

/// Read a float accessor into a Vec<f32>.
fn read_accessor_f32(
    accessor: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Vec<f32> {
    let view = match accessor.view() {
        Some(v) => v,
        None => return Vec::new(),
    };
    let buffer_data = &buffers[view.buffer().index()];
    let offset = view.offset() + accessor.offset();
    let stride = view.stride().unwrap_or(accessor.size());
    let count = accessor.count();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_offset = offset + i * stride;
        if byte_offset + 4 <= buffer_data.len() {
            let val = f32::from_le_bytes([
                buffer_data[byte_offset],
                buffer_data[byte_offset + 1],
                buffer_data[byte_offset + 2],
                buffer_data[byte_offset + 3],
            ]);
            out.push(val);
        }
    }
    out
}

/// Read a Vec3 accessor into Vec<[f32; 3]>.
#[allow(dead_code)]
fn read_accessor_vec3(
    accessor: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Vec<[f32; 3]> {
    let view = match accessor.view() {
        Some(v) => v,
        None => return Vec::new(),
    };
    let buffer_data = &buffers[view.buffer().index()];
    let offset = view.offset() + accessor.offset();
    let stride = view.stride().unwrap_or(accessor.size());
    let count = accessor.count();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_offset = offset + i * stride;
        if byte_offset + 12 <= buffer_data.len() {
            let x = f32::from_le_bytes([
                buffer_data[byte_offset],
                buffer_data[byte_offset + 1],
                buffer_data[byte_offset + 2],
                buffer_data[byte_offset + 3],
            ]);
            let y = f32::from_le_bytes([
                buffer_data[byte_offset + 4],
                buffer_data[byte_offset + 5],
                buffer_data[byte_offset + 6],
                buffer_data[byte_offset + 7],
            ]);
            let z = f32::from_le_bytes([
                buffer_data[byte_offset + 8],
                buffer_data[byte_offset + 9],
                buffer_data[byte_offset + 10],
                buffer_data[byte_offset + 11],
            ]);
            out.push([x, y, z]);
        }
    }
    out
}

/// Read a Vec4 accessor into Vec<[f32; 4]>.
fn read_accessor_vec4(
    accessor: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Vec<[f32; 4]> {
    let view = match accessor.view() {
        Some(v) => v,
        None => return Vec::new(),
    };
    let buffer_data = &buffers[view.buffer().index()];
    let offset = view.offset() + accessor.offset();
    let stride = view.stride().unwrap_or(accessor.size());
    let count = accessor.count();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_offset = offset + i * stride;
        if byte_offset + 16 <= buffer_data.len() {
            let x = f32::from_le_bytes([
                buffer_data[byte_offset],
                buffer_data[byte_offset + 1],
                buffer_data[byte_offset + 2],
                buffer_data[byte_offset + 3],
            ]);
            let y = f32::from_le_bytes([
                buffer_data[byte_offset + 4],
                buffer_data[byte_offset + 5],
                buffer_data[byte_offset + 6],
                buffer_data[byte_offset + 7],
            ]);
            let z = f32::from_le_bytes([
                buffer_data[byte_offset + 8],
                buffer_data[byte_offset + 9],
                buffer_data[byte_offset + 10],
                buffer_data[byte_offset + 11],
            ]);
            let w = f32::from_le_bytes([
                buffer_data[byte_offset + 12],
                buffer_data[byte_offset + 13],
                buffer_data[byte_offset + 14],
                buffer_data[byte_offset + 15],
            ]);
            out.push([x, y, z, w]);
        }
    }
    out
}

/// Parse all animations from a glTF document.
pub fn parse_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> AnimationData {
    let mut animations = Vec::new();

    for anim in document.animations() {
        let name = anim.name().unwrap_or("unnamed").to_string();
        let mut channels = Vec::new();
        let mut samplers = Vec::new();

        for sampler in anim.samplers() {
            let input_accessor = sampler.input();
            let output_accessor = sampler.output();

            let input = read_accessor_f32(&input_accessor, buffers);

            // Output type depends on the channel path.
            // We store everything as Vec<[f32; 4]> for uniformity.
            let output_raw = read_accessor_vec4(&output_accessor, buffers);

            // If the accessor was Vec3 (translation/scale), the 4th component
            // will be 0.0 from the padding. That's fine for our use.

            let interpolation = match sampler.interpolation() {
                gltf::animation::Interpolation::Linear => InterpolationMethod::Linear,
                gltf::animation::Interpolation::Step => InterpolationMethod::Step,
                gltf::animation::Interpolation::CubicSpline => InterpolationMethod::CubicSpline,
            };

            samplers.push(AnimationSampler {
                input,
                output: output_raw,
                interpolation,
            });
        }

        for channel in anim.channels() {
            let path = match channel.target().property() {
                gltf::animation::Property::Translation => AnimationPath::Translation,
                gltf::animation::Property::Rotation => AnimationPath::Rotation,
                gltf::animation::Property::Scale => AnimationPath::Scale,
                _ => continue,
            };

            channels.push(AnimationChannel {
                node_index: channel.target().node().index(),
                path,
                sampler_index: channel.sampler().index(),
            });
        }

        // Compute duration from the longest sampler input.
        let duration = samplers
            .iter()
            .filter_map(|s| s.input.last().copied())
            .fold(0.0f32, f32::max);

        animations.push(Animation {
            name,
            duration,
            channels,
            samplers,
        });
    }

    AnimationData { animations }
}
