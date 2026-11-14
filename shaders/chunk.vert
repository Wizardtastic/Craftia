#version 450
#extension GL_GOOGLE_include_directive : require

// Chunk vertex shader. Vertex layout (28 bytes):
//   location 0: vec3  pos   - local block corner in [0, 16]
//   location 1: vec2  uv    - local tile-repeat UV in [0, w] x [0, h]
//   location 2: uint  tile  - atlas tile index (passed flat)
//   location 3: uint  water - packed water body / surface animation
//
// All lighting is evaluated per pixel in chunk.frag by the GPU voxel ray
// tracer; nothing is baked into the geometry.
//
// Push constants:
//   offset  0: vec4 origin_pad (xyz = chunk world origin)
//   offset 16: mat4 view_proj
//   offset 80: vec4 time_and_pad (x = game_time seconds)

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in uint in_tile;
layout(location = 3) in uint in_water;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_fog;
layout(location = 2) out vec3 frag_world_pos;
layout(location = 3) flat out uint frag_tile;
layout(location = 4) flat out uint frag_water;
layout(location = 5) out float frag_water_amp;
layout(location = 6) flat out float frag_water_time;

layout(push_constant) uniform Push {
    vec4 origin_pad;
    mat4 view_proj;
    vec4 time_and_pad;
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

// Generated from crates/world/src/water_wave_model.rs by the render build
// script. The CPU uses the same model, so this path cannot drift by hand.
#include "water_wave.generated.glsl"

void main() {
    vec3 world = push.origin_pad.xyz + in_pos;
    // The water clock is separate from the day/night clock; it is the same
    // value the CPU uses for buoyancy.
    float water_time = push.time_and_pad.y;

    // Only the top cell of a water body is displaced, and only if the mesher
    // marked this vertex as a surface corner. Vertical displacement keeps the
    // quad grid watertight because every shared corner evaluates the same
    // analytic field with the same per-corner amplitude.
    uint water_kind = in_water & 7u;
    bool water_surface = (in_water & 8u) != 0u;
    float water_amp = float((in_water >> 8) & 255u) / 32.0;
    if (water_surface && water_kind != 0u && water_amp > 0.0) {
        float angle = float((in_water >> 16) & 255u) * (WATER_TAU / 255.0);
        float rapids = float((in_water >> 4) & 15u) / 15.0;
        world.y += water_wave_height(water_kind, water_amp, angle, rapids,
                                     world.xz, water_time);
    }

    gl_Position = push.view_proj * vec4(world, 1.0);
    frag_uv = in_uv;
    frag_world_pos = world;
    frag_tile = in_tile;
    frag_water = in_water;
    frag_water_amp = water_amp;
    frag_water_time = water_time;

    // Fog starts at 55% of the view distance and saturates at the far plane.
    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    float maxdist = max(cam.cam_pos_and_maxdist.w, 1.0);
    float fog_t = clamp((dist - maxdist * 0.55) / (maxdist * 0.45), 0.0, 1.0);
    frag_fog = fog_t * fog_t * (3.0 - 2.0 * fog_t);
}
