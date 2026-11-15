#version 450
#extension GL_GOOGLE_include_directive : require
// Phase 1 — indirect-draw chunk vertex shader.
//
// Identical shading to chunk.vert except the per-chunk world origin no longer
// comes from push constants (which cannot vary per-draw inside one
// `vkCmdDrawIndexedIndirect`). Instead it is read from a bindless SSBO
// (set 1, binding 1) indexed by `gl_InstanceIndex`, which equals the draw's
// `firstInstance` (= the chunk's stable slot index). `view_proj` and
// `game_time` are global to the whole pass and stay in push constants.
//
// Vertex layout, varyings, and water animation are unchanged from chunk.vert.

// Packed chunk vertex: fixed-point positions/UVs at 1/256 block.
layout(location = 0) in uvec3 in_pos_q;
layout(location = 1) in uvec2 in_uv_q;
layout(location = 2) in uint in_tile;
layout(location = 3) in uint in_water;

const float VERTEX_SCALE = 1.0 / 256.0;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_fog;
layout(location = 2) out vec3 frag_world_pos;
layout(location = 3) flat out uint frag_tile;
layout(location = 4) flat out uint frag_water;
layout(location = 5) out float frag_water_amp;
layout(location = 6) flat out float frag_water_time;

layout(push_constant) uniform Push {
    mat4 view_proj;     // offset 0  (64 bytes)
    vec4 time_pad;      // offset 64; x = game_time (seconds)
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

// Bindless per-draw chunk origins. Indexed by gl_InstanceIndex (== firstInstance).
layout(set = 1, binding = 1) readonly buffer Origins {
    vec4 origins[];     // xyz = chunk world origin, w unused
};

// Generated from the shared water-wave model. Do not hand-edit a local copy.
#include "water_wave.generated.glsl"

void main() {
    vec3 in_pos = vec3(in_pos_q) * VERTEX_SCALE;
    vec2 in_uv = vec2(in_uv_q) * VERTEX_SCALE;
    vec3 world = origins[gl_InstanceIndex].xyz + in_pos;
    // y = analytic water clock (shared with CPU buoyancy).
    float water_time = push.time_pad.y;

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
    frag_tile = in_tile;
    frag_world_pos = world;
    frag_water = in_water;
    frag_water_amp = water_amp;
    frag_water_time = water_time;

    // Matches chunk.vert: fog only in the last 45% of the view distance.
    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    float maxdist = max(cam.cam_pos_and_maxdist.w, 1.0);
    float fog_t = clamp((dist - maxdist * 0.55) / (maxdist * 0.45), 0.0, 1.0);
    frag_fog = fog_t * fog_t * (3.0 - 2.0 * fog_t);
}
