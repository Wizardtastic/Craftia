#version 450
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

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in float in_light;
layout(location = 3) in uint in_tile;
layout(location = 4) in vec4 in_light_color;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_light;
layout(location = 2) out float frag_fog;
layout(location = 3) out vec3 frag_world_pos;
layout(location = 4) flat out uint frag_tile;
layout(location = 5) flat out vec4 frag_light_color;

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

void main() {
    vec3 local = in_pos;
    vec3 origin = origins[gl_InstanceIndex].xyz;

    // Water animation: detect water via light > 1.0 (same encoding as chunk.vert).
    if (in_light > 1.0) {
        float water_level = (in_light - 1.0) / 0.5 * 8.0; // 1.0..8.0
        float height_frac = water_level / 8.0;

        vec3 world_no_anim = origin + local;
        float wave = sin(world_no_anim.x * 1.5 + push.time_pad.x * 1.8)
                   * cos(world_no_anim.z * 1.2 + push.time_pad.x * 1.4) * 0.04;
        if (abs(local.y - height_frac) < 0.01) {
            local.y += wave * height_frac;
        }
    }

    vec3 world = origin + local;
    gl_Position = push.view_proj * vec4(world, 1.0);
    frag_uv = in_uv;
    frag_light = in_light;
    frag_tile = in_tile;
    frag_world_pos = world;
    frag_light_color = in_light_color;

    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    frag_fog = clamp(1.0 - exp(-3.0 * dist / cam.cam_pos_and_maxdist.w), 0.0, 1.0);
}
