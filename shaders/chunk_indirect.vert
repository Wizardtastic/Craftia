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
layout(location = 2) in uint in_tile;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_fog;
layout(location = 2) out vec3 frag_world_pos;
layout(location = 3) flat out uint frag_tile;

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

    // Liquid geometry is kept fixed; water ripples remain in the fragment
    // shader's normals/reflections without moving submerged block textures.

    vec3 world = origin + local;
    gl_Position = push.view_proj * vec4(world, 1.0);
    frag_uv = in_uv;
    frag_tile = in_tile;
    frag_world_pos = world;

    // Matches chunk.vert: fog only in the last 45% of the view distance.
    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    float maxdist = max(cam.cam_pos_and_maxdist.w, 1.0);
    float fog_t = clamp((dist - maxdist * 0.55) / (maxdist * 0.45), 0.0, 1.0);
    frag_fog = fog_t * fog_t * (3.0 - 2.0 * fog_t);
}
