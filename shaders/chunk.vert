#version 450

// Chunk vertex shader. Vertex layout (32 bytes):
//   location 0: vec3  pos    — local block corner in [0, 16]
//   location 1: vec2  uv     — LOCAL tile-repeat UV in `[0, w] × [0, h]`
//                              (the fragment shader applies `fract()` and
//                              adds the tile origin before sampling).
//   location 2: float light  — baked face shading in [0, 1]
//                              (values > 1.0 signal water: 1.0 + (level/8)*0.5)
//   location 3: uint  tile   — atlas tile index (passed `flat` so the
//                              fragment shader can pick the per-quad tile).
//   location 4: vec4  light_color — packed RGBA light tint (passed `flat`)
//
// Push constants (offset 0): vec4 chunk_origin_and_pad = (originX, originY, originZ, _)
// Push constants (offset 16): mat4 view_proj
// Push constants (offset 80): vec4 time_and_pad = (game_time, _, _, _)
//
// World position = chunk_origin + local_pos. Clip = view_proj * world_pos.

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
    vec4 origin_pad;   // xyz = chunk world origin, w unused
    mat4 view_proj;
    vec4 time_and_pad; // x = game_time (seconds)
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

void main() {
    vec3 local = in_pos;

    // Liquid geometry is kept fixed; water ripples remain in the fragment
    // shader's normals/reflections without moving submerged block textures.

    vec3 world = push.origin_pad.xyz + local;
    gl_Position = push.view_proj * vec4(world, 1.0);
    frag_uv = in_uv;
    frag_light = in_light;
    frag_tile = in_tile;
    frag_world_pos = world;
    frag_light_color = in_light_color;

    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    frag_fog = clamp(1.0 - exp(-3.0 * dist / cam.cam_pos_and_maxdist.w), 0.0, 1.0);
}
