#version 450

// Chunk vertex shader. Vertex layout (24 bytes):
//   location 0: vec3  pos   - local block corner in [0, 16]
//   location 1: vec2  uv    - local tile-repeat UV in [0, w] x [0, h]
//   location 2: uint  tile  - atlas tile index (passed flat)
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

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_fog;
layout(location = 2) out vec3 frag_world_pos;
layout(location = 3) flat out uint frag_tile;

layout(push_constant) uniform Push {
    vec4 origin_pad;
    mat4 view_proj;
    vec4 time_and_pad;
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

void main() {
    vec3 world = push.origin_pad.xyz + in_pos;
    gl_Position = push.view_proj * vec4(world, 1.0);
    frag_uv = in_uv;
    frag_world_pos = world;
    frag_tile = in_tile;

    // Fog starts at 55% of the view distance and saturates at the far plane.
    float dist = length(world - cam.cam_pos_and_maxdist.xyz);
    float maxdist = max(cam.cam_pos_and_maxdist.w, 1.0);
    float fog_t = clamp((dist - maxdist * 0.55) / (maxdist * 0.45), 0.0, 1.0);
    frag_fog = fog_t * fog_t * (3.0 - 2.0 * fog_t);
}
