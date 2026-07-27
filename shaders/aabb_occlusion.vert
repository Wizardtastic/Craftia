#version 450

// Push constants matching the chunk pipeline layout (96 bytes):
//   vec4  origin   (offset  0) – x,y,z = chunk AABB min; w unused
//   mat4  vp       (offset 16) – view-projection matrix
//   vec4  time     (offset 80) – game_time, underwater, 0, 0 (not used here)
layout(push_constant) uniform Push {
    vec4 origin;   // .xyz = chunk origin (AABB min)
    mat4 vp;       // view-projection
    vec4 time;     // unused in this shader
} push;

const float CHUNK_SIZE = 16.0;

// 8 cube vertices procedurally generated from the 3-bit index.
// Each bit selects min(0) or max(1) along one axis.
void main() {
    // Extract which corner of the cube this vertex represents.
    // gl_VertexIndex 0..7 maps to the 8 corners of a unit cube.
    vec3 corner = vec3(
        float((gl_VertexIndex >> 0) & 1),
        float((gl_VertexIndex >> 1) & 1),
        float((gl_VertexIndex >> 2) & 1)
    );

    // Compute AABB from chunk origin + chunk size (instead of reading from
    // push constants, which contain game_time at this offset, not extent).
    vec3 aabb_min = push.origin.xyz;
    vec3 aabb_max = push.origin.xyz + CHUNK_SIZE;
    vec3 world_pos = mix(aabb_min, aabb_max, corner);

    gl_Position = push.vp * vec4(world_pos, 1.0);
}
