#version 450

// Particle billboard vertex shader (Phase 2: hard-edged + depth-aware soft
// fade via input attachment).
//
// Vertex binding 0 (32-byte stride, VERTEX rate): a unit quad already uploaded
// to the engine's existing `entity_vbo` (`crate::entity::unit_quad_vertices`).
// We consume only `pos` (vec3, offset 0) and `uv` (vec2, offset 12);
// the trailing 16 bytes of each 32-byte vertex are unused by the particle
// pipeline.
//
// Per-instance binding 1 (32-byte stride, INSTANCE rate): a
// `crate::render::particle::ParticleInstance` layout:
//   location 2: vec3  pos      — world-space particle position
//   location 3: float rot      — billboard rotation around view-forward axis
//   location 4: uint  color    — packed RGBA8 colour (unpackUnorm4x8)
//   location 5: float size     — half-extent in world units
//   location 6: float tile     — atlas tile index (0..255); the fragment
//                                shader does the tile-coord remapping.
//
// Push constants (80 bytes total):
//   offset  0: mat4 inv_view_proj    (64 bytes / 16 floats)
//   offset 64: vec4 soft_near_far    (x = softness world-units for fade,
//                                    y = camera near, z = camera far,
//                                    w = reserved)

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 inst_pos;
layout(location = 3) in float inst_rot;
layout(location = 4) in uint inst_color;
layout(location = 5) in float inst_size;
layout(location = 6) in float inst_tile;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;
layout(location = 2) flat out float frag_tile;

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 soft_near_far;
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist;
} cam;

void main() {
    // Camera-aligned billboard basis from the inverse view-projection's
    // rotation columns. `inv_view = inv_proj * inv_view`; the rotational
    // columns of `inv_view` carry the camera-to-world basis in world space.
    vec3 right = normalize(vec3(push.inv_view_proj[0][0], push.inv_view_proj[0][1], push.inv_view_proj[0][2]));
    vec3 up    = normalize(vec3(push.inv_view_proj[1][0], push.inv_view_proj[1][1], push.inv_view_proj[1][2]));

    // Per-particle rotation around the view-forward axis (right × up).
    vec3 forward = normalize(cross(right, up));
    float c = cos(inst_rot);
    float s = sin(inst_rot);
    vec3 rotated_right = right * c + forward * s;
    vec3 rotated_up    = up    * c - forward * s;

    // Build the world-space quad vertex and project through the inverse VP
    // matrix supplied as a push constant. The quad's local XY are in
    // [-1, +1] (since `entity::unit_quad_vertices` builds them in [-0.5, +0.5]
    // we still treat them as symmetric; the shader multiplies by `inst_size`).
    vec3 world = inst_pos
        + rotated_right * (in_pos.x * inst_size)
        + rotated_up    * (in_pos.y * inst_size);

    gl_Position = push.inv_view_proj * vec4(world, 1.0);

    frag_uv = in_uv;
    // Unpack RGBA8 (R in lowest byte, A in highest) → vec4 in [0, 1].
    frag_color = unpackUnorm4x8(inst_color);
    frag_tile = inst_tile;
}
