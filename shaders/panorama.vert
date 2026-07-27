#version 450

// Panorama vertex shader. Full-screen triangle (same approach as sky.vert)
// that outputs a cubemap sampling direction from the inverse view-projection.

layout(location = 0) out vec3 frag_dir;

layout(push_constant) uniform Push {
    mat4 inverse_view_proj;
    vec4 camera_pos;     // xyz = camera position, w unused
} push;

void main() {
    // Full-screen triangle from vertex ID.
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.9999, 1.0);

    // Reconstruct world-space view direction.
    vec4 world = push.inverse_view_proj * vec4(pos * 2.0 - 1.0, 1.0, 1.0);
    frag_dir = normalize(world.xyz / world.w - push.camera_pos.xyz);
}
