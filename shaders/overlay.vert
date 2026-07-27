#version 450

// Overlay vertex shader: transforms world-space line positions by the
// view-projection matrix and passes through per-vertex color.

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    gl_Position = pc.view_proj * vec4(in_pos, 1.0);
    out_color = in_color;
}
