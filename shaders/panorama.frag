#version 450

// Panorama fragment shader. Samples a cubemap using the view direction.

layout(location = 0) in vec3 frag_dir;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform samplerCube panorama;

void main() {
    vec3 dir = normalize(frag_dir);
    out_color = texture(panorama, dir);
}
