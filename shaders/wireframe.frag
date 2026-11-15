#version 450
// Flat unlit colour for F4 mesh mode. The line pipeline reuses the packed
// chunk vertex shader, but skips the expensive ray-traced chunk fragment path
// entirely: every line is one constant colour.

// Match the chunk vertex shader's full output interface. The values are not
// used, but declaring them keeps the pipeline interface validation-clean.
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in float frag_fog;
layout(location = 2) in vec3 frag_world_pos;
layout(location = 3) flat in uint frag_tile;
layout(location = 4) flat in uint frag_water;
layout(location = 5) in float frag_water_amp;
layout(location = 6) flat in float frag_water_time;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(0.72, 0.92, 1.0, 1.0);
}
