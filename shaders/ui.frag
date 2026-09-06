#version 450

// UI fragment shader. Samples one of four textures based on tex_id:
//   0 → block atlas  (hotbar icons, block icons, quads)
//   1 → font atlas   (text)
//   2 → minimap texture (terrain map)
//   3 → UI atlas     (beveled buttons, inset slots, panels, HUD icons)
// Multiplies the sampled colour by the per-vertex colour tint.

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in float frag_tex_id;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D block_atlas;
layout(set = 0, binding = 1) uniform sampler2D font_atlas;
layout(set = 0, binding = 2) uniform sampler2D minimap_tex;
layout(set = 0, binding = 3) uniform sampler2D ui_atlas;

void main() {
    vec4 tex;
    if (frag_tex_id < 0.5) {
        tex = texture(block_atlas, frag_uv);
    } else if (frag_tex_id < 1.5) {
        tex = texture(font_atlas, frag_uv);
    } else if (frag_tex_id < 2.5) {
        tex = texture(minimap_tex, frag_uv);
    } else {
        tex = texture(ui_atlas, frag_uv);
    }
    out_color = tex * frag_color;
}
