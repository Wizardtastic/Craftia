#version 450

// Entity fragment shader. Samples the texture atlas at the per-tile UV.
// Identical to chunk.frag but without shadow mapping or water animation.

const float ATLAS_TILES = 16.0;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in float frag_light;
layout(location = 2) in float frag_fog;
layout(location = 3) flat in uint frag_tile;
layout(location = 4) flat in vec4 frag_light_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist;
} cam;

layout(set = 0, binding = 1) uniform sampler2D atlas;

layout(set = 0, binding = 2) uniform Fog {
    vec4 color_and_density;
    vec4 ambient_and_sun;
} fog;

void main() {
    ivec2 tile_pos = ivec2(frag_tile % 16u, frag_tile / 16u);
    vec2 atlas_uv = (vec2(tile_pos) + vec2(fract(frag_uv.x), fract(frag_uv.y))) / ATLAS_TILES;

    vec4 tex = texture(atlas, atlas_uv);
    if (tex.a < 0.1) {
        discard;
    }

    float ambient = fog.ambient_and_sun.x;
    vec3 lit = tex.rgb * frag_light * ambient;
    // Same pure-multiplicative tint as chunk.frag (see comment there).
    // Additive emissive term scales with `lit` so dark-textured entities
    // don't falsely glow in torch color near a coloured light source.
    vec3 emissive_contribution = lit * frag_light_color.rgb * (frag_light * 0.18);
    vec3 tinted = lit * frag_light_color.rgb + emissive_contribution;
    vec3 final_color = mix(tinted, fog.color_and_density.rgb, frag_fog);
    out_color = vec4(final_color, tex.a);
}
