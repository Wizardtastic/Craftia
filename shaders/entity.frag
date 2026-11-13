#version 450

// Entity fragment shader. Samples the texture atlas at the per-tile UV.
// Uses the same physically-based sun/sky/torch lighting model as chunk.frag
// (without shadow maps — entities are small and self-contained).

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
    vec4 color_and_density;   // rgb = fog colour
    vec4 sun_color;           // rgb = direct sun radiance, a = visibility
    vec4 sky_ambient;         // rgb = sky ambient irradiance, a = night factor
    vec4 sun_dir_pad;         // xyz = sun direction
} fog;

void main() {
    ivec2 tile_pos = ivec2(frag_tile % 16u, frag_tile / 16u);
    vec2 atlas_uv = (vec2(tile_pos) + vec2(fract(frag_uv.x), fract(frag_uv.y))) / ATLAS_TILES;

    vec4 tex = texture(atlas, atlas_uv);
    if (tex.a < 0.1) {
        discard;
    }

    // `frag_light` is sky visibility; torch level rides in the tint's alpha.
    float sky_vis = clamp(frag_light, 0.0, 1.0);
    float torch_level = frag_light_color.a;

    vec3 lighting = (fog.sun_color.rgb * fog.sun_color.a + fog.sky_ambient.rgb) * sky_vis
                  + frag_light_color.rgb * (torch_level * torch_level) * 2.4;
    // Same sRGB → linear albedo conversion as chunk.frag.
    vec3 albedo = pow(tex.rgb, vec3(2.2));
    vec3 final_color = mix(albedo * lighting, fog.color_and_density.rgb, frag_fog);
    out_color = vec4(final_color, tex.a);
}
