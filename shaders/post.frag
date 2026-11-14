#version 450

// Post-processing: exposure, HDR bloom, ACES tone mapping, vignette and the
// underwater colour grade. Ambient occlusion is no longer a screen-space
// effect - it is ray traced per pixel in chunk.frag against the voxel world.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color;

layout(push_constant) uniform Push {
    vec4 params; // x = exposure, y = vignette_strength, z = time, w = underwater
} push;

#define BLOOM_INTENSITY    0.22
#define BLOOM_THRESHOLD    1.05
#define BLOOM_PIXEL_RADIUS 5.0
#define BLOOM_RINGS        2

vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main() {
    vec2 uv = frag_uv;

    vec3 hdr = texture(scene_color, uv).rgb;
    hdr *= push.params.x;

    // Bloom: extract bright HDR pixels and blur sparsely.
    vec2 texel = 1.0 / vec2(textureSize(scene_color, 0));
    vec3 bloom = vec3(0.0);
    float total_weight = 0.0;

    for (int ring = 1; ring <= BLOOM_RINGS; ring++) {
        float radius = BLOOM_PIXEL_RADIUS * float(ring);
        float weight = 1.0 / (float(ring) * float(ring));
        vec2 offsets[4] = vec2[4](
            vec2(radius, 0.0),
            vec2(-radius, 0.0),
            vec2(0.0, radius),
            vec2(0.0, -radius)
        );
        for (int d = 0; d < 4; d++) {
            vec2 sample_uv = uv + offsets[d] * texel;
            vec3 s = texture(scene_color, sample_uv).rgb * push.params.x;
            s = max(s - vec3(BLOOM_THRESHOLD), vec3(0.0));
            bloom += s * weight;
        }
        total_weight += weight * 4.0;
    }
    bloom /= total_weight;
    hdr += bloom * BLOOM_INTENSITY;

    vec3 color = aces_tonemap(hdr);

    vec2 vignette_uv = frag_uv * 2.0 - 1.0;
    float vignette = 1.0 - dot(vignette_uv, vignette_uv) * push.params.y;
    color *= clamp(vignette, 0.0, 1.0);

    if (push.params.w > 0.5) {
        float depth_factor = 1.0 - smoothstep(0.05, 0.3, frag_uv.y);
        vec3 water_color = vec3(0.02, 0.05, 0.15) * 2.0;
        color.rgb = mix(color.rgb, water_color, 0.3 * depth_factor);
        color.rgb = mix(color.rgb, vec3(0.1, 0.2, 0.4), 0.15);
    }

    out_color = vec4(color, 1.0);
}
