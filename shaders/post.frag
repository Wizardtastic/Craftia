#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;

layout(push_constant) uniform Push {
    vec4 params;     // x = exposure, y = vignette_strength, z = time, w = underwater (0 or 1)
    vec4 ssao;       // x = radius, y = bias, z = strength, w = enabled (0 or 1)
    vec4 proj;       // x = near, y = far, z = screen_w, w = screen_h
} push;

// Bloom constants.
#define BLOOM_INTENSITY    0.25
#define BLOOM_THRESHOLD    0.80
#define BLOOM_PIXEL_RADIUS 1.5
#define BLOOM_RINGS        3

// SSAO constants.
#define SSAO_SAMPLES 32

// ACES filmic tone mapping.
vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

// Linearize depth from [0,1] NDC to view-space Z (positive distance).
float linearize_depth(float d, float near, float far) {
    return near * far / (far - d * (far - near));
}

// Hash-based pseudo-random per-pixel rotation (avoids needing a noise texture).
vec2 hash_noise(vec2 uv) {
    vec3 n = fract(vec3(uv.xyx) * vec3(443.8975, 397.2973, 491.1871));
    n += dot(n, n.yzx + 19.19);
    return fract(vec2((n.x + n.y) * n.z, (n.x + n.z) * n.y)) * 2.0 - 1.0;
}

// Compute SSAO occlusion factor.
float compute_ssao(vec2 uv, float depth_linear, vec3 view_pos) {
    float radius = push.ssao.x;
    float bias   = push.ssao.y;
    float near   = push.proj.x;
    float far    = push.proj.y;
    vec2  res    = push.proj.zw;

    // Reconstruct a pseudo-normal from depth gradients.
    // This works well for axis-aligned voxel geometry.
    vec2 texel = 1.0 / res;
    float dxp = linearize_depth(texture(depth_tex, uv + vec2(texel.x, 0.0)).r, near, far);
    float dxn = linearize_depth(texture(depth_tex, uv - vec2(texel.x, 0.0)).r, near, far);
    float dyp = linearize_depth(texture(depth_tex, uv + vec2(0.0, texel.y)).r, near, far);
    float dyn = linearize_depth(texture(depth_tex, uv - vec2(0.0, texel.y)).r, near, far);
    vec3 normal = normalize(vec3(dxn - dxp, dyn - dyp, 0.002)); // slight z bias for stability

    // Generate a random rotation vector for this pixel.
    vec2 noise = hash_noise(uv * res);

    // Build a TBN matrix to orient samples along the normal.
    // Use Gram-Schmidt with the noise vector.
    vec3 tangent   = normalize(noise.x * vec3(1.0, 0.0, 0.0) + noise.y * vec3(0.0, 1.0, 0.0));
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    // Pre-computed hemisphere kernel (32 samples, cosine-weighted distribution).
    // Hardcoded here to avoid needing a UBO for the kernel.
    const vec3 kernel[SSAO_SAMPLES] = vec3[SSAO_SAMPLES](
        vec3(-0.073, 0.028, 0.071), vec3(0.044, -0.041, 0.063),
        vec3(-0.063, -0.057, 0.045), vec3(0.055, 0.073, 0.025),
        vec3(-0.023, 0.037, 0.094), vec3(0.096, -0.011, 0.019),
        vec3(-0.089, 0.064, 0.031), vec3(0.014, 0.091, 0.047),
        vec3(0.038, -0.074, 0.061), vec3(-0.051, 0.015, 0.088),
        vec3(0.071, 0.043, -0.035), vec3(-0.029, -0.086, 0.042),
        vec3(0.065, -0.033, 0.067), vec3(-0.047, 0.077, 0.021),
        vec3(0.033, 0.021, 0.092), vec3(-0.081, -0.048, 0.044),
        vec3(0.019, 0.062, -0.073), vec3(-0.056, 0.038, 0.068),
        vec3(0.087, -0.026, 0.039), vec3(-0.014, -0.071, 0.083),
        vec3(0.042, 0.089, -0.017), vec3(-0.068, 0.053, 0.058),
        vec3(0.075, -0.061, 0.028), vec3(-0.037, -0.019, 0.097),
        vec3(0.053, 0.034, -0.082), vec3(-0.091, 0.027, 0.048),
        vec3(0.026, -0.088, 0.037), vec3(-0.043, 0.069, -0.061),
        vec3(0.093, 0.017, 0.024), vec3(-0.018, -0.054, 0.079),
        vec3(0.061, 0.056, -0.043), vec3(-0.076, -0.032, 0.065)
    );

    float occlusion = 0.0;
    for (int i = 0; i < SSAO_SAMPLES; i++) {
        // Orient the sample in the hemisphere around the normal.
        vec3 sample_dir = tbn * kernel[i];
        vec3 sample_pos = view_pos + sample_dir * radius;

        // Project sample position back to screen space to get the UV.
        // We approximate the projection by working in depth-space.
        // For a perspective projection, sample_pos.xy / sample_pos.z maps to UV offset.
        vec2 sample_uv = uv + sample_dir.xy * radius / view_pos.z;

        // Clamp to screen bounds.
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        float sample_depth = linearize_depth(texture(depth_tex, sample_uv).r, near, far);

        // Range check: ignore samples that are too far from the current fragment.
        float range_check = smoothstep(0.0, 1.0, radius / abs(depth_linear - sample_depth));

        // If the sampled geometry is closer (smaller Z) than our sample position, it occludes.
        occlusion += (sample_depth <= sample_pos.z - bias ? 1.0 : 0.0) * range_check;
    }

    occlusion = 1.0 - (occlusion / float(SSAO_SAMPLES));
    return occlusion;
}

void main() {
    vec2 uv = frag_uv;

    // Underwater distortion effect.
    if (push.params.w > 0.5) {
        float distortion = sin(uv.x * 100.0 + push.params.z * 2.0) * 0.003;
        float distortion2 = cos(uv.y * 80.0 + push.params.z * 1.5) * 0.002;
        uv.x += distortion;
        uv.y += distortion2;
    }

    // Sample HDR scene and apply exposure.
    vec3 hdr = texture(scene_color, uv).rgb;
    hdr *= push.params.x;

    // --- SSAO ---
    float ssao_factor = 1.0;
    if (push.ssao.w > 0.5) {
        float raw_depth = texture(depth_tex, uv).r;
        float linear_d = linearize_depth(raw_depth, push.proj.x, push.proj.y);

        // Reconstruct approximate view-space position from UV + depth.
        // Z is positive linear depth (matching linearize_depth output).
        // XY are NDC-scaled depth for approximate projection.
        vec3 view_pos = vec3(uv * 2.0 - 1.0, 1.0) * linear_d;

        float ao = compute_ssao(uv, linear_d, view_pos);
        // Mix with full brightness based on strength parameter.
        ssao_factor = mix(1.0, ao, push.ssao.z);
        hdr *= ssao_factor;
    }

    // --- BLOOM: extract brights, blur, composite ---
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

    // Composite bloom onto HDR scene.
    hdr += bloom * BLOOM_INTENSITY;

    // Tonemap, vignette, underwater.
    vec3 color = aces_tonemap(hdr);

    vec2 vignette_uv = frag_uv * 2.0 - 1.0;
    float vignette = 1.0 - dot(vignette_uv, vignette_uv) * push.params.y;
    vignette = clamp(vignette, 0.0, 1.0);
    color *= vignette;

    if (push.params.w > 0.5) {
        float depth_factor = 1.0 - smoothstep(0.05, 0.3, frag_uv.y);
        vec3 water_color = vec3(0.02, 0.05, 0.15) * 2.0;
        color.rgb = mix(color.rgb, water_color, 0.3 * depth_factor);
        color.rgb = mix(color.rgb, vec3(0.1, 0.2, 0.4), 0.15);
    }

    out_color = vec4(color, 1.0);
}
