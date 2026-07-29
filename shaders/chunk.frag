#version 450

#extension GL_EXT_nonuniform_qualifier : enable

// Chunk fragment shader. Samples the texture atlas at the per-tile UV
// derived from `frag_tile + fract(frag_uv)`, applies baked face light, and
// blends toward a sky fog colour based on frag_fog.
//
// Atlas addressing: `frag_uv` carries LOCAL tile-repeat coordinates in
// `[0, w] x [0, h]`. `frag_tile` is the atlas tile index passed flat from
// the vertex stage. We compute `atlas_uv = (tile_origin + fract(uv)) / 16`
// so a `w`-block-wide merged quad samples the same tile `w` times instead
// of bleeding into the neighbouring atlas tiles on a 2D atlas (this is the
// fix for the "textures splattered everywhere" regression).
// The atlas sampler MUST be NEAREST + CLAMP_TO_EDGE (see crates/render/src/texture.rs). With LINEAR filtering, the fract() one-texel seam at integer UV boundaries would bleed into the same tile. With REPEAT, fract() wraps the tile internally (correct by accident, but NEAREST must stay to avoid sub-texel artifacts across tile bounds).

const float ATLAS_TILES = 16.0;

layout(constant_id = 0) const bool SHADOW_ENABLED = false;
layout(constant_id = 1) const bool TILE_REMAP_ENABLED = false;

// ── Tile material lookup table (binding 5) ────────────────────────────────
// Per-tile material parameters driven from the world block registry. Each
// entry is 16 bytes; the table is std430-aligned with u32 (align-of == 4)
// so accesses are unaligned-safe on Vulkan. `world_params` carries global
// scalars updated each frame by the engine:
//
//   .x = water surface Y (world units) — used by caustics + wet-edge paths
//   .y = wet_edge_strength             — controls cool tint amount
//   .z = caustics_strength             — controls submerged sun glare
//   .w = leaves_sss_strength           — controls chlorophyll backlight
struct BlockMaterialGpu {
    uint flags_roughness_emissive_pad; // bits 0..7 flags, 8..15 roughness, 16..23 emissive
    uint sss_tint;                     // RGBA8 packed leaves-tint
    uint wet_tint;                     // RGBA8 packed wet-edge tint
    uint absorption_pad;               // RGB8 absorption + 8-bit pad
};
layout(set = 0, binding = 5) uniform MaterialTable {
    BlockMaterialGpu materials[256];
    vec4 world_params; // see comment above
} material_table;

const uint MATERIAL_FLAG_LEAVES_SSS       = 1u << 0u;
const uint MATERIAL_FLAG_TRANSLUCENT_ABSORB = 1u << 1u;
const uint MATERIAL_FLAG_WATER             = 1u << 2u;
const uint MATERIAL_FLAG_REFLECTIVE        = 1u << 3u;

// Push constants (same layout as chunk.vert). The fragment stage reads
// `view_proj` (SSR ray-march projection + per-fragment eye depth) and
// `time_and_pad.x` (wave animation). The pipeline layout's push range covers
// VERTEX|FRAGMENT so this declaration is valid for both stages.
layout(push_constant) uniform Push {
    vec4 origin_pad;   // xyz = chunk world origin, w unused
    mat4 view_proj;
    vec4 time_and_pad; // x = game_time (seconds)
} push;

// ── Reflections (slice 3) ─────────────────────────────────────────────────
// Binding 7: single-sample copy of the scene depth, made alongside the
// binding-6 colour copy after the main pass. Sampled by the SSR ray-march
// and by the water-column Beer absorption. NEAREST sampler.
layout(set = 0, binding = 7) uniform sampler2D scene_depth;
// Binding 8: per-frame reflection/environment UBO. `sky_horizon`/`sky_zenith`
// mirror the sky UBO so reflected rays can evaluate the SAME analytic sky
// the sky pass would render — no cubemap probe needed.
layout(set = 0, binding = 8) uniform Reflection {
    vec4 sky_horizon;   // rgb = horizon colour
    vec4 sky_zenith;    // rgb = zenith colour
    vec4 sun_dir_str;   // xyz = sun dir, w = master reflection strength (0..1)
    vec4 proj_misc;     // x = near, y = far, z = camera underwater, w = SSR valid
} refl;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in float frag_light;
layout(location = 2) in float frag_fog;
layout(location = 3) in vec3 frag_world_pos;
layout(location = 4) flat in uint frag_tile;
layout(location = 5) flat in vec4 frag_light_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist; // xyz = camera pos, w = fog max distance
} cam;

layout(set = 0, binding = 1) uniform sampler2D atlas;

layout(set = 0, binding = 2) uniform Fog {
    vec4 color_and_density;   // rgb = fog colour, a unused
    vec4 ambient_and_sun;     // x = ambient brightness, yzw = sun direction
} fog;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadow_map;

layout(set = 0, binding = 4) uniform ShadowData {
    mat4 cascade_vps[4];      // 4 light-space view-projection matrices (256 bytes)
    vec4 cascade_splits;       // far-plane distance per cascade
    vec4 light_dir_and_bias;   // xyz = light direction, w = shadow bias
} shadow;

// ── Scene opaque color copy (binding 6, slice 2) ────────────────────────
// Populated by a `vkCmdCopyImage` from the main render pass's offscreen color
// attachment into a side-car image at the same extent. Both glass tinted
// absorption (TRANSLUCENT_ABSORB) and water refraction sample this, so the
// translucent mesh can blend against / refract to whatever opaque geometry
// the player was looking at. The sampler's UV is `gl_FragCoord.xy /
// vec2(screen_w, screen_h)`.
layout(set = 0, binding = 6) uniform sampler2D scene_opaque;

// Tile remap UBO for animated block textures (set 1, binding 0).
// When enabled, remaps canonical tile indices to current frame tile indices.
layout(set = 1, binding = 0) uniform TileRemap {
    uint map[256];
} tile_remap;

// 3x3 percentage-closer filtering against the selected cascade. Returns 1.0
// for fully lit fragments and 0.0 for fully occluded ones. Fragments outside
// the shadow map's valid range are treated as lit.
float compute_shadow_factor(vec3 world_pos, float view_depth) {
    int cascade_idx = 0;
    if (view_depth < shadow.cascade_splits.x) cascade_idx = 0;
    else if (view_depth < shadow.cascade_splits.y) cascade_idx = 1;
    else if (view_depth < shadow.cascade_splits.z) cascade_idx = 2;
    else cascade_idx = 3;

    vec4 light_pos = shadow.cascade_vps[cascade_idx] * vec4(world_pos, 1.0);
    vec3 proj_coords = light_pos.xyz / light_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        proj_coords.z > 1.0) {
        return 1.0;
    }

    float bias = shadow.light_dir_and_bias.w;
    float current_depth = proj_coords.z;

    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0).xy);
    float shadow_accum = 0.0;
    // 4-tap cross pattern instead of 3x3 grid (halves shadow sampling cost).
    shadow_accum += texture(shadow_map, vec4(proj_coords.xy, float(cascade_idx), current_depth - bias));
    shadow_accum += texture(shadow_map, vec4(proj_coords.xy + vec2(1.0, 0.0) * texel_size, float(cascade_idx), current_depth - bias));
    shadow_accum += texture(shadow_map, vec4(proj_coords.xy + vec2(-1.0, 0.0) * texel_size, float(cascade_idx), current_depth - bias));
    shadow_accum += texture(shadow_map, vec4(proj_coords.xy + vec2(0.0, 1.0) * texel_size, float(cascade_idx), current_depth - bias));
    shadow_accum += texture(shadow_map, vec4(proj_coords.xy + vec2(0.0, -1.0) * texel_size, float(cascade_idx), current_depth - bias));
    shadow_accum /= 5.0;

    return shadow_accum;
}

// ── Reflection helpers (slice 3) ──────────────────────────────────────────

// Linearize a [0,1] depth-buffer value to eye depth (matches post.frag and
// the glam perspective_rh [0,1] projection: near*far / (far - d*(far-near))).
float linearize_depth(float d) {
    float near = refl.proj_misc.x;
    float far = refl.proj_misc.y;
    return near * far / (far - d * (far - near));
}

// Analytic sky for a reflected ray — an exact mirror of sky.frag's gradient +
// sun disc/halo (stars omitted: negligible in a reflection). Feeds every
// reflection path as the ray-miss fallback, replacing the spec'd per-frame
// cubemap probe at zero per-frame cost.
vec3 sky_reflection_color(vec3 dir) {
    dir = normalize(dir);
    float up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 c = mix(refl.sky_horizon.rgb, refl.sky_zenith.rgb, up);
    vec3 sd = normalize(refl.sun_dir_str.xyz);
    if (sd.y > 0.0) {
        float sun_dot = max(dot(dir, sd), 0.0);
        c += vec3(1.0, 0.95, 0.8) * (pow(sun_dot, 256.0) + pow(sun_dot, 8.0) * 0.3);
    }
    return c;
}

// Procedural water-surface normal. Octave 1 uses the same frequencies as the
// vertex shader's displacement so the shading normal agrees with the mesh's
// silhouette; octave 2 adds fragment-only detail. The derivative of
// h(x,z) = A·sin(px)·cos(qz) is analytic: dh/dx = A·kx·cos(px)·cos(qz),
// dh/dz = −A·kz·sin(px)·sin(qz).
vec3 water_wave_normal(vec3 wp, float t) {
    // Octave 1 (matches vertex wave; amplitude exaggerated ~2.5x for shading).
    float a1 = 0.04 * 2.5;
    float p1 = wp.x * 1.5 + t * 1.8;
    float q1 = wp.z * 1.2 + t * 1.4;
    float dhdx = a1 * 1.5 * cos(p1) * cos(q1);
    float dhdz = -a1 * 1.2 * sin(p1) * sin(q1);
    // Octave 2 (finer ripple, fragment only).
    float a2 = 0.022;
    float p2 = wp.x * 3.9 + t * 2.7;
    float q2 = wp.z * 3.3 + t * 2.2;
    dhdx += a2 * 3.9 * cos(p2) * cos(q2);
    dhdz += -a2 * 3.3 * sin(p2) * sin(q2);
    return normalize(vec3(-dhdx, 1.0, -dhdz));
}

// Flat face normal from screen-space derivatives. For axis-aligned voxel
// faces the cross of dFdx/dFdy IS the cube-face normal (up to sign); the
// caller flips it toward the camera. Used by glass + opaque reflective tiles.
vec3 face_normal(vec3 view_dir) {
    vec3 n = normalize(cross(dFdx(frag_world_pos), dFdy(frag_world_pos)));
    return (dot(n, view_dir) > 0.0) ? -n : n;
}

// Screen-space reflection ray-march against the binding-7 scene depth copy.
// Returns vec4(rgb, confidence); confidence is 0 on miss (caller falls back
// to the analytic sky) and fades near screen edges where the ray would
// sample off-screen geometry it cannot see.
//
// Marching happens in world space; each step is projected with push.view_proj
// to (uv, eye_depth). Eye depth of the marched point is clip.w (w_clip =
// -z_view for the RH perspective), directly comparable to linearize_depth().
// The scene depth contains only the OPAQUE pass (copied before the
// transparent pass), so water never self-occludes and the lake bed / shores
// are valid hit targets.
vec4 ssr_raymarch(vec3 origin, vec3 dir, float view_dist) {
    // Scale ray-march quality with fragment distance: close water gets the
    // full 16-step budget; distant water steps down to save GPU time.
    int max_steps = view_dist < 30.0 ? 16 : (view_dist < 60.0 ? 8 : 4);
    float step_len = 0.45;
    // Per-pixel jitter breaks up the banding of the coarse fixed-step march.
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 p = origin + dir * (0.55 + jitter * 0.35);
    for (int i = 0; i < 16; i++) {
        if (i >= max_steps) break;
        vec4 clip = push.view_proj * vec4(p, 1.0);
        if (clip.w <= 0.0) return vec4(0.0);           // marched behind camera
        vec2 uv = vec2(clip.x / clip.w * 0.5 + 0.5,
                       0.5 - clip.y / clip.w * 0.5);   // negative-height viewport flip
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
        float ray_d = clip.w;
        float scene_d = linearize_depth(textureLod(scene_depth, uv, 0.0).r);
        if (ray_d > scene_d) {
            // Penetrated the scene surface. Accept the hit only within a
            // thickness window so we don't glue reflections to geometry the
            // ray passed far behind.
            if (ray_d - scene_d < 1.2) {
                vec3 col = textureLod(scene_opaque, uv, 0.0).rgb;
                vec2 edge = min(uv, 1.0 - uv);
                float fade = smoothstep(0.0, 0.10, min(edge.x, edge.y));
                return vec4(col, fade);
            }
            return vec4(0.0);
        }
        p += dir * step_len;
        step_len *= 1.15;
    }
    return vec4(0.0);
}

void main() {
    // Remap tile index if animated block textures are enabled.
    uint actual_tile = frag_tile;
    if (TILE_REMAP_ENABLED) {
        actual_tile = tile_remap.map[frag_tile];
    }

    // Per-tile atlas UV: offset by the tile's grid position and wrap the
    // local repeat coordinates back into `[0, 1)` per tile. This is what
    // prevents greedy-merged quads from sampling the neighbouring atlas
    // tile on a 2D texture atlas.
    ivec2 tile_pos = ivec2(actual_tile % 16u, actual_tile / 16u);
    vec2 atlas_uv = (vec2(tile_pos) + vec2(fract(frag_uv.x), fract(frag_uv.y))) / ATLAS_TILES;

    vec4 tex = texture(atlas, atlas_uv);
    // Discard near-zero-alpha fragments so leaves/glass cutouts look right.
    if (tex.a < 0.1) {
        discard;
    }
    // frag_light > 1.0 signals water (encoded level). Clamp for lighting.
    float light = min(frag_light, 1.0);
    // Apply baked per-vertex light * dynamic ambient (day/night dimming).
    float ambient = fog.ambient_and_sun.x;

    float shadow_factor = 1.0;
    if (SHADOW_ENABLED) {
        float view_depth = length(cam.cam_pos_and_maxdist.xyz - frag_world_pos);
        shadow_factor = compute_shadow_factor(frag_world_pos, view_depth);
    }
    vec3 lit = tex.rgb * light * ambient * shadow_factor;
    // Apply saturated colored-light tint so torch/colored-block tints are
    // immediately recognisable on the surface of nearby blocks.
    vec3 emissive_contribution = lit * frag_light_color.rgb * (light * 0.18);
    vec3 tinted = lit * frag_light_color.rgb + emissive_contribution;
    vec3 final = mix(tinted, fog.color_and_density.rgb, frag_fog);

    // ── Per-tile material effects (binding 5) ────────────────────────────
    BlockMaterialGpu mat = material_table.materials[actual_tile];
    uint flags = mat.flags_roughness_emissive_pad & 0xFFu;
    vec3 sun_dir = fog.ambient_and_sun.yzw;
    float water_y         = material_table.world_params.x;
    float wet_edge_amount = material_table.world_params.y;
    float caustics_amount = material_table.world_params.z;
    float leaves_amount   = material_table.world_params.w;

    // ── Leaves subsurface backlight ────────────────────────────────────
    // Cheap "flat-face normal" via screen-space derivatives: the cross of
    // dFdx and dFdy on world_pos gives the unit normal of the triangle's
    // plane, which for axis-aligned voxel faces IS the cube-face outward
    // normal. `dot(-sun_dir, n) > 0` ⇒ sun is hitting the back of the face
    // (i.e. lighting the leaves from behind). Tint is the registry's
    // sss_tint (chlorophyll yellow by default).
    if ((flags & MATERIAL_FLAG_LEAVES_SSS) != 0u && leaves_amount > 0.0) {
        // View-direction based backlight proxy. Voxel leaf cubes face any
        // of 6 directions at any world position and a cross-product flat
        // normal from dFdx/dFdy is unreliable. When sun_dir roughly matches
        // camera→fragment the player is looking at the BACK side of the
        // leaf and the front face is sunlit from behind (chlorophyll effect).
        vec3 cam_to_frag = normalize(frag_world_pos - cam.cam_pos_and_maxdist.xyz);
        float backlight = clamp(max(dot(sun_dir, cam_to_frag), 0.0) * 1.4, 0.0, 1.0);
        vec3 sss_rgb = vec3(
            float(mat.sss_tint & 0xFFu),
            float((mat.sss_tint >> 8u) & 0xFFu),
            float((mat.sss_tint >> 16u) & 0xFFu)
        ) / 255.0;
        float sss_alpha = float((mat.sss_tint >> 24u) & 0xFFu) / 255.0;
        // Tint (hue-shift) toward chlorophyll, not just brighten — pure
        // multiplicative tint produces a visible hue shift when lit from
        // behind; additive terms wash the texture toward the tint color.
        float backlight_mix = leaves_amount * backlight * sss_alpha * (ambient * 0.85);
        final = mix(final, final * sss_rgb * 1.5, backlight_mix);
    }

    // ── Sun caustics on submerged terrain ──────────────────────────────
    // Cheap approximation: any block that is below the global water surface
    // Y (and isn't itself water so we don't add caustics inside the water
    // volume) gets a procedural sin/cos glare modulated by the sun.
    // Not neighbour-perfect but visually convincing and zero per-voxel data.
    bool is_water_block = (flags & MATERIAL_FLAG_WATER) != 0u;
    if (!is_water_block && caustics_amount > 0.0 && frag_world_pos.y < water_y) {
        float depth_below = water_y - frag_world_pos.y;
        // Caustics fade out deeper than 4 voxels below the surface so far
        // seafloor doesn't compete with the player's attention.
        float depth_attenuation = clamp(1.0 - depth_below / 4.0, 0.0, 1.0);
        float caustics_pattern = sin(frag_world_pos.x * 0.6) * cos(frag_world_pos.z * 0.6);
        float glare = max(caustics_pattern, 0.0) * depth_attenuation * caustics_amount * shadow_factor * ambient;
        final += glare * vec3(0.85, 0.92, 1.0) * 0.45;
    }

    // ── Reflections (slice 3): shared view vectors ───────────────────────
    // Computed once for every reflection-capable tile (water, glass,
    // opaque REFLECTIVE). `refl_master == 0` disables all reflection paths.
    float refl_master = refl.sun_dir_str.w;
    bool reflections_on = refl_master > 0.001;
    bool underwater_cam = refl.proj_misc.z > 0.5;
    vec3 cam_pos = cam.cam_pos_and_maxdist.xyz;
    vec3 view_vec = frag_world_pos - cam_pos;
    float view_dist = length(view_vec);
    vec3 view_dir = view_vec / max(view_dist, 1e-4);
    // Output alpha — water recomputes it (fresnel drives opacity); every
    // other path keeps the texture alpha.
    float out_alpha = tex.a;

    // Pre-compute scene size once (shared by water refraction and glass absorption).
    vec2 scene_size = vec2(textureSize(scene_opaque, 0));

    // ── WATER: refraction + fresnel reflection (SSR with sky fallback) ───
    // Replaces the base shading for water tiles. The refraction samples the
    // binding-6 opaque colour distorted by the procedural wave normal, with
    // Beer-Lambert absorption over the REAL water column depth (binding-7
    // scene depth minus this fragment's eye depth). The reflection is a
    // screen-space ray-march falling back to the analytic sky on miss,
    // weighted by Schlick fresnel. A specular sun glint rides on top.
    if (is_water_block && reflections_on && !underwater_cam) {
        float t = push.time_and_pad.x;
        vec3 n = water_wave_normal(frag_world_pos, t);
        vec3 r = reflect(view_dir, n);
        float cos_theta = clamp(dot(-view_dir, n), 0.0, 1.0);
        float fres = (0.02 + 0.98 * pow(1.0 - cos_theta, 5.0)) * refl_master;

        vec2 frag_uv_ss = gl_FragCoord.xy / scene_size;
        float frag_eye = (push.view_proj * vec4(frag_world_pos, 1.0)).w;

        // Refraction: wave-normal offset with a perspective-ish falloff so
        // distant water distorts less (screen-space magnitude stays sane).
        vec3 refracted;
        {
            vec2 refr_off = n.xz * (0.06 / max(frag_eye * 0.12, 0.35));
            vec2 refr_uv = clamp(frag_uv_ss + refr_off, vec2(0.001), vec2(0.999));
            // Sample `scene_depth` only when SSR / depth-resolve is genuinely
            // available on this device. Otherwise the depth copy is skipped
            // (`depth_resolve_mode = None`) and `scene_opaque_depth` is
            // unwritten-undefined, which would make water chunks sample
            // garbage `linearize_depth()` values and render as a solid-white
            // flash. Falling back to a far-plane sentinel (1.0) drives
            // `water_col` to the post-clamp 0.6 floor via the existing
            // `scene_eye > 0.95 * proj_misc.y` branch, making the chunk
            // render as an opaque sky-mirror instead of a white-out flash.
            bool ssr_valid_refraction = refl.proj_misc.w > 0.5;
            float scene_d_sample = ssr_valid_refraction ? textureLod(scene_depth, refr_uv, 0.0).r : 1.0;
            float scene_eye = linearize_depth(scene_d_sample);
            float water_col = max(scene_eye - frag_eye, 0.0);
            // Sky behind the surface (ray exits): clamp to a shallow column so
            // edge-on water against the sky doesn't go black-blue.
            if (scene_eye > 0.95 * refl.proj_misc.y) water_col = 0.6;
            water_col = min(water_col, 8.0);
            vec3 absorb_rgb = vec3(
                float(mat.absorption_pad & 0xFFu),
                float((mat.absorption_pad >> 8u) & 0xFFu),
                float((mat.absorption_pad >> 16u) & 0xFFu)
            ) / 255.0;
            vec3 beer = exp(-absorb_rgb * water_col * 2.5);
            refracted = textureLod(scene_opaque, refr_uv, 0.0).rgb * beer;
            // Slight body tint from the tile texture so the water keeps its
            // identity instead of becoming a perfect (boring) window.
            refracted = mix(refracted, refracted * (tex.rgb * 1.8 + 0.05), 0.30);
        }

        // Reflection: SSR where possible, analytic sky otherwise. SSR fades
        // out with fragment distance (hides artifacts, saves marching).
        // Checkerboard pattern: only run the expensive SSR on half the pixels
        // (odd/even screen coords), using the sky fallback for the other half.
        // This halves the fragment-shader cost of reflections with minimal
        // visual impact thanks to spatial blending.
        vec3 sky_col = sky_reflection_color(r);
        vec3 reflected = sky_col;
        bool checker = (int(gl_FragCoord.x + gl_FragCoord.y) & 1) == 0;
        if (refl.proj_misc.w > 0.5 && view_dist < 60.0 && checker) {
            vec4 hit = ssr_raymarch(frag_world_pos, r, view_dist);
            float conf = hit.a * (1.0 - smoothstep(24.0, 60.0, view_dist));
            reflected = mix(sky_col, hit.rgb, conf);
        }

        vec3 water_col = mix(refracted, reflected, fres);

        // Sun glint: tight specular lobe along the reflected ray, shadowed
        // and day/night scaled like everything else.
        vec3 sd = normalize(refl.sun_dir_str.xyz);
        if (sd.y > 0.0) {
            float glint = pow(max(dot(r, sd), 0.0), 400.0);
            water_col += vec3(1.0, 0.95, 0.8) * glint * 3.0 * shadow_factor * ambient;
        }

        final = mix(water_col, fog.color_and_density.rgb, frag_fog);
        // Fresnel makes grazing water a mirror (opaque); head-on it stays a
        // window (texture alpha).
        out_alpha = clamp(tex.a + fres * 1.2, 0.0, 1.0);
    }

    // ── TRANSLUCENT_ABSORB (slice 2): glass-tinted absorption ─────────
    // Sample the scene_opaque_color side-car at this fragment's screen UV.
    // Beer's-law: the glass/ice-alpha absorbs the scene color by the
    // absorption_pad.rgb coefficient packed per-tile; the more opaque the
    // glass, the less scene_color comes through. We modulate by ambient
    // (so dark nights show dark glass) and tint toward the tile absorption
    // RGB so the visual identity of glassy blocks reads as green-glass,
    // blue-ice, etc.
    if ((flags & MATERIAL_FLAG_TRANSLUCENT_ABSORB) != 0u) {
        vec2 scene_uv = gl_FragCoord.xy / scene_size;
        vec3 scene_rgb = texture(scene_opaque, scene_uv).rgb;
        // absorption_pad: low 24 bits = RGB coefficients (0..1, inverse
        // saturation). High 8 bits reserved for alpha tint in slice 3.
        vec3 absorption = vec3(
            float(mat.absorption_pad & 0xFFu),
            float((mat.absorption_pad >> 8u) & 0xFFu),
            float((mat.absorption_pad >> 16u) & 0xFFu)
        ) / 255.0;
        // Beer's law I = I0 * exp(-k * d). Without per-pixel depth for now
        // we use fragment alpha (already known to the chunk frag) as a
        // thickness proxy, modulated by the tile's absorption tint.
        float thickness = tex.a;
        vec3 transmitted = scene_rgb * exp(-absorption * (thickness * 4.0));
        // Premultiplied-alpha blend toward tint RGB. The chunk pipeline's
        // blend is SRC_ALPHA / ONE_MINUS_SRC_ALPHA so the alpha channel
        // already controls how much scene we replace.
        final = mix(scene_rgb, transmitted, smoothstep(0.0, 1.0, thickness));
        final *= (ambient * 0.95 + 0.05);

        // Glass reflection (slice 3): a subtle fresnel sheen so glass reads
        // as glass. The flat voxel-face normal comes from screen-space
        // derivatives; SSR where available, analytic sky otherwise.
        if (reflections_on && !underwater_cam) {
            vec3 n = face_normal(view_dir);
            vec3 r = reflect(view_dir, n);
            float cos_theta = clamp(dot(-view_dir, n), 0.0, 1.0);
            float fres = (0.04 + 0.96 * pow(1.0 - cos_theta, 5.0)) * refl_master;
            vec3 sky_col = sky_reflection_color(r);
            vec3 refl_col = sky_col;
            if (refl.proj_misc.w > 0.5 && view_dist < 60.0) {
                vec4 hit = ssr_raymarch(frag_world_pos + n * 0.05, r, view_dist);
                float conf = hit.a * (1.0 - smoothstep(20.0, 45.0, view_dist));
                refl_col = mix(sky_col, hit.rgb, conf);
            }
            final += refl_col * fres * 0.35 * ambient;
        }
    }

    // ── REFLECTIVE (slice 3): opaque glossy tiles (sky reflection + glint) ──
    // Opaque blocks can't SSR (they render in the main pass, BEFORE the
    // scene copies are refreshed), so they get the analytic sky reflection +
    // a sun glint, scaled by (1 - roughness). Registered per-tile (e.g. snow
    // sheen); zero cost when the flag is unset.
    if ((flags & MATERIAL_FLAG_REFLECTIVE) != 0u && reflections_on && !underwater_cam) {
        float rough = float((mat.flags_roughness_emissive_pad >> 8u) & 0xFFu) / 255.0;
        float gloss = (1.0 - rough) * refl_master;
        if (gloss > 0.01) {
            vec3 n = face_normal(view_dir);
            vec3 r = reflect(view_dir, n);
            float cos_theta = clamp(dot(-view_dir, n), 0.0, 1.0);
            float fres = 0.04 + 0.96 * pow(1.0 - cos_theta, 5.0);
            vec3 sky_col = sky_reflection_color(r);
            final = mix(final, sky_col * (ambient * 0.9 + 0.1), fres * gloss * 0.6);
            vec3 sd = normalize(refl.sun_dir_str.xyz);
            if (sd.y > 0.0) {
                float glint = pow(max(dot(r, sd), 0.0), 64.0);
                final += vec3(1.0, 0.95, 0.8) * glint * gloss * 0.8 * shadow_factor * ambient;
            }
        }
    }

    // ── Wet-edge tint ──────────────────────────────────────────────────
    // Cheap approximation: a band of voxels immediately below the water
    // surface is "wet". Ideal neighbour-accurate detection would sample the
    // block horizontally adjacent to this one, but the chunk shader has no
    // neighbour data — so we treat the water surface band itself as wet. In
    // practice this puts a soft cool tint on top of land blocks whose top
    // face sits at or just under the water Y (sand banks, dirt shores), which
    // is exactly where visible wet edges appear.
    if (wet_edge_amount > 0.0 && !is_water_block) {
        float band_dist = abs(frag_world_pos.y - (water_y - 0.05));
        float wetness = clamp(1.0 - band_dist / 0.55, 0.0, 1.0);
        vec3 wet_rgb = vec3(
            float(mat.wet_tint & 0xFFu),
            float((mat.wet_tint >> 8u) & 0xFFu),
            float((mat.wet_tint >> 16u) & 0xFFu)
        ) / 255.0;
        // Fall back to a cool dampened blue if the tile didn't specify one.
        if (wet_rgb == vec3(0.0)) wet_rgb = vec3(0.85, 0.92, 1.0);
        // Modulate strictly by ambient (no +0.1 floor) so wet-edge tint
        // tapers to zero at night, matching the day-night scaling the rest
        // of the lighting already uses.
        final += wet_rgb * wetness * wet_edge_amount * 0.18 * ambient;
    }

    out_color = vec4(final, out_alpha);
}
