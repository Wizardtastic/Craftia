#version 450

#extension GL_EXT_nonuniform_qualifier : enable

// Chunk fragment shader - fully dynamic, GPU ray-traced lighting.
//
// Nothing is baked into the mesh. Every pixel reconstructs its world position
// and axis-aligned voxel normal, then traces rays through a GPU 3D voxel grid
// (descriptor binding 3, R8_UINT block IDs) to evaluate:
//
//   * sun and moon soft shadows (cone-jittered DDA rays)
//   * ray-traced ambient occlusion + sky diffuse GI (hemisphere rays)
//   * one-bounce emissive block lighting (torch/lava/firefly gather rays)
//   * GGX-ish specular reflections for glossy surfaces
//   * water reflections + ray-traced water-column refraction absorption
//   * sun caustics under water, leaves subsurface backlight
//   * single-scattering in-scatter along the view ray (light shafts)
//
// The sun/moon directions, radiance and sky palette come from the engine's
// astronomical model (time of day + latitude/longitude + day of year).

const float ATLAS_TILES = 16.0;
const float PI = 3.14159265359;

layout(constant_id = 0) const bool TILE_REMAP_ENABLED = false;

// ── Atlas + tile materials (bindings 1 and 5) ─────────────────────────────
struct BlockMaterialGpu {
    uint flags_roughness_emissive_pad;
    uint sss_tint;
    uint wet_tint;
    uint absorption_pad;
};
layout(set = 0, binding = 5) uniform MaterialTable {
    BlockMaterialGpu materials[256];
    vec4 world_params;
} material_table;

const uint MATERIAL_FLAG_LEAVES_SSS        = 1u << 0u;
const uint MATERIAL_FLAG_TRANSLUCENT_ABSORB = 1u << 1u;
const uint MATERIAL_FLAG_WATER              = 1u << 2u;
const uint MATERIAL_FLAG_REFLECTIVE         = 1u << 3u;

// ── Ray-tracing uniforms (binding 4) ──────────────────────────────────────
layout(set = 0, binding = 4) uniform Raytrace {
    vec4 sun_dir_vis;       // xyz = direction to sun, w = visibility 0..1
    vec4 sun_color_radius;  // rgb = sun radiance, a = angular radius (rad)
    vec4 moon_dir_phase;    // xyz = direction to moon, w = illuminated fraction
    vec4 moon_color_radius; // rgb = moon radiance, a = angular radius (rad)
    vec4 sky_zenith;        // rgb = zenith radiance, a = night factor
    vec4 sky_horizon;       // rgb = horizon radiance, a = time (seconds)
    vec4 fog_density;       // rgb = fog colour, a = density scale
    vec4 grid_origin;       // xyz = voxel grid world origin
    vec4 grid_extent;       // xyz = grid extent in blocks, w = quality 0..1
    vec4 params;            // x = reflection strength, y = shadow softness, z = AO strength, w = time
} rt;

// ── Per-block light data (binding 9) ──────────────────────────────────────
// Indexed by the block ID stored in the voxel grid.
//   kind: 0 air, 1 opaque, 2 water, 3 translucent, 4 foliage
//   emission_level: 0..15
//   emission_rgb: packed RGB8
//   roughness: 0 = mirror, 255 = matte
struct BlockLightGpu {
    uint kind;
    uint emission_level;
    uint emission_rgb;
    uint roughness;
};
layout(set = 0, binding = 9) uniform BlockLights {
    BlockLightGpu blocks[256];
} block_lights;

const uint BLOCK_KIND_AIR        = 0u;
const uint BLOCK_KIND_OPAQUE     = 1u;
const uint BLOCK_KIND_WATER      = 2u;
const uint BLOCK_KIND_TRANSLUCENT= 3u;
const uint BLOCK_KIND_FOLIAGE    = 4u;

// ── Voxel grid (binding 3) ────────────────────────────────────────────────
layout(set = 0, binding = 3) uniform usampler3D voxel_grid;

// ── Scene copies for water refraction / glass absorption ──────────────────
layout(set = 0, binding = 6) uniform sampler2D scene_opaque;
layout(set = 0, binding = 7) uniform sampler2D scene_depth;

layout(set = 0, binding = 8) uniform Reflection {
    vec4 sky_horizon;   // rgb = horizon radiance
    vec4 sky_zenith;    // rgb = zenith radiance
    vec4 sun_dir_str;   // xyz = sun direction, w = reflection strength
    vec4 proj_misc;     // x = near, y = far, z = underwater, w = ssr valid
    mat4 view_proj;     // current frame's view-projection (SSR reprojection)
} refl;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in float frag_fog;
layout(location = 2) in vec3 frag_world_pos;
layout(location = 3) flat in uint frag_tile;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist;
} cam;

layout(set = 0, binding = 1) uniform sampler2D atlas;

layout(set = 0, binding = 2) uniform Fog {
    vec4 color_and_density;
    vec4 sun_color;
    vec4 sky_ambient;
    vec4 sun_dir_pad;
} fog;

// Animated tile remap (identity unless animated textures are enabled).
layout(set = 1, binding = 0) uniform TileRemap {
    uint map[256];
} tile_remap;

// ── Hash / noise ──────────────────────────────────────────────────────────
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 hash32(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

// ── Voxel grid access ─────────────────────────────────────────────────────
bool grid_contains(vec3 p) {
    vec3 rel = p - rt.grid_origin.xyz;
    return all(greaterThanEqual(rel, vec3(0.0))) &&
           all(lessThan(rel, rt.grid_extent.xyz));
}

uint voxel_id(ivec3 c) {
    ivec3 o = ivec3(round(rt.grid_origin.xyz));
    ivec3 extent = ivec3(round(rt.grid_extent.xyz));
    ivec3 rel = c - o;
    if (any(lessThan(rel, ivec3(0))) || any(greaterThanEqual(rel, extent))) {
        return 0u;
    }
    return texelFetch(voxel_grid, rel, 0).r;
}

BlockLightGpu block_light(uint id) {
    return block_lights.blocks[id & 255u];
}

// ── Analytic sky (mirrors sky.frag so reflected/ambient rays agree) ───────
vec3 sky_radiance(vec3 dir) {
    dir = normalize(dir);
    vec3 sun_dir = normalize(rt.sun_dir_vis.xyz);
    vec3 moon_dir = normalize(rt.moon_dir_phase.xyz);
    float sun_up = sun_dir.y;
    float day = smoothstep(-0.10, 0.22, sun_up);
    float twilight = exp(-pow((sun_up - 0.02) / 0.16, 2.0));

    float air_mass = 1.0 / (max(sun_up, -0.16) + 0.18);
    vec3 extinction = exp(-air_mass * vec3(0.35, 0.95, 2.40) * 0.55);
    vec3 sun_tint = mix(vec3(1.0), extinction, 0.85);

    float up = clamp(dir.y, 0.0, 1.0);
    float grad = pow(up, 0.42);
    vec3 col = mix(rt.sky_horizon.rgb, rt.sky_zenith.rgb, grad);

    vec2 sun_az = sun_dir.xz;
    float az_len = length(sun_az);
    sun_az = az_len > 1e-4 ? sun_az / az_len : vec2(1.0, 0.0);
    vec2 dir_az = dir.xz;
    float d_az_len = length(dir_az);
    dir_az = d_az_len > 1e-4 ? dir_az / d_az_len : sun_az;
    float sunward = clamp(dot(dir_az, sun_az) * 0.5 + 0.5, 0.0, 1.0);

    float mu = max(dot(dir, sun_dir), 0.0);
    float mie = pow(mu, 8.0) * 0.28 + pow(mu, 96.0) * 0.42 + pow(mu, 1200.0) * 2.6;
    col += sun_tint * mie * rt.sun_dir_vis.w * vec3(1.0, 0.94, 0.84);

    col += sun_tint * twilight * pow(1.0 - up, 3.0) * (0.25 + 0.75 * sunward)
         * 1.15 * rt.sun_dir_vis.w;
    col += sun_tint * twilight * pow(up, 0.35) * sunward * 0.12 * rt.sun_dir_vis.w;

    // Sun disc.
    float disc = smoothstep(0.99980, 0.99990, dot(dir, sun_dir));
    col += sun_tint * disc * 9.0 * smoothstep(-0.03, 0.02, sun_up);

    // Moon disc + halo, scaled by the illuminated fraction.
    if (moon_dir.y > -0.05) {
        float moon_mu = max(dot(dir, moon_dir), 0.0);
        float moon_disc = smoothstep(0.99972, 0.99988, moon_mu);
        float maria = hash12(floor(dir.xz * 180.0));
        float moon_vis = smoothstep(-0.04, 0.06, moon_dir.y);
        float phase = clamp(rt.moon_dir_phase.w, 0.0, 1.0);
        col += vec3(0.86, 0.90, 1.05) * moon_disc * (0.75 + 0.35 * maria)
             * (1.2 + 2.0 * phase) * moon_vis;
        col += vec3(0.86, 0.90, 1.05) * pow(moon_mu, 320.0) * 0.10 * moon_vis;
    }

    // Night sky floor, so shadowed surfaces never crush to pure black.
    col += vec3(0.004, 0.006, 0.016) * rt.sky_zenith.a * (1.0 - day);
    return max(col, vec3(0.0));
}

// ── Voxel ray trace ───────────────────────────────────────────────────────
struct TraceHit {
    float transmittance; // fraction of light that survives the march
    vec3 emission;       // emitted radiance gathered along the ray
    vec3 position;       // first blocking hit position (if hit)
    uint block;          // first blocking block ID (0 = miss)
    float distance;      // distance to the blocking hit
    bool hit;
    bool water;          // the blocking hit is water
};

// Amanatides & Woo DDA through the block grid. `collect_emission` makes the
// march gather torch/lava/firefly radiance from every emissive voxel crossed.
TraceHit trace_voxels(vec3 origin, vec3 dir, float max_dist, int max_steps,
                      bool collect_emission) {
    TraceHit r;
    r.transmittance = 1.0;
    r.emission = vec3(0.0);
    r.position = origin + dir * max_dist;
    r.block = 0u;
    r.distance = max_dist;
    r.hit = false;
    r.water = false;

    vec3 d = normalize(dir);
    ivec3 cell = ivec3(floor(origin));
    ivec3 stp = ivec3(sign(d));
    vec3 inv = 1.0 / max(abs(d), vec3(1e-6));

    vec3 tmax;
    tmax.x = (d.x >= 0.0) ? (float(cell.x + 1) - origin.x) * inv.x
                          : (origin.x - float(cell.x)) * inv.x;
    tmax.y = (d.y >= 0.0) ? (float(cell.y + 1) - origin.y) * inv.y
                          : (origin.y - float(cell.y)) * inv.y;
    tmax.z = (d.z >= 0.0) ? (float(cell.z + 1) - origin.z) * inv.z
                          : (origin.z - float(cell.z)) * inv.z;
    vec3 tdelta = inv;

    float t = 0.0;
    for (int i = 0; i < max_steps; i++) {
        uint id = voxel_id(cell);
        if (id != 0u) {
            BlockLightGpu m = block_light(id);
            if (m.emission_level > 0u && collect_emission) {
                vec3 e = vec3(float(m.emission_rgb & 255u),
                              float((m.emission_rgb >> 8u) & 255u),
                              float((m.emission_rgb >> 16u) & 255u)) / 255.0;
                float level = float(m.emission_level) / 15.0;
                // Inverse-square-ish falloff, clamped for stability up close.
                float fade = 1.0 / (1.0 + 0.22 * t * t);
                r.emission += e * level * 10.0 * fade * r.transmittance;
            }
            if (m.kind == BLOCK_KIND_OPAQUE) {
                r.hit = true;
                r.block = id;
                r.distance = t;
                r.position = origin + d * t;
                break;
            } else if (m.kind == BLOCK_KIND_WATER) {
                // Beer absorption through every water cell crossed.
                r.transmittance *= exp(-0.35);
                r.water = true;
            } else if (m.kind == BLOCK_KIND_FOLIAGE) {
                // Leaves dapple sunlight instead of blocking it entirely.
                r.transmittance *= 0.62;
            } else if (m.kind == BLOCK_KIND_TRANSLUCENT) {
                r.transmittance *= 0.86;
            }
        }

        if (tmax.x < tmax.y && tmax.x < tmax.z) {
            t = tmax.x;
            tmax.x += tdelta.x;
            cell.x += stp.x;
        } else if (tmax.y < tmax.z) {
            t = tmax.y;
            tmax.y += tdelta.y;
            cell.y += stp.y;
        } else {
            t = tmax.z;
            tmax.z += tdelta.z;
            cell.z += stp.z;
        }
        if (t > max_dist) break;
    }
    return r;
}

// Shadow ray toward `light_dir`. Multiple jittered rays inside the light's
// angular disc produce a real, distance-stable penumbra.
float soft_visibility(vec3 p, vec3 n, vec3 light_dir, float angular_radius,
                      float jitter_seed) {
    if (light_dir.y <= 0.0) return 0.0;
    vec3 helper = abs(light_dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t1 = normalize(cross(light_dir, helper));
    vec3 t2 = cross(light_dir, t1);
    float a = jitter_seed * 6.2831853;
    vec3 offset = t1 * cos(a) + t2 * sin(a);
    // Two samples per pixel plus a third for high quality settings.
    vec3 o1 = offset * angular_radius;
    vec3 o2 = (t1 * cos(a + 2.0944) + t2 * sin(a + 2.0944)) * angular_radius * 1.6;
    vec3 o3 = (t1 * cos(a + 4.1888) + t2 * sin(a + 4.1888)) * angular_radius * 1.6;
    vec3 origin = p + n * 0.02;
    float budget = 72.0;
    float sum = trace_voxels(origin, light_dir + o1, budget, 96, false).transmittance;
    sum += trace_voxels(origin, light_dir + o2, budget, 96, false).transmittance;
    if (rt.grid_extent.w > 0.66) {
        sum += trace_voxels(origin, light_dir + o3, budget, 96, false).transmittance;
        sum /= 3.0;
    } else {
        sum *= 0.5;
    }
    return sum;
}

// Cosine-weighted hemisphere ambient: 3 traced rays weighted by the analytic
// sky radiance they see. This is ray-traced sky occlusion + diffusely
// reflected skylight in one term.
vec3 sky_ambient_raytraced(vec3 p, vec3 n, float seed) {
    vec3 up = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);
    float a = seed * 6.2831853;
    vec3 dirs[3];
    dirs[0] = normalize(n + t * cos(a) * 0.75 + b * sin(a) * 0.75);
    dirs[1] = normalize(n + t * cos(a + 2.0944) * 0.85 + b * sin(a + 2.0944) * 0.85);
    dirs[2] = normalize(n + t * cos(a + 4.1888) * 0.45 + b * sin(a + 4.1888) * 0.45);

    vec3 sum = vec3(0.0);
    int samples = rt.grid_extent.w > 0.5 ? 3 : 2;
    for (int i = 0; i < 3; i++) {
        if (i >= samples) break;
        vec3 d = dirs[i];
        float ndl = max(dot(n, d), 0.0);
        if (ndl <= 0.0) continue;
        TraceHit hit = trace_voxels(p + n * 0.02, d, 12.0, 20, true);
        vec3 radiance = sky_radiance(d) * hit.transmittance;
        // A hit surface contributes one diffuse bounce (approximated by its
        // emissive term and a fraction of the sky it would see) instead of
        // going black: this is cheap one-bounce indirect light.
        if (hit.hit) {
            radiance += hit.emission * 0.35
                      + sky_radiance(n) * 0.06 * hit.transmittance;
        }
        sum += radiance * ndl;
    }
    return sum / float(max(samples, 1));
}

// One-bounce specular reflection. Hits are resolved against the voxel grid;
// visible hit points are coloured from the scene copy so reflected terrain
// keeps its albedo, while off-screen hits fall back to the sky.
vec3 trace_reflection(vec3 p, vec3 n, vec3 v, float roughness, float seed) {
    vec3 r = reflect(-v, n);
    // Microfacet cone: wider with roughness.
    vec3 up = abs(r.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, r));
    vec3 b = cross(r, t);
    float a = seed * 6.2831853;
    float spread = roughness * roughness * 0.55;
    vec3 dir = normalize(r + (t * cos(a) + b * sin(a)) * spread);

    vec3 fallback = sky_radiance(dir);
    TraceHit hit = trace_voxels(p + n * 0.03, dir, 48.0, 64, true);
    if (!hit.hit) {
        return fallback + hit.emission;
    }
    vec3 colour = fallback * 0.55;
    // Project the traced hit into the scene copy: if that surface is on
    // screen we can reuse its shaded colour (a ray-traced reflection with
    // real terrain albedo, no screen-space search).
    vec4 clip = refl.view_proj * vec4(hit.position, 1.0);
    if (clip.w > 0.0) {
        vec2 uv = vec2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
            colour = textureLod(scene_opaque, uv, 0.0).rgb;
        }
    }
    return colour + hit.emission;
}

// ── Surface helpers ───────────────────────────────────────────────────────
vec3 face_normal(vec3 view_dir) {
    vec3 n = normalize(cross(dFdx(frag_world_pos), dFdy(frag_world_pos)));
    return (dot(n, view_dir) < 0.0) ? -n : n;
}

vec3 water_wave_normal(vec3 wp, float t) {
    float a1 = 0.04 * 2.5;
    float p1 = wp.x * 1.5 + t * 1.8;
    float q1 = wp.z * 1.2 + t * 1.4;
    float dhdx = a1 * 1.5 * cos(p1) * cos(q1);
    float dhdz = -a1 * 1.2 * sin(p1) * sin(q1);
    float a2 = 0.022;
    float p2 = wp.x * 3.9 + t * 2.7;
    float q2 = wp.z * 3.3 + t * 2.2;
    dhdx += a2 * 3.9 * cos(p2) * cos(q2);
    dhdz += -a2 * 3.3 * sin(p2) * sin(q2);
    return normalize(vec3(-dhdx, 1.0, -dhdz));
}

float fresnel_schlick(float cos_theta, float f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// Ray-traced water column depth: march straight down until an opaque voxel.
// This replaces any depth-buffer guessing with the real geometry beneath the
// surface, and gives Beer absorption the true optical path.
float water_column_depth(vec3 surface) {
    TraceHit hit = trace_voxels(surface + vec3(0.0, -0.05, 0.0),
                                vec3(0.0, -1.0, 0.0), 24.0, 32, false);
    return hit.hit ? clamp(hit.distance, 0.2, 16.0) : 6.0;
}

void main() {
    uint actual_tile = frag_tile;
    if (TILE_REMAP_ENABLED) {
        actual_tile = tile_remap.map[frag_tile];
    }

    ivec2 tile_pos = ivec2(actual_tile % 16u, actual_tile / 16u);
    vec2 atlas_uv = (vec2(tile_pos) + vec2(fract(frag_uv.x), fract(frag_uv.y))) / ATLAS_TILES;

    vec4 tex = texture(atlas, atlas_uv);
    if (tex.a < 0.1) {
        discard;
    }

    // Atlas textures are authored in sRGB but sampled through a UNORM view.
    vec3 albedo = pow(tex.rgb, vec3(2.2));

    BlockMaterialGpu mat = material_table.materials[actual_tile];
    uint flags = mat.flags_roughness_emissive_pad & 0xFFu;
    float roughness = float((mat.flags_roughness_emissive_pad >> 8u) & 0xFFu) / 255.0;
    float emissive_amount = float((mat.flags_roughness_emissive_pad >> 16u) & 0xFFu) / 255.0;
    float water_y = material_table.world_params.x;
    float wet_edge_amount = material_table.world_params.y;
    float caustics_amount = material_table.world_params.z;
    float leaves_amount = material_table.world_params.w;

    vec3 cam_pos = cam.cam_pos_and_maxdist.xyz;
    vec3 to_eye = cam_pos - frag_world_pos;
    float view_dist = length(to_eye);
    vec3 v = to_eye / max(view_dist, 1e-4);
    vec3 n = face_normal(v);

    vec3 sun_dir = normalize(rt.sun_dir_vis.xyz);
    vec3 moon_dir = normalize(rt.moon_dir_phase.xyz);
    float sun_vis = clamp(rt.sun_dir_vis.w, 0.0, 1.0);
    float sun_ndl = max(dot(n, sun_dir), 0.0);
    float moon_ndl = max(dot(n, moon_dir), 0.0);

    float seed = hash12(gl_FragCoord.xy + fract(rt.params.w) * 17.0);
    float shadow_softness = max(rt.params.y, 0.0008);
    float sun_ang = max(rt.sun_color_radius.a, shadow_softness);

    // ── Ray-traced direct sun + moon ─────────────────────────────────────
    float sun_shadow = 1.0;
    if (sun_vis > 0.001 && sun_ndl > 0.0) {
        sun_shadow = soft_visibility(frag_world_pos, n, sun_dir, sun_ang, seed);
    }
    float moon_shadow = 1.0;
    if (moon_ndl > 0.0) {
        moon_shadow = soft_visibility(frag_world_pos, n, moon_dir,
                                      max(rt.moon_color_radius.a, shadow_softness),
                                      fract(seed * 7.31));
    }
    vec3 direct = rt.sun_color_radius.rgb * sun_ndl * sun_vis * sun_shadow;
    direct += rt.moon_color_radius.rgb * moon_ndl * moon_shadow;

    // ── Ray-traced ambient occlusion + sky GI ────────────────────────────
    vec3 ambient = sky_ambient_raytraced(frag_world_pos, n, seed);
    ambient *= rt.params.z;

    // ── Ray-traced emissive block light ──────────────────────────────────
    vec3 emissive_light = vec3(0.0);
    {
        // Two short gather rays biased toward grazing directions catch
        // torches, lava and glowing blocks with real occlusion.
        vec3 up = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 t = normalize(cross(up, n));
        vec3 b = cross(n, t);
        float a = seed * 6.2831853;
        vec3 d1 = normalize(n * 0.55 + t * cos(a) + b * sin(a));
        vec3 d2 = normalize(n * 0.55 + t * cos(a + PI) + b * sin(a + PI));
        TraceHit h1 = trace_voxels(frag_world_pos + n * 0.02, d1, 28.0, 48, true);
        TraceHit h2 = trace_voxels(frag_world_pos + n * 0.02, d2, 28.0, 48, true);
        emissive_light = (h1.emission + h2.emission) * 0.5;
    }

    vec3 lighting = direct + ambient + emissive_light;

    // ── Specular reflection (ray traced) ─────────────────────────────────
    float gloss = (1.0 - roughness) * clamp(rt.params.x, 0.0, 1.0);
    bool is_water_block = (flags & MATERIAL_FLAG_WATER) != 0u;
    if (!is_water_block && gloss > 0.02) {
        float cos_theta = clamp(dot(v, n), 0.0, 1.0);
        float f0 = (flags & MATERIAL_FLAG_REFLECTIVE) != 0u ? 0.045 : 0.02;
        float fres = fresnel_schlick(cos_theta, f0);
        vec3 refl = trace_reflection(frag_world_pos, n, v,
                                     roughness, fract(seed * 3.17));
        lighting += refl * fres * gloss;
        // Tight sun/moon glints on top of the traced reflection.
        vec3 r = reflect(-v, n);
        if (sun_vis > 0.0) {
            float spec = pow(max(dot(r, sun_dir), 0.0),
                             8.0 + 200.0 * (1.0 - roughness));
            lighting += rt.sun_color_radius.rgb * spec * gloss * sun_shadow * fres;
        }
        float mspec = pow(max(dot(r, moon_dir), 0.0), 8.0 + 200.0 * (1.0 - roughness));
        lighting += rt.moon_color_radius.rgb * mspec * gloss * moon_shadow * fres;
    }

    vec3 tinted = albedo * lighting;
    tinted += albedo * emissive_amount * 8.0;

    // ── Water: ray-traced reflection + refraction absorption ─────────────
    float out_alpha = tex.a;
    if (is_water_block && refl.proj_misc.z < 0.5) {
        float t = rt.sky_horizon.w;
        vec3 wn = water_wave_normal(frag_world_pos, t);
        float cos_theta = clamp(dot(v, wn), 0.0, 1.0);
        float fres = fresnel_schlick(cos_theta, 0.02) * clamp(rt.params.x, 0.0, 1.0);

        // Reflection: traced through the voxel world, falling back to sky.
        vec3 reflected = trace_reflection(frag_world_pos, wn, v, 0.08, seed);
        // Sun/moon glitter path.
        vec3 wr = reflect(-v, wn);
        if (sun_vis > 0.0) {
            reflected += rt.sun_color_radius.rgb
                       * pow(max(dot(wr, sun_dir), 0.0), 500.0) * 3.0 * sun_shadow;
        }

        // Refraction: reuse the visible opaque scene, but derive the water
        // column from a ray-traced march through the actual lake bed.
        vec2 scene_uv = gl_FragCoord.xy / vec2(textureSize(scene_opaque, 0));
        float column = water_column_depth(frag_world_pos);
        vec3 absorption = vec3(
            float(mat.absorption_pad & 0xFFu),
            float((mat.absorption_pad >> 8u) & 0xFFu),
            float((mat.absorption_pad >> 16u) & 0xFFu)) / 255.0;
        if (absorption == vec3(0.0)) absorption = vec3(0.42, 0.12, 0.07);
        vec3 beer = exp(-absorption * column * 2.4);
        vec3 refracted = textureLod(scene_opaque, clamp(scene_uv, vec2(0.001), vec2(0.999)), 0.0).rgb * beer;
        refracted = mix(refracted, refracted * (albedo * 2.6 + 0.03), 0.30);
        // Water body receives the same traced lighting (sparkle + depth).
        refracted += albedo * (ambient + direct) * 0.25;

        vec3 water_col = mix(refracted, reflected, clamp(fres + 0.10, 0.0, 1.0));
        tinted = water_col;
        out_alpha = clamp(tex.a + fres * 1.2, 0.0, 1.0);
    }

    // ── Glass / translucent absorption ───────────────────────────────────
    if ((flags & MATERIAL_FLAG_TRANSLUCENT_ABSORB) != 0u && !is_water_block) {
        vec2 scene_uv = gl_FragCoord.xy / vec2(textureSize(scene_opaque, 0));
        vec3 scene_rgb = texture(scene_opaque, scene_uv).rgb;
        vec3 absorption = vec3(
            float(mat.absorption_pad & 0xFFu),
            float((mat.absorption_pad >> 8u) & 0xFFu),
            float((mat.absorption_pad >> 16u) & 0xFFu)) / 255.0;
        float thickness = tex.a;
        vec3 transmitted = scene_rgb * exp(-absorption * (thickness * 4.0));
        tinted = mix(scene_rgb, transmitted, smoothstep(0.0, 1.0, thickness));
        tinted *= dot(lighting, vec3(0.2126, 0.7152, 0.0722)) * 0.95 + 0.05;
    }

    // ── Leaves subsurface backlight ──────────────────────────────────────
    if ((flags & MATERIAL_FLAG_LEAVES_SSS) != 0u && leaves_amount > 0.0) {
        vec3 cam_to_frag = normalize(frag_world_pos - cam_pos);
        float back = clamp(max(dot(sun_dir, cam_to_frag), 0.0) * 1.4, 0.0, 1.0);
        vec3 sss_rgb = vec3(
            float(mat.sss_tint & 0xFFu),
            float((mat.sss_tint >> 8u) & 0xFFu),
            float((mat.sss_tint >> 16u) & 0xFFu)) / 255.0;
        float sss_alpha = float((mat.sss_tint >> 24u) & 0xFFu) / 255.0;
        float amount = leaves_amount * back * sss_alpha * sun_shadow * sun_vis;
        tinted = mix(tinted, tinted * sss_rgb * 1.5 + rt.sun_color_radius.rgb * 0.08 * sss_rgb,
                     clamp(amount, 0.0, 1.0));
    }

    // ── Sun caustics on submerged terrain ────────────────────────────────
    if (!is_water_block && caustics_amount > 0.0 && frag_world_pos.y < water_y) {
        float depth_below = water_y - frag_world_pos.y;
        float depth_attenuation = clamp(1.0 - depth_below / 4.0, 0.0, 1.0);
        float pattern = sin(frag_world_pos.x * 0.6 + rt.params.w * 1.3)
                      * cos(frag_world_pos.z * 0.6 - rt.params.w * 0.9);
        float glare = max(pattern, 0.0) * depth_attenuation * caustics_amount
                    * sun_shadow * sun_vis;
        tinted += glare * rt.sun_color_radius.rgb * 0.10;
    }

    // ── Wet edge ─────────────────────────────────────────────────────────
    if (wet_edge_amount > 0.0 && !is_water_block) {
        float band_dist = abs(frag_world_pos.y - (water_y - 0.05));
        float wetness = clamp(1.0 - band_dist / 0.55, 0.0, 1.0);
        vec3 wet_rgb = vec3(
            float(mat.wet_tint & 0xFFu),
            float((mat.wet_tint >> 8u) & 0xFFu),
            float((mat.wet_tint >> 16u) & 0xFFu)) / 255.0;
        if (wet_rgb == vec3(0.0)) wet_rgb = vec3(0.85, 0.92, 1.0);
        tinted += wet_rgb * wetness * wet_edge_amount * 0.10
                * dot(lighting, vec3(0.2126, 0.7152, 0.0722));
    }

    // ── Aerial perspective + single-scattered light shafts ───────────────
    // One shadow ray to a midpoint along the view ray gives real volumetric
    // in-scatter through canopies and cave mouths, instead of a flat fog mix.
    float fog_t = frag_fog;
    vec3 fog_col = rt.fog_density.rgb;
    if (fog_t > 0.001) {
        vec3 mid = mix(frag_world_pos, cam_pos, 0.5);
        float vis = 0.0;
        if (sun_vis > 0.0) {
            vis = soft_visibility(mid, -sun_dir, sun_dir, sun_ang * 3.0,
                                  fract(seed * 5.13));
        }
        vec3 scatter = rt.sun_color_radius.rgb * vis * 0.55
                     + rt.moon_color_radius.rgb * 0.25;
        vec3 haze = fog_col + scatter;
        tinted = mix(tinted, haze, clamp(fog_t * min(rt.fog_density.a, 1.0), 0.0, 1.0));
    }

    out_color = vec4(max(tinted, vec3(0.0)), out_alpha);
}
