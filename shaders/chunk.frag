#version 450

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

// Chunk fragment shader - fully dynamic, GPU ray-traced lighting.
//
// Nothing is baked into the mesh. Every pixel reconstructs its world position
// and axis-aligned voxel normal, then traces rays through a GPU 3D voxel grid
// (descriptor binding 3, R8_UINT block IDs) to evaluate:
//
//   * sun and moon soft shadows (cone-jittered DDA rays)
//   * ray-traced ambient occlusion + sky diffuse GI (hemisphere rays)
//   * two-bounce diffuse GI with per-block albedo colour bleed
//   * soft, occluded point lights for torches/lava/fireflies (diffuse through
//     glass/water/leaves, fully blocked by opaque geometry)
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
    uint albedo_rgb;
    uint pad0;
    uint pad1;
    uint pad2;
};
layout(set = 0, binding = 9) uniform BlockLights {
    BlockLightGpu blocks[256];
} block_lights;

// ── Dynamic emissive block lights (binding 10) ───────────────────────────
// CPU-gathered torch/lava/glowstone positions near the camera. Direct point
// lights make light spread smoothly instead of relying on a gather ray
// randomly crossing an emissive voxel.
layout(set = 0, binding = 10) uniform EmissiveLights {
    vec4 light_pos_range[32];      // xyz = world position, w = cutoff range
    vec4 light_color_intensity[32];// rgb = linear colour, a = intensity
    vec4 light_meta;               // x = active count
} lights;

// ── Local water-body grid (binding 11) ───────────────────────────────────
// Header = (origin.x, origin.z, size, valid). Row-major cells, each with two
// (surface, bed) layers ordered highest surface first:
//   surface > 0 : real water body; caustics apply above bed and below surface
//   surface < 0 : dry land within two blocks of water (wet-edge band only;
//                 bed stores the fringe height)
//   -1000       : no nearby water in this layer
// Keeping the bed, and a second layer, lets a cave aquifer under a lake or
// ocean stay separate from the surface body instead of inheriting its depth.
const float WATER_GRID_NONE = -1000.0;
const int WATER_GRID_LAYERS = 2;
layout(set = 0, binding = 11) readonly buffer WaterSurfaceGrid {
    vec4 header;
    vec2 bodies[];
} water_grid;

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
layout(location = 4) flat in uint frag_water;
layout(location = 5) in float frag_water_amp;
layout(location = 6) flat in float frag_water_time;

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

struct LocalWater {
    float caustic_surface;
    float wet_surface;
    bool actual_water;
    bool wet;
};

/// Pick the nearest water body above this world position for caustics, and
/// the nearest waterline for the wet-edge band. A cave aquifer is selected
/// for cave terrain because the surface lake's bed is far above that terrain.
LocalWater local_water(vec3 p) {
    LocalWater w;
    w.caustic_surface = 0.0;
    w.wet_surface = 0.0;
    w.actual_water = false;
    w.wet = false;
    if (water_grid.header.w < 0.5) {
        return w;
    }
    int size = int(water_grid.header.z);
    ivec2 cell = ivec2(floor(p.xz)) - ivec2(water_grid.header.xy);
    if (cell.x < 0 || cell.y < 0 || cell.x >= size || cell.y >= size) {
        return w;
    }
    int base = (cell.y * size + cell.x) * WATER_GRID_LAYERS;
    float best_caustic = 1e9;
    float best_wet_dist = 1e9;
    for (int i = 0; i < WATER_GRID_LAYERS; i++) {
        vec2 body = water_grid.bodies[base + i];
        if (body.x <= WATER_GRID_NONE + 1.0) {
            continue;
        }
        if (body.x < 0.0) {
            // Dry fringe: wet edge only, never caustics.
            float h = -body.x;
            float d = abs(p.y - (h - 0.05));
            if (d < 0.55 && d < best_wet_dist) {
                best_wet_dist = d;
                w.wet = true;
                w.wet_surface = h;
            }
            continue;
        }
        float surface = body.x;
        float bed = body.y;
        // Caustics belong to the body directly above this fragment. Checking
        // the bed rejects a distant lake surface when the fragment is inside
        // a cave whose roof separates it from that lake.
        if (surface > p.y && bed <= p.y + 1.05 && surface < best_caustic) {
            best_caustic = surface;
            w.actual_water = true;
            w.caustic_surface = surface;
        }
        float d = abs(p.y - (surface - 0.05));
        if (d < 0.55 && d < best_wet_dist) {
            best_wet_dist = d;
            w.wet = true;
            w.wet_surface = surface;
        }
    }
    return w;
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
    vec3 normal;         // outward normal of the blocking face (if hit)
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
    r.normal = -d;
    int axis = -1;
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
                // Inverse-square falloff from a small emitting volume.
                float fade = 1.0 / (0.25 + t * t);
                r.emission += e * level * 10.0 * fade * r.transmittance;
            }
            if (m.kind == BLOCK_KIND_OPAQUE) {
                r.hit = true;
                r.block = id;
                // Nothing survives an opaque hit. Keeping this at 1.0 made
                // every visibility check that reads `transmittance` treat a
                // blocked ray as fully lit (the cause of daylight in caves).
                r.transmittance = 0.0;
                r.distance = t;
                r.position = origin + d * t;
                if (axis == 0) r.normal = vec3(-sign(d.x), 0.0, 0.0);
                else if (axis == 1) r.normal = vec3(0.0, -sign(d.y), 0.0);
                else if (axis == 2) r.normal = vec3(0.0, 0.0, -sign(d.z));
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
            axis = 0;
        } else if (tmax.y < tmax.z) {
            t = tmax.y;
            tmax.y += tdelta.y;
            cell.y += stp.y;
            axis = 1;
        } else {
            t = tmax.z;
            tmax.z += tdelta.z;
            cell.z += stp.z;
            axis = 2;
        }
        if (t > max_dist) break;
    }
    return r;
}

// Fixed low-discrepancy taps inside the light's angular disc. These are NOT
// rotated per pixel or per frame: a stable penumbra is far less objectionable
// than the shimmering grain produced by stochastic shadow rays.
const vec2 SHADOW_TAPS[8] = vec2[8](
    vec2(0.0, 0.0),
    vec2(0.85, 0.20),
    vec2(-0.65, 0.55),
    vec2(0.10, -0.90),
    vec2(-0.40, -0.55),
    vec2(0.55, 0.75),
    vec2(-0.90, -0.15),
    vec2(0.75, -0.65)
);

// Shadow ray toward `light_dir`. Samples a fixed set of offsets inside the
// light's angular disc to produce a stable penumbra.
float soft_visibility(vec3 p, vec3 n, vec3 light_dir, float angular_radius) {
    if (light_dir.y <= 0.0) return 0.0;
    vec3 helper = abs(light_dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t1 = normalize(cross(light_dir, helper));
    vec3 t2 = cross(light_dir, t1);
    int taps = rt.grid_extent.w > 0.66 ? 6 : (rt.grid_extent.w > 0.33 ? 4 : 3);
    vec3 origin = p + n * 0.02;
    float sum = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= taps) break;
        vec2 o = SHADOW_TAPS[i] * angular_radius * 1.8;
        vec3 dir = normalize(light_dir + t1 * o.x + t2 * o.y);
        sum += trace_voxels(origin, dir, 72.0, 96, false).transmittance;
    }
    return sum / float(taps);
}

// Average block albedo (sRGB8 -> linear) used by diffuse one-bounce GI.
vec3 block_albedo(uint id) {
    BlockLightGpu m = block_light(id);
    vec3 a = vec3(
        float(m.albedo_rgb & 255u),
        float((m.albedo_rgb >> 8u) & 255u),
        float((m.albedo_rgb >> 16u) & 255u)) / 255.0;
    return max(pow(a, vec3(2.2)), vec3(0.02));
}

// Soft visibility for a point light. A handful of fixed taps inside the
// light's tiny angular disc gives a stable penumbra; opaque geometry blocks
// every tap (so an enclosed torch is fully dark), while glass/water/leaves
// attenuate and slightly diffuse the surviving taps instead of hard-cutting.
float point_visibility(vec3 p, vec3 n, vec3 l, float d) {
    vec3 helper = abs(l.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t1 = normalize(cross(l, helper));
    vec3 t2 = cross(l, t1);
    // The visible flame/glow volume is wider than the physical wick. A larger
    // angular source makes the penumbra wide enough that shadows from
    // different torches overlap and blend instead of meeting in hard lines.
    float radius = clamp(0.85 / max(d, 0.75), 0.018, 0.45);
    // Eight fixed taps at high quality smooth the penumbra into a continuous
    // gradient rather than a handful of visible contour bands.
    int taps = rt.grid_extent.w > 0.66 ? 8 : (rt.grid_extent.w > 0.33 ? 5 : 4);
    vec3 origin = p + n * 0.035;
    float sum = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= taps) break;
        vec2 o = SHADOW_TAPS[i] * radius;
        vec3 dir = normalize(l + t1 * o.x + t2 * o.y);
        TraceHit th = trace_voxels(origin, dir, max(d - 0.30, 0.0), 48, false);
        sum += th.hit ? 0.0 : th.transmittance;
    }
    return sum / float(taps);
}

// Cheap softened visibility for indirect point-light bounces. A single hard
// ray made indirect torch shadows meet in visible lines; three taps are enough
// to break up that edge without the cost of the full eight-tap direct pass.
float point_visibility_cheap(vec3 p, vec3 n, vec3 l, float d) {
    vec3 helper = abs(l.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t1 = normalize(cross(l, helper));
    vec3 t2 = cross(l, t1);
    float radius = clamp(0.70 / max(d, 0.75), 0.018, 0.38);
    vec3 origin = p + n * 0.035;
    float sum = 0.0;
    for (int i = 0; i < 3; i++) {
        vec2 o = SHADOW_TAPS[i] * radius;
        vec3 dir = normalize(l + t1 * o.x + t2 * o.y);
        TraceHit th = trace_voxels(origin, dir, max(d - 0.30, 0.0), 48, false);
        sum += th.hit ? 0.0 : th.transmittance;
    }
    return sum / 3.0;
}

// Direct illumination from CPU-gathered emissive blocks. Sharp point lights
// with inverse-square falloff, a smooth cutoff and ray-traced occlusion.
// shadow_mode: 0 = none (debug), 1 = soft direct, 2 = cheap soft bounce.
vec3 sample_emissive_lights(vec3 p, vec3 n, int shadow_mode) {
    vec3 sum = vec3(0.0);
    int count = int(lights.light_meta.x);
    for (int i = 0; i < 32; i++) {
        if (i >= count) break;
        vec4 pr = lights.light_pos_range[i];
        // Reject stale lights immediately. Covering a torch replaces its
        // source voxel with a non-emissive block before the throttled CPU
        // scan runs; without this check the old point light keeps shining
        // through the new cover for several frames.
        uint source = voxel_id(ivec3(floor(pr.xyz)));
        if (block_light(source).emission_level == 0u) continue;
        vec3 to_light = pr.xyz - p;
        float d2 = dot(to_light, to_light);
        float range = pr.w;
        if (d2 >= range * range || d2 < 1e-4) continue;
        float d = sqrt(d2);
        vec3 l = to_light / d;
        // Smooth wrapped diffuse: a linear clamp leaves a visible crease
        // where N.L crosses zero. Smoothstep approaches zero with zero slope,
        // so overlapping torch pools fade into each other instead of forming
        // a hard diagonal terminator. Shadowing still blocks walls completely.
        float ndl = smoothstep(-0.15, 0.85, dot(n, l));
        if (ndl <= 0.0) continue;
        // Inverse-square irradiance from a finite flame volume: the 0.25
        // term is the squared 0.5-block source radius, which keeps the very
        // close field finite without flattening the whole falloff curve.
        // This gives a bright pool near the flame and a much faster drop-off
        // than the old 1/(1 + 0.12 d^2) approximation.
        float atten = 1.0 / (0.25 + d2);
        // Fade only where inverse-square light is already weak, so the range
        // limit cannot draw a visible ring.
        atten *= 1.0 - smoothstep(range * 0.6, range, d);
        float vis = 1.0;
        if (shadow_mode == 1) {
            vis = point_visibility(p, n, l, d);
        } else if (shadow_mode == 2) {
            vis = point_visibility_cheap(p, n, l, d);
        }
        vec4 ci = lights.light_color_intensity[i];
        sum += ci.rgb * ci.w * atten * ndl * vis;
    }
    return sum;
}

// Direct light arriving at an arbitrary hit point, used by GI bounces.
// Emissive light is shadowed here: unshadowed bounce light was the source of
// torchlight leaking through walls into sealed rooms.
vec3 surface_direct_light(vec3 p, vec3 n) {
    vec3 sd = normalize(rt.sun_dir_vis.xyz);
    float sv = clamp(rt.sun_dir_vis.w, 0.0, 1.0);
    float ndl = max(dot(n, sd), 0.0);
    float vis = 0.0;
    if (ndl > 0.0 && sv > 0.001) {
        TraceHit th = trace_voxels(p + n * 0.03, sd, 48.0, 64, false);
        vis = th.hit ? 0.0 : th.transmittance;
    }
    vec3 c = rt.sun_color_radius.rgb * ndl * sv * vis;
    // Sky ambient at the bounce point must also be occluded, otherwise every
    // interior wall would inject daylight into a sealed room through GI.
    float sky_vis = trace_voxels(p + n * 0.03, vec3(0.0, 1.0, 0.0),
                                 64.0, 96, false).transmittance;
    c += sky_radiance(n) * 0.35 * sky_vis;
    c += sample_emissive_lights(p + n * 0.03, n, 2);
    return c;
}

// A single extra cosine ray at a GI hit, approximating the second diffuse
// bounce. This is what lets warm light wrap past a corner instead of stopping
// at the first lit face. Only evaluated at high ray-tracing quality.
vec3 second_bounce(vec3 p, vec3 n, vec3 incoming) {
    vec3 rd = reflect(incoming, n);
    if (dot(rd, n) <= 0.01) rd = n;
    rd = normalize(rd);
    TraceHit h = trace_voxels(p + n * 0.03, rd, 10.0, 16, true);
    vec3 c = h.emission;
    if (!h.hit) {
        c += sky_radiance(rd) * h.transmittance;
    } else {
        c += block_albedo(h.block) * surface_direct_light(h.position, h.normal);
    }
    return c;
}

// Cosine-weighted hemisphere GI from a fixed Fibonacci direction set. The
// directions do not change per pixel or per frame; only real occlusion varies,
// which removes the stochastic grain while keeping contact shadows.
// A hit surface contributes its albedo times its own direct sun/sky/emissive
// lighting, giving a true one-bounce colour bleed.
vec3 sky_ambient_raytraced(vec3 p, vec3 n) {
    vec3 up = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);
    int samples = rt.grid_extent.w > 0.66 ? 8 : (rt.grid_extent.w > 0.33 ? 5 : 3);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (i >= samples) break;
        float u = (float(i) + 0.5) / float(samples);
        float phi = 2.39996323 * float(i);
        float cos_theta = sqrt(1.0 - u);
        float sin_theta = sqrt(u);
        vec3 d = normalize(t * cos(phi) * sin_theta
                         + b * sin(phi) * sin_theta
                         + n * cos_theta);
        // Long enough to reach the roof of large caves/rooms. A short trace
        // used to miss the ceiling and then sample sky, leaking ambient light
        // into underground spaces.
        TraceHit hit = trace_voxels(p + n * 0.02, d, 48.0, 64, true);
        vec3 radiance = hit.emission;
        if (!hit.hit) {
            // Only count sky on a miss if the ray actually escaped to open
            // sky. In a large sealed room the ray can exceed the trace range
            // while still indoors; checking upward from its endpoint keeps
            // that space pitch dark.
            vec3 endpoint = p + d * 48.0;
            float end_sky = trace_voxels(endpoint + vec3(0.0, 0.01, 0.0),
                                         vec3(0.0, 1.0, 0.0), 64.0, 96,
                                         false).transmittance;
            radiance += sky_radiance(d) * hit.transmittance * end_sky;
        } else {
            vec3 albedo = block_albedo(hit.block);
            vec3 bounce = surface_direct_light(hit.position, hit.normal);
            if (rt.grid_extent.w > 0.66) {
                bounce += second_bounce(hit.position, hit.normal, d) * 0.55;
            }
            radiance += albedo * bounce;
        }
        sum += radiance;
    }
    return sum / float(samples);
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
    // A reflected ray that hit geometry must not also collect sky radiance:
    // doing so lit sealed rooms/underground surfaces as if the wall were a
    // window. Off-screen hits stay black (or use the shaded scene copy below).
    vec3 colour = vec3(0.0);
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

// Generated from crates/world/src/water_wave_model.rs. The CPU evaluator is
// emitted from the same expression tree, so the two cannot drift by hand.
#include "water_wave.generated.glsl"

// Analytic normal of the same wave field (central difference).
vec3 water_wave_normal(vec3 wp, uint kind, float amp, float angle,
                       float rapids, float t) {
    if (amp <= 0.0) return vec3(0.0, 1.0, 0.0);
    float e = 0.12;
    float h0 = water_wave_height(kind, amp, angle, rapids, wp.xz, t);
    float hx = water_wave_height(kind, amp, angle, rapids, wp.xz + vec2(e, 0.0), t);
    float hz = water_wave_height(kind, amp, angle, rapids, wp.xz + vec2(0.0, e), t);
    return normalize(vec3(-(hx - h0) / e, 1.0, -(hz - h0) / e));
}

// Fine wind ripples layered on top of the displaced swell. These are normal
// detail only (no vertex displacement), which gives the surface a broken,
// water-like highlight without increasing the transparent mesh tessellation.
vec3 water_ripple_normal(vec2 p, float t) {
    const float a1 = 0.58;
    const float a2 = -0.74;
    float k1 = WATER_TAU / 2.6;
    float k2 = WATER_TAU / 1.55;
    float w1 = sqrt(WATER_G * k1);
    float w2 = sqrt(WATER_G * k2);
    float along1 = p.x * cos(a1) + p.y * sin(a1);
    float along2 = p.x * cos(a2) + p.y * sin(a2);
    float amp1 = 0.055;
    float amp2 = 0.032;
    float phase1 = k1 * along1 - w1 * t * 1.15;
    float phase2 = k2 * along2 - w2 * t * 1.35 + 2.1;
    float dhdx = amp1 * k1 * cos(phase1) * cos(a1)
               + amp2 * k2 * cos(phase2) * cos(a2);
    float dhdz = amp1 * k1 * cos(phase1) * sin(a1)
               + amp2 * k2 * cos(phase2) * sin(a2);
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

    BlockMaterialGpu mat = material_table.materials[actual_tile];
    uint flags = mat.flags_roughness_emissive_pad & 0xFFu;
    bool is_water_block = (flags & MATERIAL_FLAG_WATER) != 0u;
    uint water_kind = frag_water & 7u;
    float water_amp = frag_water_amp;
    float water_angle = float((frag_water >> 16) & 255u) * (WATER_TAU / 255.0);
    float water_rapids = float((frag_water >> 4) & 15u) / 15.0;
    float water_flow = float((frag_water >> 24) & 255u) / 255.0 * 2.6;
    float water_time = frag_water_time;

    ivec2 tile_pos = ivec2(actual_tile % 16u, actual_tile / 16u);
    // River water scrolls downstream. The direction and speed come from the
    // mesher's flow metadata, so a fast narrow chute visibly races while a
    // broad calm river drifts.
    vec2 tile_uv = vec2(fract(frag_uv.x), fract(frag_uv.y));
    if (is_water_block && water_kind == 2u) {
        vec2 flow = vec2(cos(water_angle), sin(water_angle));
        tile_uv = fract(tile_uv + flow * water_flow * water_time * 0.18);
    }
    vec2 atlas_uv = (vec2(tile_pos) + tile_uv) / ATLAS_TILES;

    vec4 tex = texture(atlas, atlas_uv);
    if (tex.a < 0.1) {
        discard;
    }

    // Atlas textures are authored in sRGB but sampled through a UNORM view.
    vec3 albedo = pow(tex.rgb, vec3(2.2));

    float roughness = float((mat.flags_roughness_emissive_pad >> 8u) & 0xFFu) / 255.0;
    float emissive_amount = float((mat.flags_roughness_emissive_pad >> 16u) & 0xFFu) / 255.0;
    float water_y = material_table.world_params.x;
    float wet_edge_amount = material_table.world_params.y;
    float caustics_amount = material_table.world_params.z;
    float leaves_amount = material_table.world_params.w;

    // Follow the local water bodies (lake/river/cave surfaces). If the grid
    // is not valid yet (first frame), keep the legacy global fallback.
    LocalWater local = local_water(frag_world_pos);
    if (water_grid.header.w < 0.5) {
        local.caustic_surface = water_y;
        local.wet_surface = water_y;
        local.actual_water = true;
        local.wet = true;
    }

    vec3 cam_pos = cam.cam_pos_and_maxdist.xyz;
    vec3 to_eye = cam_pos - frag_world_pos;
    float view_dist = length(to_eye);
    vec3 v = to_eye / max(view_dist, 1e-4);
    // Water surfaces are displaced at 1 m vertices, so the geometric face
    // normal is constant per triangle and produces visible triangular flashes
    // in the sun/reflection path. Use the analytic normal of the wave field
    // for every up-facing water fragment; keep the face normal for vertical
    // shoreline steps. This also makes the shadow rays use the same smooth
    // surface the reflection/refraction code sees.
    vec3 n = face_normal(v);
    if (is_water_block && abs(n.y) > 0.2) {
        n = water_wave_normal(frag_world_pos, water_kind, water_amp,
                              water_angle, water_rapids, water_time);
        if (water_amp > 0.02) {
            vec3 ripple = water_ripple_normal(frag_world_pos.xz, water_time);
            n = normalize(vec3(n.x + ripple.x, n.y, n.z + ripple.z));
        }
        if (dot(n, v) < 0.0) n = -n;
    }

    vec3 sun_dir = normalize(rt.sun_dir_vis.xyz);
    vec3 moon_dir = normalize(rt.moon_dir_phase.xyz);
    float sun_vis = clamp(rt.sun_dir_vis.w, 0.0, 1.0);
    float sun_ndl = max(dot(n, sun_dir), 0.0);
    float moon_ndl = max(dot(n, moon_dir), 0.0);

    // Static per-pixel hash: only used to break up rough reflections. It must
    // not include time, or every ray-traced term shimmers between frames.
    float seed = hash12(gl_FragCoord.xy);
    float shadow_softness = max(rt.params.y, 0.0008);
    float sun_ang = max(rt.sun_color_radius.a, shadow_softness);

    // ── Ray-traced direct sun + moon ─────────────────────────────────────
    float sun_shadow = 1.0;
    if (sun_vis > 0.001 && sun_ndl > 0.0) {
        sun_shadow = soft_visibility(frag_world_pos, n, sun_dir, sun_ang);
    }
    float moon_shadow = 1.0;
    if (moon_ndl > 0.0) {
        moon_shadow = soft_visibility(frag_world_pos, n, moon_dir,
                                      max(rt.moon_color_radius.a, shadow_softness));
    }
    vec3 direct = rt.sun_color_radius.rgb * sun_ndl * sun_vis * sun_shadow;
    direct += rt.moon_color_radius.rgb * moon_ndl * moon_shadow;

    // ── Ray-traced ambient occlusion + sky GI ────────────────────────────
    vec3 ambient = sky_ambient_raytraced(frag_world_pos, n);
    ambient *= rt.params.z;

    // ── Dynamic emissive block lights (torches, lava, glowstone) ─────────
    // Direct point lights with ray-traced occlusion: these produce smooth,
    // physically-shaped pools of light instead of sparse gather-ray hits.
    vec3 emissive_light = sample_emissive_lights(frag_world_pos, n, 1);

    vec3 lighting = direct + ambient + emissive_light;

    // ── Specular reflection (ray traced) ─────────────────────────────────
    float gloss = (1.0 - roughness) * clamp(rt.params.x, 0.0, 1.0);
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
        float t = water_time;
        // `n` was already replaced with the analytic wave normal on up-facing
        // water above. Do not recompute or mix it with the faceted geometry
        // normal here: that mix was the source of triangle-shaped highlights.
        vec3 wn = n;
        float cos_theta = clamp(dot(v, wn), 0.0, 1.0);
        float fres = fresnel_schlick(cos_theta, 0.02) * clamp(rt.params.x, 0.0, 1.0);

        // Reflection: traced through the voxel world, falling back to sky.
        vec3 reflected = trace_reflection(frag_world_pos, wn, v, 0.08, seed);
        // A traced reflection can be black when it hits off-screen geometry,
        // and a wave face tilted toward the camera can reflect below the
        // horizon. Both produced thin black slivers across the surface. Keep
        // the reflection no darker than the sky/water base instead.
        reflected = max(reflected, sky_radiance(reflect(-v, wn)) * 0.65);
        reflected = max(reflected, vec3(0.035, 0.16, 0.24));
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
        // Waves bend the view ray: shift the opaque-scene lookup along the
        // surface normal, then absorb over the true water-column length.
        float bend = 0.006 + 0.020 * water_rapids;
        vec2 refract_uv = clamp(scene_uv + wn.xz * bend * clamp(column * 0.5, 0.25, 1.0),
                                vec2(0.001), vec2(0.999));
        vec3 refracted = textureLod(scene_opaque, refract_uv, 0.0).rgb * beer;
        refracted = mix(refracted, refracted * (albedo * 2.6 + 0.03), 0.30);
        // The scene copy can contain unrendered/black pixels just outside the
        // shoreline; keep those from turning into black cracks in the water.
        refracted = max(refracted, vec3(0.02, 0.10, 0.16));
        // Dark, saturated deep water instead of a mirror of the pale seabed.
        // At depth the block grid of the seabed should disappear; near shore
        // the bottom remains visible through the shallow-water path.
        vec3 water_base = vec3(0.035, 0.16, 0.24);
        float depth_fade = smoothstep(1.5, 10.0, column);
        refracted = mix(refracted, water_base, depth_fade * 0.55);
        // Water body receives the same traced lighting (sparkle + depth).
        refracted += water_base * (ambient + direct + emissive_light) * 0.30;

        vec3 water_col = mix(refracted, reflected, clamp(fres + 0.10, 0.0, 1.0));
        water_col = max(water_col, vec3(0.015, 0.07, 0.11));

        // White water: rapids, the shallow band at the shore, and the crests
        // of large swells (which only reach full height well offshore).
        float foam = water_rapids * 0.45;
        foam += (1.0 - smoothstep(0.4, 2.5, column)) * 0.28;
        if (water_amp > 0.05) {
            float crest = water_wave_height(water_kind, water_amp, water_angle,
                                            water_rapids, frag_world_pos.xz, t)
                        / max(water_amp, 1e-4);
            foam += smoothstep(0.80, 1.0, crest) * 0.18;
        }
        foam = clamp(foam, 0.0, 0.75);
        water_col = mix(water_col, vec3(0.82, 0.90, 0.94), foam);

        tinted = water_col;
        // Deep water must not show the opaque seabed through the blend: its
        // block grid reads as cracks in the otherwise smooth surface. Keep
        // shallow water translucent so the bottom and shoreline stay visible.
        out_alpha = clamp(tex.a + fres * 1.2 + foam * 0.3 + depth_fade * 0.50, 0.0, 1.0);
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
    if (!is_water_block && caustics_amount > 0.0 && local.actual_water &&
            frag_world_pos.y < local.caustic_surface) {
        float depth_below = local.caustic_surface - frag_world_pos.y;
        float depth_attenuation = clamp(1.0 - depth_below / 4.0, 0.0, 1.0);
        float pattern = sin(frag_world_pos.x * 0.6 + rt.params.w * 1.3)
                      * cos(frag_world_pos.z * 0.6 - rt.params.w * 0.9);
        float glare = max(pattern, 0.0) * depth_attenuation * caustics_amount
                    * sun_shadow * sun_vis;
        tinted += glare * rt.sun_color_radius.rgb * 0.10;
    }

    // ── Wet edge ─────────────────────────────────────────────────────────
    if (wet_edge_amount > 0.0 && !is_water_block && local.wet) {
        float band_dist = abs(frag_world_pos.y - (local.wet_surface - 0.05));
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
        float sun_fog = 0.0;
        if (sun_vis > 0.001) {
            sun_fog = soft_visibility(mid, -sun_dir, sun_dir, sun_ang * 3.0);
        }
        float moon_fog = 0.0;
        if (moon_dir.y > 0.001) {
            moon_fog = soft_visibility(mid, -moon_dir, moon_dir,
                                       max(rt.moon_color_radius.a, shadow_softness) * 3.0);
        }
        // Fog is only lit where it can actually see the sky/sun/moon. A dark
        // cave or sealed room therefore fades to black instead of grey haze.
        // Water is a horizontal surface; tracing straight up from a midway
        // point that can be inside a wave produces broken dark fog bands.
        // Water already gets its depth/reflection treatment above, so use the
        // open-sky fog colour instead of the voxel shadow ray here.
        float sky_fog = is_water_block
            ? 1.0
            : trace_voxels(mid, vec3(0.0, 1.0, 0.0), 96.0, 128, false).transmittance;
        vec3 haze = fog_col * sky_fog
                  + rt.sun_color_radius.rgb * sun_fog * 0.55
                  + rt.moon_color_radius.rgb * moon_fog * 0.25;
        tinted = mix(tinted, haze, clamp(fog_t * min(rt.fog_density.a, 1.0), 0.0, 1.0));
    }

    out_color = vec4(max(tinted, vec3(0.0)), out_alpha);
}
