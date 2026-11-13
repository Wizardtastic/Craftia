#version 450

// Sky fragment shader — analytic atmospheric model.
//
// The old shader mixed two engine-supplied colours and added a fixed white
// sun disc. This version derives everything from the sun's elevation:
//
//   * Air mass (1 / (sin(elevation) + k)) drives a per-channel extinction
//     (Rayleigh-ish) so the sun whites out at noon and reddens at the
//     horizon, and the twilight band only appears near sunrise/sunset.
//   * A horizon→zenith gradient is mixed with a forward-scattering lobe
//     around the sun (Mie glow) and an exaggerated disc for HDR bloom.
//   * A moon disc (opposite the sun) and two layers of twinkling stars take
//     over as the sun sets. Night sky radiance comes from the engine's sky
//     ambient term so the sky and the lit world stay consistent.
//
// Output is linear HDR — the offscreen colour target is RGBA16F and the post
// pass applies exposure, bloom and ACES tone mapping.

layout(location = 0) in vec3 frag_dir;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Sky {
    vec4 sun_dir_vis;    // xyz = sun direction, w = visibility (0..1)
    vec4 sun_color;      // rgb = sun tint * intensity (HDR), a unused
    vec4 sky_ambient;    // rgb = sky ambient irradiance, a = night factor
    vec4 fog_time;       // rgb = fog colour, a = time (seconds)
} sky;

const float PI = 3.14159265359;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Two layers of sparse, twinkling stars keyed off a quantised direction.
vec3 star_field(vec3 dir, float t) {
    vec3 col = vec3(0.0);
    for (int layer = 0; layer < 2; layer++) {
        float scale = layer == 0 ? 170.0 : 310.0;
        vec3 p = dir * scale;
        vec3 id = floor(p);
        vec3 f = fract(p) - 0.5;
        float h = hash21(id.xy + id.z * 7.13);
        float threshold = layer == 0 ? 0.984 : 0.992;
        if (h > threshold) {
            float mag = (h - threshold) / (1.0 - threshold);
            float twinkle = 0.72 + 0.28 * sin(t * (1.3 + h * 2.7) + h * 40.0);
            float core = smoothstep(0.24, 0.0, length(f));
            col += vec3(0.82, 0.88, 1.0) * core * twinkle * (0.35 + mag * 2.2);
        }
    }
    return col;
}

void main() {
    vec3 dir = normalize(frag_dir);
    vec3 sun_dir = normalize(sky.sun_dir_vis.xyz);
    float sun_vis = sky.sun_dir_vis.w;   // 0 at night, 1 in full day
    float night = sky.sky_ambient.w;     // 0 at day, 1 at night
    float sun_up = sun_dir.y;

    // 0 in deep night, 1 once the sun is well above the horizon.
    float day = smoothstep(-0.10, 0.22, sun_up);
    // Bell curve around the horizon: sunrise/sunset scattering band.
    float twilight = exp(-pow((sun_up - 0.02) / 0.16, 2.0));

    // Approximate optical air mass. Near the horizon the path through the
    // atmosphere is long, so blue light is extinguished first (sunset red).
    float air_mass = 1.0 / (max(sun_up, -0.16) + 0.18);
    vec3 extinction = exp(-air_mass * vec3(0.35, 0.95, 2.40) * 0.55);
    vec3 sun_tint = mix(vec3(1.0), extinction, 0.85);

    // Hue of the daytime sky. Kept in linear space; the post pass tonemaps.
    vec3 zenith_day   = vec3(0.145, 0.360, 0.860);
    vec3 horizon_day  = vec3(0.610, 0.775, 0.980);
    vec3 zenith_night = vec3(0.004, 0.006, 0.016);
    vec3 horizon_night = vec3(0.010, 0.016, 0.034);

    vec3 zenith = mix(zenith_night, zenith_day, day);
    vec3 horizon = mix(horizon_night, horizon_day, day);

    float up = clamp(dir.y, 0.0, 1.0);
    // Slight bias toward the horizon so the gradient reads like real sky.
    float grad = pow(up, 0.42);
    vec3 sky_col = mix(horizon, zenith, grad);

    // Azimuth aligned with the sun (guarded against a degenerate vector).
    vec2 sun_az = sun_dir.xz;
    float az_len = length(sun_az);
    sun_az = az_len > 1e-4 ? sun_az / az_len : vec2(1.0, 0.0);
    vec2 dir_az = dir.xz;
    float dir_az_len = length(dir_az);
    dir_az = dir_az_len > 1e-4 ? dir_az / dir_az_len : sun_az;
    float sunward = clamp(dot(dir_az, sun_az) * 0.5 + 0.5, 0.0, 1.0);

    // ── Mie forward scattering around the sun ──
    float mu = max(dot(dir, sun_dir), 0.0);
    float mie = pow(mu, 8.0) * 0.28 + pow(mu, 96.0) * 0.42 + pow(mu, 1200.0) * 2.6;
    sky_col += sun_tint * mie * sun_vis * vec3(1.0, 0.94, 0.84);

    // ── Twilight band hugging the horizon, strongest toward the sun ──
    sky_col += sun_tint * twilight * pow(1.0 - up, 3.0)
             * (0.25 + 0.75 * sunward) * 1.15 * sun_vis;

    // Warm glow bleeding up from the horizon on the sun's side.
    sky_col += sun_tint * twilight * pow(up, 0.35) * sunward * 0.12 * sun_vis;

    // ── Sun disc (HDR, bloomed by the post pass) ──
    float disc = smoothstep(0.99980, 0.99990, dot(dir, sun_dir));
    float sun_energy = sky.sun_color.r * 0.85 + 0.35;
    sky_col += sun_tint * disc * sun_energy * 9.0 * smoothstep(-0.03, 0.02, sun_up);

    // ── Moon: opposite the sun, with a soft glow and faint maria ──
    vec3 moon_dir = -sun_dir;
    if (moon_dir.y > -0.05) {
        float moon_mu = max(dot(dir, moon_dir), 0.0);
        float moon_disc = smoothstep(0.99972, 0.99988, moon_mu);
        float maria = hash21(floor(dir.xz * 180.0));
        vec3 moon_col = vec3(0.86, 0.90, 1.05);
        float moon_vis = smoothstep(-0.04, 0.06, moon_dir.y);
        sky_col += moon_col * moon_disc * (0.75 + 0.35 * maria) * 3.2 * moon_vis * night;
        sky_col += moon_col * pow(moon_mu, 320.0) * 0.10 * moon_vis * night;
    }

    // ── Stars ──
    sky_col += star_field(dir, sky.fog_time.a) * night
             * smoothstep(-0.02, 0.16, dir.y) * 2.4;

    // Night sky floor from the engine's ambient term keeps the horizon and
    // the world's ambient lighting in the same colour family.
    sky_col += sky.sky_ambient.rgb * 0.45 * night;

    // ── Fog/horizon haze: blend into the world fog colour so distant chunks
    //    and the sky share a horizon line. ──
    float haze = smoothstep(0.14, -0.04, dir.y);
    sky_col = mix(sky_col, sky.fog_time.rgb, haze * 0.75);

    // Slight dither to break up 8-bit banding after tone mapping.
    float dither = (hash21(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
    out_color = vec4(max(sky_col + dither, vec3(0.0)), 1.0);
}
