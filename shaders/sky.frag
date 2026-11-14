#version 450

// Sky fragment shader - analytic atmosphere driven by the astronomical sun and
// moon. The engine computes their real altitude/azimuth from the world's
// latitude/longitude, day of year and time of day, so sunrise and sunset
// directions, twilight length and the moon's phase all follow the date and
// location rather than a fixed animation.

layout(location = 0) in vec3 frag_dir;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Sky {
    vec4 sun_dir_vis;       // xyz = sun dir, w = visibility 0..1
    vec4 sun_color;         // rgb = sun radiance, a = angular radius
    vec4 moon_dir_phase;    // xyz = moon dir, w = phase 0..1
    vec4 moon_color;        // rgb = moon radiance, a = angular radius
    vec4 sky_ambient;       // rgb = ambient floor, a = night factor
    vec4 fog_time;          // rgb = fog colour, a = time (seconds)
} sky;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

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
    vec3 moon_dir = normalize(sky.moon_dir_phase.xyz);
    float sun_vis = sky.sun_dir_vis.w;
    float night = sky.sky_ambient.a;
    float sun_up = sun_dir.y;

    float day = smoothstep(-0.10, 0.22, sun_up);
    float twilight = exp(-pow((sun_up - 0.02) / 0.16, 2.0));

    // Optical air mass: long path near the horizon extinguishes blue first.
    float air_mass = 1.0 / (max(sun_up, -0.16) + 0.18);
    vec3 extinction = exp(-air_mass * vec3(0.35, 0.95, 2.40) * 0.55);
    vec3 sun_tint = mix(vec3(1.0), extinction, 0.85);

    vec3 zenith_day = vec3(0.145, 0.360, 0.860);
    vec3 horizon_day = vec3(0.610, 0.775, 0.980);
    vec3 zenith_night = vec3(0.004, 0.006, 0.016);
    vec3 horizon_night = vec3(0.010, 0.016, 0.034);

    vec3 zenith = mix(zenith_night, zenith_day, day);
    vec3 horizon = mix(horizon_night, horizon_day, day);

    float up = clamp(dir.y, 0.0, 1.0);
    float grad = pow(up, 0.42);
    vec3 sky_col = mix(horizon, zenith, grad);

    vec2 sun_az = sun_dir.xz;
    float az_len = length(sun_az);
    sun_az = az_len > 1e-4 ? sun_az / az_len : vec2(1.0, 0.0);
    vec2 dir_az = dir.xz;
    float d_az_len = length(dir_az);
    dir_az = d_az_len > 1e-4 ? dir_az / d_az_len : sun_az;
    float sunward = clamp(dot(dir_az, sun_az) * 0.5 + 0.5, 0.0, 1.0);

    float mu = max(dot(dir, sun_dir), 0.0);
    float mie = pow(mu, 8.0) * 0.28 + pow(mu, 96.0) * 0.42 + pow(mu, 1200.0) * 2.6;
    sky_col += sun_tint * mie * sun_vis * vec3(1.0, 0.94, 0.84);

    sky_col += sun_tint * twilight * pow(1.0 - up, 3.0)
             * (0.25 + 0.75 * sunward) * 1.15 * sun_vis;
    sky_col += sun_tint * twilight * pow(up, 0.35) * sunward * 0.12 * sun_vis;

    // Sun disc, sized by the physical angular radius passed from the engine.
    float sun_radius = max(sky.sun_color.a, 0.0047);
    float disc_edge = cos(sun_radius);
    float disc = smoothstep(disc_edge + 0.00006, disc_edge + 0.00012, dot(dir, sun_dir));
    float sun_energy = max(sky.sun_color.r, 0.0) * 0.85 + 0.35;
    sky_col += sun_tint * disc * sun_energy * 9.0 * smoothstep(-0.03, 0.02, sun_up);

    // Moon disc + halo, scaled by the illuminated fraction (phase).
    if (moon_dir.y > -0.05) {
        float moon_mu = max(dot(dir, moon_dir), 0.0);
        float moon_vis = smoothstep(-0.04, 0.06, moon_dir.y);
        float phase = clamp(sky.moon_dir_phase.w, 0.0, 1.0);
        float moon_radius = max(sky.moon_color.a, 0.0045);
        float moon_edge = cos(moon_radius);
        float moon_disc = smoothstep(moon_edge + 0.00006, moon_edge + 0.00012, moon_mu);
        float maria = hash21(floor(dir.xz * 180.0));
        sky_col += vec3(0.86, 0.90, 1.05) * moon_disc * (0.75 + 0.35 * maria)
                 * (1.2 + 2.0 * phase) * moon_vis;
        sky_col += vec3(0.86, 0.90, 1.05) * pow(moon_mu, 320.0) * 0.10 * moon_vis;
    }

    sky_col += star_field(dir, sky.fog_time.a) * night
             * smoothstep(-0.02, 0.16, dir.y) * 2.4;

    sky_col += sky.sky_ambient.rgb * 0.45 * night;

    // Horizon haze blends into the world fog for a seamless horizon line.
    float haze = smoothstep(0.14, -0.04, dir.y);
    sky_col = mix(sky_col, sky.fog_time.rgb, haze * 0.75);

    float dither = (hash21(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
    out_color = vec4(max(sky_col + dither, vec3(0.0)), 1.0);
}
