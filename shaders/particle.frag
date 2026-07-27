#version 450

// Particle fragment shader (Phase 2: hard-edged atlas tint + depth-aware
// soft fade).
//
// Picks the right sub-tile of the 16×16 atlas using the per-instance tile
// index, then reads the depth attachment written by subpass 0 to compute
// the soft fade at the particle/scene intersection.
//
// Pipeline blend state is fixed to premultiplied alpha:
//   src.RGB * ONE  +  dst.RGB * (1 - src.A)
//
// Push constants match `shaders/particle.vert`:
//   offset  0: mat4 inv_view_proj (16 floats)
//   offset 64: vec4 soft_near_far (x = softness world-units, y = near, z = far)

const float ATLAS_TILES = 16.0;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;
layout(location = 2) flat in float frag_tile;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler2D atlas;

// Phase 2: input attachment at index 0 (matches pipeline.rs subpass_1
// input_attachments[0].attachment = 1, layout = DEPTH_STENCIL_READ_ONLY_OPTIMAL
// for D32_SFLOAT). With MSAA the depth is multisampled, so we use
// subpassInputMS and sample at gl_SampleID. On single-sample (MSAA off),
// gl_SampleID is always 0 so this still works correctly.
layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInputMS depth_input;

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 soft_near_far;
} push;

// Linearize a [0, 1] Vulkan depth buffer value into world-space distance
// from the camera plane. OpenGL/vulkan standard formula for the [0, 1]
// clip-space projection produced by glam's `Mat4::perspective_rh`.
// Declared AFTER `push` so the per-call .y/.z swizzle resolves.
float linearize(float d) {
    float n = push.soft_near_far.y;
    float f = push.soft_near_far.z;
    return (n * f) / max(f - d * (f - n), 1e-6);
}

void main() {
    // ── Depth-aware soft fade ──
    // The fragments own depth `gl_FragCoord.z` is in the same [0,1] domain as
    // the scene depth; both need linearizing to compare in world-units.
    float scene_d = subpassLoad(depth_input, gl_SampleID).x;
    float frag_d  = gl_FragCoord.z;

    float scene_z = linearize(scene_d);
    float frag_z  = linearize(frag_d);

    // Hard occlusion: the particle surface is behind the scene (further from
    // the camera). soft = 0 here so the alpha fades hard; we still discard
    // anything that is strictly behind by more than the softness margin.
    float diff = scene_z - frag_z;
    if (diff < 0.0) {
        discard;
    }

    float softness = push.soft_near_far.x;
    // 1.0 = full alpha at the particle surface, 0.0 = fully faded into geometry.
    float fade = clamp(diff / max(softness, 1e-6), 0.0, 1.0);

    // ── Atlas & colour tint ──
    float tile_index = clamp(frag_tile, 0.0, 255.0);
    ivec2 tile_pos = ivec2(int(tile_index) % 16, int(tile_index) / 16);
    vec2 atlas_uv = (vec2(tile_pos) + vec2(fract(frag_uv.x), fract(frag_uv.y))) / ATLAS_TILES;

    vec4 tex = texture(atlas, atlas_uv);
    if (tex.a < 0.1) {
        discard;
    }

    vec3 tinted = tex.rgb * frag_color.rgb;
    float a = tex.a * frag_color.a * fade;
    out_color = vec4(tinted * a, a);
}
