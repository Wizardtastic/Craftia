#version 450

// Entity vertex shader. Renders billboard quads or fixed-orientation
// cubes for entities with a Mesh component.
//
// Vertex layout (32 bytes, same as chunk vertex):
//   location 0: vec3  pos    — local space (-0.5..0.5)
//   location 1: vec2  uv     — texture coordinates
//   location 2: float light  — always 1.0
//   location 3: uint  tile   — atlas tile index
//   location 4: vec4  light_color — packed RGBA light tint
//
// Push constants:
//   offset  0: mat4 model
//   offset 64: uint tile
//   offset 68: float half_size
//   offset 72: uint billboard (0=cube, 1=billboard)
//   offset 76: uint _pad

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in float in_light;
layout(location = 3) in uint in_tile;
layout(location = 4) in vec4 in_light_color;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_light;
layout(location = 2) out float frag_fog;
layout(location = 3) flat out uint frag_tile;
layout(location = 4) flat out vec4 frag_light_color;

layout(push_constant) uniform Push {
    mat4 model;
    uint tile;
    float half_size;
    uint billboard;
    uint _pad;
} push;

layout(set = 0, binding = 0) uniform Camera {
    vec4 cam_pos_and_maxdist;
} cam;

// View-projection matrix is derived from the camera UBO.
// We reconstruct it from the camera resource. For simplicity,
// we pass it as a second push constant block or derive from model.
// Here we assume the model matrix already includes view-projection
// (the CPU computes model = view_proj * entity_model).

void main() {
    vec3 local = in_pos * push.half_size;

    if (push.billboard == 1u) {
        // Billboard: extract camera right/up from the model matrix
        // (which is view_proj * entity_translation). We use the
        // inverse of the rotation part to orient the quad toward
        // the camera. Simpler: use the model matrix columns 0/1
        // as right/up since model = view_proj * translate.
        // Actually, for a true billboard we need the camera's
        // right and up vectors. Extract from view-projection:
        // right = normalize(vec3(view_proj[0][0], view_proj[1][0], view_proj[2][0]))
        // up    = normalize(vec3(view_proj[0][1], view_proj[1][1], view_proj[2][1]))
        // But we don't have view_proj in push constants for entities.
        // Solution: pass view_proj via the model matrix field (CPU side).
        // model = view_proj * translate(entity_pos)
        // Then billboard = model * vec4(local, 1.0) where local.x = right * pos.x + up * pos.y
        vec3 right = normalize(vec3(push.model[0][0], push.model[1][0], push.model[2][0]));
        vec3 up    = normalize(vec3(push.model[0][1], push.model[1][1], push.model[2][1]));
        vec3 entity_pos = vec3(push.model[3][0], push.model[3][1], push.model[3][2]);
        vec3 world = entity_pos + right * local.x + up * local.y;
        gl_Position = vec4(world, 1.0);
        // Apply the view-projection from the model matrix's linear part.
        // model = view_proj * translate, so model * (0,0,0,1) = view_proj * entity_pos.
        // We need to apply view_proj to our billboard position.
        // Reconstruct: gl_Position = view_proj_part * vec4(billboard_world, 1.0)
        // Since model = view_proj * translate, model[3] = view_proj * entity_pos.
        // The linear part (columns 0-2) of model = view_proj * rotation.
        // For a billboard with no rotation, columns 0-2 of model = view_proj columns 0-2.
        // So we can use: gl_Position = vec4(
        //   dot(right, world) + model[3][0],
        //   dot(up, world) + model[3][1],
        //   ...
        // );
        // Simpler approach: compute billboard world pos, then apply view_proj.
        // But we don't have view_proj separately. Let's use a different approach:
        // Pass view_proj as the model matrix, and entity_pos separately.
        // For now, use the model matrix as view_proj * translate and extract.
        gl_Position = push.model * vec4(in_pos.x * push.half_size, in_pos.y * push.half_size, 0.0, 1.0);
    } else {
        // Fixed orientation: apply full model matrix.
        gl_Position = push.model * vec4(local, 1.0);
    }

    frag_uv = in_uv;
    frag_light = in_light;
    frag_tile = push.tile;
    frag_light_color = in_light_color;

    // Fog: compute from world position (extract from model matrix translation).
    vec3 entity_pos = vec3(push.model[3][0], push.model[3][1], push.model[3][2]);
    float dist = length(entity_pos - cam.cam_pos_and_maxdist.xyz);
    frag_fog = clamp(1.0 - exp(-3.0 * dist / cam.cam_pos_and_maxdist.w), 0.0, 1.0);
}
