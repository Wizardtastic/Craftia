#version 450

// Entity vertex shader with skeletal animation (GPU skinning).
//
// Vertex layout (binding 0, 28 bytes — same as entity.vert):
//   location 0: vec3  pos
//   location 1: vec2  uv
//   location 2: float light
//   location 3: uint  tile
//
// Skin data (binding 1, 32 bytes):
//   location 4: uvec4 joint_indices
//   location 5: vec4  joint_weights
//
// Descriptor set 1, binding 0: JointMatrices UBO (64 × mat4)
//
// Push constants (same as entity.vert):
//   offset  0: mat4 model
//   offset 64: uint tile
//   offset 68: float half_size
//   offset 72: uint billboard
//   offset 76: uint _pad

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in float in_light;
layout(location = 3) in uint in_tile;
// Skinning attributes (binding 1)
layout(location = 4) in uvec4 in_joint_indices;
layout(location = 5) in vec4 in_joint_weights;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float frag_light;
layout(location = 2) out float frag_fog;
layout(location = 3) flat out uint frag_tile;

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

// Joint matrices for skinning (set 1, binding 0).
layout(set = 1, binding = 0) uniform JointMatrices {
    mat4 joints[64];
} skin;

void main() {
    // Apply skinning: blend up to 4 joint transforms.
    vec3 skinned_pos = vec3(0.0);

    for (int i = 0; i < 4; i++) {
        uint joint = in_joint_indices[i];
        float weight = in_joint_weights[i];
        if (weight > 0.0 && joint < 64u) {
            mat4 joint_matrix = skin.joints[joint];
            skinned_pos += weight * (joint_matrix * vec4(in_pos, 1.0)).xyz;
        }
    }

    vec3 local = skinned_pos * push.half_size;

    // Billboard not typically used with skinned meshes, but handle it.
    if (push.billboard == 1u) {
        vec3 right = normalize(vec3(push.model[0][0], push.model[1][0], push.model[2][0]));
        vec3 up    = normalize(vec3(push.model[0][1], push.model[1][1], push.model[2][1]));
        vec3 entity_pos = vec3(push.model[3][0], push.model[3][1], push.model[3][2]);
        vec3 world = entity_pos + right * local.x + up * local.y;
        gl_Position = vec4(world, 1.0);
        gl_Position = push.model * vec4(local, 1.0);
    } else {
        // Standard transform: apply model matrix to skinned position.
        gl_Position = push.model * vec4(local, 1.0);
    }

    frag_uv = in_uv;
    frag_light = in_light;
    frag_tile = push.tile;

    // Fog distance.
    vec3 entity_pos = vec3(push.model[3][0], push.model[3][1], push.model[3][2]);
    float dist = length(entity_pos - cam.cam_pos_and_maxdist.xyz);
    frag_fog = clamp(1.0 - exp(-3.0 * dist / cam.cam_pos_and_maxdist.w), 0.0, 1.0);
}
