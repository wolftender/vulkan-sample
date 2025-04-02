#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_uv;

layout(set = 0, binding = 0) uniform CbPerFrame {
    mat4 view;
    mat4 proj;
} cbPerFrame;

layout(set = 2, binding = 0) uniform CbPerObject {
    mat4 world;
} cbPerObject;

void main() {
    vec4 world_pos = cbPerObject.world * vec4(in_position, 1.0);

    out_position = world_pos.xyz;
    out_normal = in_normal;
    out_uv = in_uv;

    gl_Position = cbPerFrame.proj * cbPerFrame.view * world_pos;
}
