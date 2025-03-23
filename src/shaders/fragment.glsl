#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = vec4(in_uv, 0.0, 1.0);
}
