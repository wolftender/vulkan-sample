#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 frag_color;

layout(set = 1, binding = 0) uniform sampler2D u_albedo;

void main() {
    vec3 albedo = texture(u_albedo, in_uv).xyz;
    frag_color = vec4(albedo, 1.0);
}
