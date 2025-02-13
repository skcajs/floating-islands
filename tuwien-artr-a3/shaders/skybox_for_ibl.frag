#version 460
#extension GL_GOOGLE_include_directive : enable
#include "sphere_mapping.glsl"

layout(set = 0, binding = 1) uniform sampler2D texBackground;

layout (location = 0) in  vec3 posCube;
layout (location = 0) out vec4 fs_out;

void main() 
{
    vec2 uv = sphere_map_direction_to_uv(normalize(posCube));
    vec3 color = texture(texBackground, uv).rgb;
	fs_out = vec4(color, 1);
}

