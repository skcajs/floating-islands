#version 460
#extension GL_GOOGLE_include_directive : enable
#include "shader_structures.glsl"

layout (location = 0) in vec3 inPosition; 

layout (location = 0) out vec3 posCube;

// Uniform buffer "uboMatricesAndUserInput", containing camera matrices and user input
layout (set = 0, binding = 0) uniform UniformBlock { matrices_and_user_input uboMatricesAndUserInput; };

void main() {
	posCube = inPosition;
	vec4 posCS = uboMatricesAndUserInput.mProjMatrix * mat4(mat3(uboMatricesAndUserInput.mViewMatrix)) * vec4(inPosition.xyz, 1.0); // mat4(mat3()): remove translation of view matrix
	gl_Position = posCS.xyww; 
}

