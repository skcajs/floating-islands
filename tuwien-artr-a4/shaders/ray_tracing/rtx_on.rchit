#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#include "../lightsource_limits.h"
#include "../shader_structures.glsl"
#include "../custom_packing.glsl"


// ###### MATERIAL DATA ##################################
// The actual material buffer (of type MaterialGpuData):
// It is bound to descriptor set at index 0 and
// within the descriptor set, to binding location 0
layout(set = 0, binding = 0) buffer Material
{
	MaterialGpuData materials[];
} materialsBuffer;

// Array of samplers containing all the material's images:
// These samplers are referenced from materials by
// index, namely by all those m*TexIndex members.
layout(set = 0, binding = 1) uniform sampler2D textures[];
// -------------------------------------------------------

// ###### USER INPUT AND LIGHT SOURCES ###################
// Uniform buffer "uboMatricesAndUserInput", containing camera matrices and user input
layout (set = 1, binding = 0) uniform UniformBlock { matrices_and_user_input uboMatricesAndUserInput; };

// "mLightsources" uniform buffer containing all the light source data:
layout(set = 1, binding = 1) uniform LightsourceData
{
	// x,y ... ambient light sources start and end indices; z,w ... directional light sources start and end indices
	uvec4 mRangesAmbientDirectional;
	// x,y ... point light sources start and end indices; z,w ... spot light sources start and end indices
	uvec4 mRangesPointSpot;
	// Contains all the data of all the active light sources
	LightsourceGpuData mLightData[MAX_NUMBER_OF_LIGHTSOURCES];
} uboLights;
// -------------------------------------------------------

// ###### RAY PAYLOAD TO traceRayEXT #####################
// Incoming payload as defined in the ray generation shader:
layout(location = 0) rayPayloadInEXT vec3 hitMissColor;
// -------------------------------------------------------

// ###### HIT ATTRIBUTES #################################
hitAttributeEXT vec3 attribs;
// -------------------------------------------------------

// ###### HIT ATTRIBUTES #################################
layout(set = 4, binding = 0) uniform usamplerBuffer uIndexBuffers[];
layout(set = 4, binding = 1) uniform samplerBuffer uNormalBuffers[];

// ###### CLOSEST HIT SHADER MAIN ########################
void main()
{
	// We get the barycentric coordinates from the attributes of the found hit:
    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	// Read the custom index (that we have stored in the GeometryInstance a.k.a. reference from TLAS->BLAS) of the hit geometry:
	const int instanceIndex = nonuniformEXT(gl_InstanceCustomIndexEXT);

	// Set the barycentrics as color:
	hitMissColor = barycentrics;

	// TODO Bonus Task 2 RTX ON: Use the instanceIndex to read the indices of the hit triangle from uIndexBuffers!
	const ivec3 indices = ivec3(texelFetch(uIndexBuffers[instanceIndex], gl_PrimitiveID).rgb);

	//
	// TODO Bonus Task 2 RTX ON: Use the indices to read all the normals of the hit triangles from uNormalBuffers, and
	//                           then, use the barycentric coordinates calculate the correct normal for the hit position!
	//                           Assign the computed normal to the resulting reflection color, i.e. to hitMissColor;
	//
}
// -------------------------------------------------------

