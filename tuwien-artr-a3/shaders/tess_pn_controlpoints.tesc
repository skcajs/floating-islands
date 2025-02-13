#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#include "shader_structures.glsl"
// -------------------------------------------------------


layout(vertices = 3) out;

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

// ###### PIPELINE INPUT DATA ############################
// Unique push constants per draw call (You can think of
// these like single uniforms in OpenGL):
layout(push_constant) uniform PushConstantsBlock { PushConstants pushConstants; };

// Uniform buffer "uboMatricesAndUserInput", containing camera matrices and user input
layout (set = 1, binding = 0) uniform UniformBlock { matrices_and_user_input uboMatricesAndUserInput; };
// -------------------------------------------------------

// ###### TESC INPUT AND OUPUT ###########################
layout (location = 0) in VertexData
{
	vec3 positionOS;    // vertex position in object space
	vec3 positionVS;    // vertex position in view space
	vec2 texCoords;     // texture coordinates
	vec3 normalOS;      // normal in object space
	vec3 tangentOS;     // tangent in object space
	vec3 bitangentOS;   // bitangent in object space
} tc_in[];

layout (location = 0) out TescTeseData
{
	vec2 texCoords;     // texture coordinates
	vec3 tangentOS;     // tangent in object space
	vec3 bitangentOS;   // bitangent in object space
} tc_out[];

// Path data passed on to tessellation evaluation shader:
layout (location = 4) patch out PatchData patch_data;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################

// projects the point onto the plane defined by planePoint and planeNormal
vec3 project_point_to_plane(vec3 point, vec3 planePoint, vec3 planeNormal)
{
	vec3 v = point - planePoint;
	float len = dot(v, planeNormal);
	vec3 d = len * planeNormal;
	return point - d;
}

// calculates the reflection vector of an incident vector around a plane normal vector
vec3 mirror_vec_at_plane(vec3 vector, vec3 planeNormal)
{
	planeNormal = normalize(planeNormal);
	float len = dot(vector, planeNormal);
	vec3 d = (2 * len) * planeNormal;
	return vector - d;
}

// normalizes the interpolation by a factor of 2
// the sum of f1, f2 and f3 must be exactly 2 
vec3 interpolate2(float f1, float f2, float f3, vec3 P1, vec3 P2, vec3 P3)
{
	return (f1 * P1 + f2 * P2 + f3 * P3) / 2;
}

// normalizes the interpolation by a factor of 3
// the sum of f1, f2 and f3 must be exactly 3 
vec3 interpolate3(float f1, float f2, float f3, vec3 P1, vec3 P2, vec3 P3)
{
	return (f1 * P1 + f2 * P2 + f3 * P3) / 3;
}

// -------------------------------------------------------

// ###### TESSELLATION CONTROL SHADER MAIN ###############
void main()
{
	if (gl_InvocationID < 3)
	{
		tc_out[gl_InvocationID].texCoords = tc_in[gl_InvocationID].texCoords;
		tc_out[gl_InvocationID].tangentOS = tc_in[gl_InvocationID].tangentOS;
		tc_out[gl_InvocationID].bitangentOS = tc_in[gl_InvocationID].bitangentOS;
	}

	if (gl_InvocationID == 0)
	{
		const int matIndex = pushConstants.mMaterialIndex;
		// A mesh will either be tessellated if a flag is set in its material data (will be the case for terrain),
		// or it can also be forced through the push constants of the current draw call:
		const float doTessellateMesh = materialsBuffer.materials[matIndex].mCustomData[0];
		// => If doTessellateMesh is 0, it shall not be tessellated;
		//    If doTessellateMesh is 1, apply the user specified tessellation level.
		const float userDefinedTessLevel = uboMatricesAndUserInput.mUserInput[0];
		const float tessLevel =
			1.0 + // A tess level of 0.0 would mean that the patch is culled. Generally, we want to render patches => therefore, 1.0.
			(userDefinedTessLevel - 1.0) * doTessellateMesh; // The patches to be tessellated might get higher tess levels applied.

		vec4 positionVS1 = vec4(tc_in[0].positionVS, 1.0);
		vec4 positionVS2 = vec4(tc_in[1].positionVS, 1.0);
		vec4 positionVS3 = vec4(tc_in[2].positionVS, 1.0);

		// Set the tessellation levels
		gl_TessLevelOuter[0] = tessLevel;
		gl_TessLevelOuter[1] = tessLevel;
		gl_TessLevelOuter[2] = tessLevel;
		gl_TessLevelInner[0] = tessLevel;

		// --- COMPUTE PATCH DATA ---
		// E.g. Bezier triangle coefficients are passed per-patch to the tessellation evaluation
		// shader. Therefore, we store them in the `patch out PatchData` data structure.
		// (Data which is to be passed per vertex is stored in the `out TescTeseData` data structure.)

		vec3 P1 = tc_in[0].positionOS;
		vec3 P2 = tc_in[1].positionOS;
		vec3 P3 = tc_in[2].positionOS;
		vec3 N1 = tc_in[0].normalOS;
		vec3 N2 = tc_in[1].normalOS;
		vec3 N3 = tc_in[2].normalOS;

		// Distribute points evenly within the triangle.
		// Use linear interpolation between the 3 vertices.
		vec3 m300 = interpolate3(3, 0, 0,   P1, P2, P3);
		vec3 m030 = interpolate3(0, 3, 0,   P1, P2, P3);
		vec3 m003 = interpolate3(0, 0, 3,   P1, P2, P3);
		vec3 m021 = interpolate3(0, 2, 1,   P1, P2, P3);
		vec3 m012 = interpolate3(0, 1, 2,   P1, P2, P3);
		vec3 m201 = interpolate3(2, 0, 1,   P1, P2, P3);
		vec3 m102 = interpolate3(1, 0, 2,   P1, P2, P3);
		vec3 m210 = interpolate3(2, 1, 0,   P1, P2, P3);
		vec3 m120 = interpolate3(1, 2, 0,   P1, P2, P3);
		vec3 m111 = interpolate3(1, 1, 1,   P1, P2, P3);

		// The 3 corners are not displaced:
		vec3 b300 = m300;
		vec3 b030 = m030;
		vec3 b003 = m003;
		// Each intermediate point is projected onto the plane defined by the NEAREST vertex and its normal:
		vec3 b021 = project_point_to_plane(m021, b030, N2);
		vec3 b012 = project_point_to_plane(m012, b003, N3);
		vec3 b102 = project_point_to_plane(m102, b003, N3);
		vec3 b201 = project_point_to_plane(m201, b300, N1);
		vec3 b210 = project_point_to_plane(m210, b300, N1);
		vec3 b120 = project_point_to_plane(m120, b030, N2);
		// handle the center
		vec3 E = (b021 + b012 + b102 + b201 + b210 + b120) / 6;
		vec3 V = m111;
		vec3 b111 = E + (E - V) / 2;

		// Store the calculated values inside the patch out data structure:
		patch_data.b030 = b030;
		patch_data.b003 = b003;
		patch_data.b300 = b300;
		patch_data.b021 = b021;
		patch_data.b012 = b012;
		patch_data.b102 = b102;
		patch_data.b201 = b201;
		patch_data.b210 = b210;
		patch_data.b120 = b120;
		patch_data.b111 = b111;

		// Compute Bezier control points for normals:
		vec3 m200 = interpolate2(2, 0, 0,   N1, N2, N3);
		vec3 m020 = interpolate2(0, 2, 0,   N1, N2, N3);
		vec3 m002 = interpolate2(0, 0, 2,   N1, N2, N3);
		vec3 m011 = interpolate2(0, 1, 1,   N1, N2, N3);
		vec3 m101 = interpolate2(1, 0, 1,   N1, N2, N3);
		vec3 m110 = interpolate2(1, 1, 0,   N1, N2, N3);

		// The 3 corners keep their normals:
		vec3 n200 = m200;
		vec3 n020 = m020;
		vec3 n002 = m002;
		// The intermediate normals are mirrored:
		vec3 n011 = mirror_vec_at_plane(m011, P3 - P2);
		vec3 n101 = mirror_vec_at_plane(m101, P3 - P1);
		vec3 n110 = mirror_vec_at_plane(m110, P2 - P1);

		// Store the calculated values inside the patch out data structure:
		patch_data.n200 = normalize(n200);
		patch_data.n020 = normalize(n020);
		patch_data.n002 = normalize(n002);
		patch_data.n011 = normalize(n011);
		patch_data.n101 = normalize(n101);
		patch_data.n110 = normalize(n110);

	}
}
// -------------------------------------------------------

