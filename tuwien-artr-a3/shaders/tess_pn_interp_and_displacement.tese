#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#include "shader_structures.glsl"
// -------------------------------------------------------

layout(triangles, equal_spacing, cw) in;

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
layout (location = 0) in TescTeseData
{
	vec2 texCoords;
	vec3 tangentOS;
	vec3 bitangentOS;
} tc_in[];

// Patch data incoming from tesc
layout (location = 4) patch in PatchData patch_data;

// Interpolated data tese -> frag
layout (location = 0) out VertexData
{
	vec3 positionOS;
	vec3 positionVS;
	vec2 texCoords;
	vec3 normalOS;
	vec3 tangentOS;
	vec3 bitangentOS;
} te_out;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################
vec4 sample_from_height_texture(vec2 texCoords)
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mHeightTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mHeightTexOffsetTiling;
	vec2 tc = texCoords * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], tc);
}

vec4 sample_from_height_texture_lod(vec2 texCoords, float lod)
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mHeightTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mHeightTexOffsetTiling;
	vec2 tc = texCoords * offsetTiling.zw + offsetTiling.xy;
	return textureLod(textures[texIndex], tc, lod);
}

vec3 calc_position()
{
	vec3 b300 = patch_data.b300;
	vec3 b030 = patch_data.b030;
	vec3 b003 = patch_data.b003;
	vec3 b210 = patch_data.b210;
	vec3 b120 = patch_data.b120;
	vec3 b021 = patch_data.b021;
	vec3 b012 = patch_data.b012;
	vec3 b102 = patch_data.b102;
	vec3 b201 = patch_data.b201;
	vec3 b111 = patch_data.b111;

	float u = gl_TessCoord.x;
	float v = gl_TessCoord.y;
	float w = gl_TessCoord.z;

	if (uboMatricesAndUserInput.mUserInput.z == 0.0)
	{
		return b300 * u + b030 * v + b003 * w;
	}
	return b300 * (u * u * u)
	     + b030 * (v * v * v)
	     + b003 * (w * w * w)
	     + b210 * 3 * (u * u) * v 
	     + b120 * 3 * u  * (v * v)
	     + b201 * 3 * (u * u) * w
	     + b021 * 3 * (v * v) * w 
	     + b102 * 3 * u  * (w * w) 
	     + b012 * 3 * v  * (w * w)
	     + b111 * 6 * u  * v  * w;
}

vec3 calc_normal()
{
	vec3 n200 = patch_data.n200;
	vec3 n020 = patch_data.n020;
	vec3 n002 = patch_data.n002;
	vec3 n011 = patch_data.n011;
	vec3 n101 = patch_data.n101;
	vec3 n110 = patch_data.n110;

	float u = gl_TessCoord.x;
	float v = gl_TessCoord.y;
	float w = gl_TessCoord.z;

	vec3 n = n200 * (u * u)
	       + n020 * (v * v)
	       + n002 * (w * w)
	       + n110 * u * v
	       + n011 * v * w
	       + n101 * u * w;
	if (uboMatricesAndUserInput.mUserInput.z == 0.0)
	{
		n = n200 * u + n020 * v + n002 * w;
	}

	return normalize(n);
}
// -------------------------------------------------------

// ##### TESSELLATION EVALUATION SHADER MAIN #############
void main()
{
	vec4 vertexPositionOS       = vec4(calc_position(), 1.0);
	vec3 vertexNormalOS         = calc_normal();
	vec2 vertexTexCoord         =           gl_TessCoord.x * tc_in[0].texCoords   + gl_TessCoord.y * tc_in[1].texCoords   + gl_TessCoord.z * tc_in[2].texCoords;
	vec3 vertexTangentOS        = normalize(gl_TessCoord.x * tc_in[0].tangentOS   + gl_TessCoord.y * tc_in[1].tangentOS   + gl_TessCoord.z * tc_in[2].tangentOS);
	vec3 vertexBitangentOS      = normalize(gl_TessCoord.x * tc_in[0].bitangentOS + gl_TessCoord.y * tc_in[1].bitangentOS + gl_TessCoord.z * tc_in[2].bitangentOS);

	int matIndex = pushConstants.mMaterialIndex;
	float meshSpecificDisplacementStrength = materialsBuffer.materials[matIndex].mCustomData[1];
	float userDefinedDisplacementStrength = uboMatricesAndUserInput.mUserInput[1];
	float displacementStrength = userDefinedDisplacementStrength * meshSpecificDisplacementStrength;

	float displacement = sample_from_height_texture(vertexTexCoord).r - 0.5;


	vec4 displacedVertexPositionOS = vertexPositionOS + vec4(vertexNormalOS * displacement * displacementStrength, 0.0);

	mat4 vmMatrix = uboMatricesAndUserInput.mViewMatrix * pushConstants.mModelMatrix;
	mat4 pMatrix = uboMatricesAndUserInput.mProjMatrix;
	vec4 vertexVS = vmMatrix * displacedVertexPositionOS;
	vec4 vertexCS =  pMatrix * vertexVS;

	te_out.positionOS  = displacedVertexPositionOS.xyz;
	te_out.positionVS  = vertexVS.xyz;
	te_out.texCoords   = vertexTexCoord;
	te_out.normalOS    = vertexNormalOS;
	te_out.tangentOS   = vertexTangentOS;
	te_out.bitangentOS = vertexBitangentOS;

	gl_Position = vertexCS;
}
// ----------------------------------------------

