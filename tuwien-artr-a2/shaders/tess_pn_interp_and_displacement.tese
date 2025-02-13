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

	float w = gl_TessCoord.x;
	float u = gl_TessCoord.y;
	float v = gl_TessCoord.z;

	// 
	// TODO Task 2: Implement cubic Bezier triangle interpolation for the positions!
	// 

	return b300 * w * w * w 
	+ b030 * u * u * u  
	+ b003 * v * v * v 
	+ b210 * 3.0 * w * w * u 
	+ b120 * 3.0 * w * u * u 
	+ b201 * 3.0 * w * w * v 
	+ b021 * 3.0 * u * u * v 
	+ b102 * 3.0 * w * v * v 
	+ b012 * 3.0 * u * v * v 
	+ b111 * 6.0 * w * u * v;
	 
}

vec3 calc_normal()
{
	vec3 n200 = patch_data.n200;
	vec3 n020 = patch_data.n020;
	vec3 n002 = patch_data.n002;
	vec3 n011 = patch_data.n011;
	vec3 n101 = patch_data.n101;
	vec3 n110 = patch_data.n110;

	float w = gl_TessCoord.x;
	float u = gl_TessCoord.y;
	float v = gl_TessCoord.z;

	// 
	// TODO Task 2: Implement quadratic Bezier triangle interpolation for the normals!
	// 

	vec3 n = n200 * w * w 
	+ n020 * u * u 
	+ n002 * v * v 
	+ n110 * w * u 
	+ n011 * u * v 
	+ n101 * w * v;

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
	// 
	// TODO Task 6: Implement displacement mapping and take the value of displacementStrength into account!
	// 
	float displacementStrength = userDefinedDisplacementStrength * meshSpecificDisplacementStrength;

	// - 0.5 so that the terrain stays level.
	float height = sample_from_height_texture(vertexTexCoord).x - 0.5;

	vec4 displacedVertexPositionOS = vertexPositionOS + vec4(normalize(vertexNormalOS) * (height * displacementStrength), 0.0);

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

