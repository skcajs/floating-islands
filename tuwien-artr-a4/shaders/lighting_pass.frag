#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#include "lightsource_limits.h"
#include "shader_structures.glsl"
#include "custom_packing.glsl"
// -------------------------------------------------------

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

// ###### FRAG INPUT #####################################
layout (location = 0) in VertexData
{
	vec2 texCoords;   // texture coordinates
} fs_in;

layout (input_attachment_index = 0, set = 2, binding = 0) uniform subpassInput iDepth;
layout (input_attachment_index = 1, set = 2, binding = 1) uniform subpassInput iUvNrm;
layout (input_attachment_index = 2, set = 2, binding = 2) uniform usubpassInput iMatId;
// -------------------------------------------------------

// ###### FRAG OUTPUT ####################################
layout (location = 0) out vec4 oFragColor;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################
vec4 sample_from_diffuse_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_specular_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_reflection_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mReflectionTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mReflectionTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_extra_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mExtraTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mExtraTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

// the gradients apply to the final uv-coordinates (after offset tiling)
vec4 sample_from_diffuse_texture_with_final_gradient(int matIndex, vec2 uv, vec2 gradX, vec2 gradY)
{
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return textureGrad(textures[texIndex], texCoords, gradX, gradY);
}

// the gradients apply to the final uv-coordinates (after offset tiling)
vec4 sample_from_specular_texture_with_final_gradient(int matIndex, vec2 uv, vec2 gradX, vec2 gradY)
{
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return textureGrad(textures[texIndex], texCoords, gradX, gradY);
}

// the gradients apply to the final uv-coordinates (after offset tiling)
vec4 sample_from_reflection_texture_with_final_gradient(int matIndex, vec2 uv, vec2 gradX, vec2 gradY)
{
	int texIndex = materialsBuffer.materials[matIndex].mReflectionTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mReflectionTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return textureGrad(textures[texIndex], texCoords, gradX, gradY);
}

// the gradients apply to the final uv-coordinates (after offset tiling)
vec4 sample_from_extra_texture_with_final_gradient(int matIndex, vec2 uv, vec2 gradX, vec2 gradY)
{
	int texIndex = materialsBuffer.materials[matIndex].mExtraTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mExtraTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return textureGrad(textures[texIndex], texCoords, gradX, gradY);
}


// Calculates the light attenuation dividend for the given attenuation vector.
// @param atten attenuation data
// @param dist  distance
// @param dist2 squared distance
float calc_attenuation(vec4 atten, float dist, float dist2)
{
	return atten[0] + atten[1] * dist + atten[2] * dist2;
}

// Calculates the diffuse and specular illumination contribution for the given
// parameters according to the Blinn-Phong lighting model.
// All parameters must be normalized.
vec3 calc_blinn_phong_contribution(vec3 toLight, vec3 toEye, vec3 normal, vec3 diffFactor, vec3 specFactor, float specShininess)
{
	float nDotL = max(0.0, dot(normal, toLight)); // lambertian coefficient
	vec3 h = normalize(toLight + toEye);
	float nDotH = max(0.0, dot(normal, h));
	float specPower = (nDotH == 0 && specShininess == 0) ? 1 : pow(nDotH, specShininess);

	vec3 diffuse = diffFactor * nDotL; // component-wise product
	vec3 specular = specFactor * specPower;

	return diffuse + specular;
}

// Calculates the diffuse and specular illumination contribution for all the light sources.
// All calculations are performed in view space
vec3 calc_illumination_in_vs(vec3 posVS, vec3 normalVS, vec3 diff, vec3 spec, float shini)
{
	vec3 diffAndSpec = vec3(0.0, 0.0, 0.0);

	// Calculate shading in view space since all light parameters are passed to the shader in view space
	vec3 eyePosVS = vec3(0.0, 0.0, 0.0);
	vec3 toEyeNrmVS = normalize(eyePosVS - posVS);

	// directional lights
	for (uint i = uboLights.mRangesAmbientDirectional[2]; i < uboLights.mRangesAmbientDirectional[3]; ++i) {
		vec3 toLightDirVS = normalize(-uboLights.mLightData[i].mDirection.xyz);
		vec3 dirLightIntensity = uboLights.mLightData[i].mColor.rgb;
		diffAndSpec += dirLightIntensity * calc_blinn_phong_contribution(toLightDirVS, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	// point lights
	for (uint i = uboLights.mRangesPointSpot[0]; i < uboLights.mRangesPointSpot[1]; ++i)
	{
		vec3 lightPosVS = uboLights.mLightData[i].mPosition.xyz;
		vec3 toLight = lightPosVS - posVS;
		float distSq = dot(toLight, toLight);
		float dist = sqrt(distSq);
		vec3 toLightNrm = toLight / dist;

		float atten = calc_attenuation(uboLights.mLightData[i].mAttenuation, dist, distSq);
		vec3 intensity = uboLights.mLightData[i].mColor.rgb / atten;
		diffAndSpec += intensity * calc_blinn_phong_contribution(toLightNrm, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	// spot lights
	for (uint i = uboLights.mRangesPointSpot[2]; i < uboLights.mRangesPointSpot[3]; ++i)
	{
		vec3 lightPosVS = uboLights.mLightData[i].mPosition.xyz;
		vec3 toLight = lightPosVS - posVS;
		float distSq = dot(toLight, toLight);
		float dist = sqrt(distSq);
		vec3 toLightNrm = toLight / dist;

		float atten = calc_attenuation(uboLights.mLightData[i].mAttenuation, dist, distSq);
		vec3 intensity = uboLights.mLightData[i].mColor.rgb / atten;

		vec3 dirVS = uboLights.mLightData[i].mDirection.xyz;
		float cosOfHalfOuter = uboLights.mLightData[i].mAnglesFalloff[0];
		float cosOfHalfInner = uboLights.mLightData[i].mAnglesFalloff[1];
		float falloff = uboLights.mLightData[i].mAnglesFalloff[2];
		float cosAlpha = dot(-toLightNrm, dirVS);
		float da = cosAlpha - cosOfHalfOuter;
		float fade = cosOfHalfInner - cosOfHalfOuter;
		intensity *= da <= 0.0 ? 0.0 : pow(min(1.0, da / max(0.0001, fade)), falloff);

		diffAndSpec += intensity * calc_blinn_phong_contribution(toLightNrm, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	return diffAndSpec;
}

vec3 reconstruct_vs_position_from_depth(float depth)
{
	// reconstruct position from depth buffer
	vec4 clipSpace = vec4(fs_in.texCoords * 2.0 - 1.0, depth, 1.0);
	vec4 viewSpace = inverse(uboMatricesAndUserInput.mProjMatrix) * clipSpace;
	vec3 positionVS = viewSpace.xyz / viewSpace.w;
	return positionVS;
}
// -------------------------------------------------------
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Physically based shading

const float PI = 3.14159265359;

// Calculate the D-term for PBS
//  The parameter n_dot_h corresponds to cos(Theta) [ or n.m ], and roughness corresponds to alpha in the formulas in the PBS-slides (p.37)
float calc_microsurface_normal_distribution(float roughness, float n_dot_h)
{
	// TODO Task 6:
	//  - Calculate a normal distribution function for the microsurface model
	//  - Hint: The parameter n_dot_h is the dot product of the surface normal n and the halfway vector h.
	//  -       If you choose to implement the Trowbridge-Reitz NDF (slides p.37), note that:
	//  -         n_dot_h is the same as n.m in the formula in the slides as pointed out later on p.40.
	//  -         roughness corresponds to alpha in the formula in the slides

	return 1.0f; // <- fix that!
}

// Calculate the F-term for PBS
//  The parameter h_dot_v corresponds to cos(Theta) in the PBS-slides (p.16).
vec3 calc_fresnel_reflectance(vec3 albedo, float metallic, float h_dot_v)
{
	// TODO Task 6:
	//  - Calculate the Fresnel reflectance
	//  - Hint: The parameter h_dot_v is the dot product of the halfway vector h and the direction to the eye v.
	//  - Hint: You can assume F0 = vec3(0.04) for perfect dielectrics (metallic == 0.0), and F0 = albedo for pure metallics (metallic == 1.0) and blend accordingly.
	//
	//  A note for curious students:
	//    If you wonder why we use h.v for cos(Theta) here, instead of n.v: This is not a mistake!
	//    See the last sentence ("In microfacet models...") in  https://en.wikipedia.org/wiki/Schlick%27s_approximation for why this is the correct way.

	return vec3(0.04); // <- fix that!
}

// Calculate the G-term for PBS
float calc_geometry_term(float roughness, float n_dot_v, float n_dot_l)
{
	// - nothing is required to do for you here ;)
	//   (but feel free to experiment with different geometry terms if you want)

	// Calculate the G-term with a Smith/Schlick-GGX approximation
	float r_plus_1 = roughness + 1.0;
	float k = r_plus_1 * r_plus_1 / 8.0;
	float G_v = (n_dot_v / (n_dot_v * (1.0 - k) + k)); // G-term for view  (occlusion/masking)
	float G_l = (n_dot_l / (n_dot_l * (1.0 - k) + k)); // G-term for light (shadowing)
	float G = G_v * G_l;
	return G;
}

// Calculate the illumination contribution for the given parameters according to the Cook-Torrance lighting model.
// All parameters must be normalized.
vec3 calc_cook_torrance_contribution(vec3 to_light, vec3 to_eye, vec3 normal, vec3 albedo, float metallic, float roughness)
{
	// calculate the halfway vector (halfway between the direction to the light and the direction to the eye)
	vec3 half_vec = normalize(to_light + to_eye);

	// precalculate some dot products
	float n_dot_l = max(0.0, dot(normal, to_light));
	float n_dot_h = max(0.0, dot(normal, half_vec));
	float n_dot_v = max(0.0, dot(normal, to_eye));
	float h_dot_v = max(0.0, dot(half_vec, to_eye));

	// calc D-term (distribution)
	float D = calc_microsurface_normal_distribution(roughness, n_dot_h);

	// calc F-term (Fresnel)
	vec3 F = calc_fresnel_reflectance(albedo, metallic, h_dot_v);

	// calc G-term (geometric)
	float G = calc_geometry_term(roughness, n_dot_v, n_dot_l);

	// TODO Task 6:
	// - Calculate the specular part of the Cook-Torrance contribution
	// - This corresponds to f_microfacet(l,v) in the slides (p.50)
	// - Hint: to avoid a division by zero (if to_light or to_eye is perpendicular to normal), divide by max(...., 0.001) instead.
	vec3 f_microfacet = vec3(0); // <- fix that!

	// The Cook-Torrance-BRDF also has a diffuse component, which we calculate here:
	vec3 diffuse_part = ((vec3(1.0) - F) * (1.0 - metallic)) * albedo / PI;

	return (diffuse_part + f_microfacet) * n_dot_l;
}

// Calculate Physically Based Illumination in View-Space
vec3 calc_pbs_illumination_in_vs(vec3 posVS, vec3 normalVS, vec3 albedo, float metallic, float roughness)
{
	vec3 pbsIllumination = vec3(0.0, 0.0, 0.0);

	// TODO Task 6:
	// - Calculate Physically Based Illumination
	// - Hint: This function will probably look almost identical to calc_illumination_in_vs() in blinnphong_and_normal_mapping.frag
	// -       But instead of calling calc_blinn_phong_contribution(), use calc_cook_torrance_contribution() here!

	return pbsIllumination;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// ###### FRAGMENT SHADER MAIN #############################
void main()
{
	float   depth =     subpassLoad(iDepth).r;
	vec4 uvNormal =     subpassLoad(iUvNrm).rgba;
	uint tmp_umatIndex;
	vec4 tmp_ddx_ddy;
	unpack_material_and_texture_gradients(subpassLoad(iMatId), tmp_umatIndex, tmp_ddx_ddy);
	int matIndex = int(tmp_umatIndex);
	vec2 final_texture_gradient_x = tmp_ddx_ddy.xy;
	vec2 final_texture_gradient_y = tmp_ddx_ddy.zw;

	if (depth == 1) discard;

	// unpack uv and normal
	vec2 uv = uvNormal.rg;
	vec3 normalVS = vec3(cos(uvNormal.z) * cos(uvNormal.w), sin(uvNormal.z) * cos(uvNormal.w), sin(uvNormal.w));

	// reconstruct position from depth buffer
	vec4 clipSpace = vec4(fs_in.texCoords * 2 - 1, depth, 1);
	vec4 viewSpace = inverse(uboMatricesAndUserInput.mProjMatrix) * clipSpace;
	vec3 positionVS = viewSpace.xyz / viewSpace.w;

	vec3  diffTexColor = sample_from_diffuse_texture_with_final_gradient    (matIndex, uv, final_texture_gradient_x, final_texture_gradient_y).rgb;
	float specTexValue = sample_from_specular_texture_with_final_gradient   (matIndex, uv, final_texture_gradient_x, final_texture_gradient_y).r;

	// Initialize all the colors:
	vec3 ambient    = materialsBuffer.materials[matIndex].mAmbientReflectivity.rgb * diffTexColor;
	vec3 emissive   = materialsBuffer.materials[matIndex].mEmissiveColor.rgb;
	vec3 diff       = materialsBuffer.materials[matIndex].mDiffuseReflectivity.rgb * diffTexColor;
	vec3 spec       = materialsBuffer.materials[matIndex].mSpecularReflectivity.rgb * specTexValue;
	float shininess = materialsBuffer.materials[matIndex].mShininess;

	// Calculate ambient illumination:
	vec3 ambientIllumination = vec3(0.0, 0.0, 0.0);
	for (uint i = uboLights.mRangesAmbientDirectional[0]; i < uboLights.mRangesAmbientDirectional[1]; ++i) {
		ambientIllumination += uboLights.mLightData[i].mColor.rgb * ambient;
	}

	// Calculate diffuse and specular illumination from all light sources:
	vec3 diffAndSpecIllumination = calc_illumination_in_vs(positionVS, normalVS, diff, spec, shininess);

	// Add all together:
	oFragColor = vec4(ambientIllumination + emissive + diffAndSpecIllumination, 1.0);
}
// -------------------------------------------------------

