#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#include "lightsource_limits.h"
#include "shader_structures.glsl"
#include "sphere_mapping.glsl"
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

layout(push_constant) uniform PushConstantsBlock { PushConstants pushConstants; };

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

// 
// TODO Task 1: Create input attachment declarations according to your renderpass configuration!
//              Perform all the illumination here in this shader!
//
// TODO Task 3: For enabling this shader being executed per sample, you'll have to switch 
//              from subpassInput to subpassInputMS, which enables reading of the data 
//              per sample. 
//              Furthermore, you'll have to use subpassLoad(..., gl_SampleID)
//

layout (input_attachment_index = 0, set = 2, binding = 0) uniform subpassInputMS iColor;
layout (input_attachment_index = 1, set = 2, binding = 1) uniform subpassInputMS iDepth;
layout (input_attachment_index = 2, set = 2, binding = 2) uniform subpassInputMS iNormal;
layout (input_attachment_index = 3, set = 2, binding = 3) uniform subpassInputMS iAmbient;
layout (input_attachment_index = 4, set = 2, binding = 4) uniform subpassInputMS iEmissive;
layout (input_attachment_index = 5, set = 2, binding = 5) uniform subpassInputMS iDiffuse;
layout (input_attachment_index = 6, set = 2, binding = 6) uniform subpassInputMS iSpecular;
layout (input_attachment_index = 7, set = 2, binding = 7) uniform subpassInputMS iSrm;


// Image Based Lighting textures
layout(set = 3, binding = 0) uniform sampler2D texIblIrradianceMap;
layout(set = 3, binding = 1) uniform sampler2D texIblPrefilteredEnvironmentMap;
layout(set = 3, binding = 2) uniform sampler2D texIblBrdfLookupTable;
// -------------------------------------------------------

// ###### FRAG OUTPUT ####################################
layout (location = 0) out vec4 oFragColor;
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
    float alpha = roughness*roughness; // It say's alpha = roughness... BUT... this looks so much better! :) 
    float alpha2 = alpha * alpha;
    float n_dot_h2 = n_dot_h * n_dot_h;

    float denominator = (n_dot_h2 * (alpha2 - 1.0) + 1.0);
    denominator = PI * denominator * denominator;

    return alpha2 / denominator;
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
	vec3 F0 = mix(vec3(0.04), albedo, metallic);
	return F0 + (1 - F0) * pow(1 - h_dot_v, 5); // <- fix that!
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
	vec3 f_microfacet = ( F * G * D ) / max(4.0 * n_dot_l * n_dot_v, 0.001) ; // <- fix that!

	// The Cook-Torrance-BRDF also has a diffuse component, which we calculate here:
	vec3 diffuse_part = ((vec3(1.0) - F) * (1.0 - metallic)) * albedo / PI;

	return (diffuse_part + f_microfacet) * n_dot_l;
}

// Calculates the light attenuation dividend for the given attenuation vector.
// @param atten attenuation data
// @param dist  distance
// @param dist2 squared distance
float calc_attenuation(vec4 atten, float dist, float dist2)
{
	return atten[0] + atten[1] * dist + atten[2] * dist2;
}


// Calculate Physically Based Illumination in View-Space
vec3 calc_pbs_illumination_in_vs(vec3 posVS, vec3 normalVS, vec3 albedo, float metallic, float roughness)
{
	vec3 pbsIllumination = vec3(0.0, 0.0, 0.0);

	// TODO Task 6:
	// - Calculate Physically Based Illumination
	// - Hint: This function will probably look almost identical to calc_illumination_in_vs() in blinnphong_and_normal_mapping.frag
	// -       But instead of calling calc_blinn_phong_contribution(), use calc_cook_torrance_contribution() here!

		// Calculate shading in view space since all light parameters are passed to the shader in view space
	vec3 eyePosVS = vec3(0.0, 0.0, 0.0);
	vec3 toEyeNrmVS = normalize(eyePosVS - posVS);

	// Directional lights:
	for (uint i = uboLights.mRangesAmbientDirectional[2]; i < uboLights.mRangesAmbientDirectional[3]; ++i) {
		vec3 toLightDirVS = normalize(-uboLights.mLightData[i].mDirection.xyz);
		vec3 dirLightIntensity = uboLights.mLightData[i].mColor.rgb;
		pbsIllumination += dirLightIntensity * calc_cook_torrance_contribution(toLightDirVS, toEyeNrmVS, normalVS, albedo, metallic, roughness);
	}

	// Point lights:
	for (uint i = uboLights.mRangesPointSpot[0]; i < uboLights.mRangesPointSpot[1]; ++i)
	{
		vec3 lightPosVS = uboLights.mLightData[i].mPosition.xyz;
		vec3 toLight = lightPosVS - posVS;
		float distSq = dot(toLight, toLight);
		float dist = sqrt(distSq);
		vec3 toLightNrm = toLight / dist;

		float atten = calc_attenuation(uboLights.mLightData[i].mAttenuation, dist, distSq);
		vec3 intensity = uboLights.mLightData[i].mColor.rgb / atten;

		pbsIllumination += intensity * calc_cook_torrance_contribution(toLightNrm, toEyeNrmVS, normalVS, albedo, metallic, roughness);
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

		pbsIllumination += intensity * calc_cook_torrance_contribution(toLightNrm, toEyeNrmVS, normalVS, albedo, metallic, roughness);
	}

	return pbsIllumination * uboMatricesAndUserInput.mPbsLightBoost;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



// Calculates the diffuse and specular illumination contribution for the given
// parameters according to the Blinn-Phong lighting model.
// All parameters must be normalized.
vec3 calc_blinn_phong_contribution(vec3 toLight, vec3 toEye, vec3 normal, vec3 diffFactor, vec3 specFactor, float specShininess)
{
	float nDotL = max(0.0, dot(normal, toLight)); // lambertian coefficient
	vec3 h = normalize(toLight + toEye);
	float nDotH = max(0.0, dot(normal, h));
	float specPower = pow(nDotH, specShininess);

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

	// Directional lights:
	for (uint i = uboLights.mRangesAmbientDirectional[2]; i < uboLights.mRangesAmbientDirectional[3]; ++i) {
		vec3 toLightDirVS = normalize(-uboLights.mLightData[i].mDirection.xyz);
		vec3 dirLightIntensity = uboLights.mLightData[i].mColor.rgb;
		diffAndSpec += dirLightIntensity * calc_blinn_phong_contribution(toLightDirVS, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	// Point lights:
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


// ###### FRAGMENT SHADER MAIN #############################
void main()
{

	int matIndex = pushConstants.mMaterialIndex;

	float depth = subpassLoad(iDepth, gl_SampleID).r;

	vec3 ndcPosition = vec3(
        fs_in.texCoords * 2.0 - 1.0,
        depth
    );

	vec4 viewPosition = uboMatricesAndUserInput.mInverseProjMatrix * vec4(ndcPosition, 1.0);
    viewPosition /= viewPosition.w; // Perform perspective divide
    vec3 positionVS = viewPosition.xyz;

	vec3 nsc = subpassLoad(iNormal, gl_SampleID).rgb;
	vec3 normalVS = vec3(sin(nsc.y)*cos(nsc.x), sin(nsc.y)*sin(nsc.x), cos(nsc).y);

//	// Initialize all the colors:
	vec3 ambient = subpassLoad(iAmbient, gl_SampleID).rgb;
	vec3 emissive = subpassLoad(iEmissive, gl_SampleID).rgb;
	vec3 diff = subpassLoad(iDiffuse, gl_SampleID).rgb;
	vec3 spec = subpassLoad(iSpecular, gl_SampleID).rgb;
	float shininess = subpassLoad(iSrm, gl_SampleID).r;
	float roughness = subpassLoad(iSrm, gl_SampleID).g;
	float metallic = subpassLoad(iSrm, gl_SampleID).b;

	// Calculate ambient illumination:
	vec3 ambientIllumination = vec3(0.0, 0.0, 0.0);
	for (uint i = uboLights.mRangesAmbientDirectional[0]; i < uboLights.mRangesAmbientDirectional[1]; ++i) {
		ambientIllumination += uboLights.mLightData[i].mColor.rgb * ambient;
	}

	vec3 illumination = vec3(0.0);

	if (uboMatricesAndUserInput.mPbsEnabled) 
	{
		illumination = calc_pbs_illumination_in_vs(positionVS, normalVS, diff, metallic, roughness);
	} 
	else 
	{
		// Calculate diffuse and specular illumination from all light sources:
		illumination = calc_illumination_in_vs(positionVS, normalVS, diff, spec, shininess);
	}


	// Add all together:
	oFragColor = vec4(ambientIllumination + emissive + illumination, 1.0);

	// TODO Task 1:
	//  - Install input attachments (see uniform subpassInput above!)
	//  - Sample the G-Buffer data written in the G-Buffer pass from them
	//  - Perform lighting in THIS shader and no longer in blinnphong_and_normal_mapping.frag

	// TODO Task 6:
	// If (and only if) uboMatricesAndUserInput.mPbsEnabled is true, then:
	//  - Instead of performing Blinn-Phong lighting, perform Physically Based Shading
	//  - Sample roughness and metallic values from wherever you stored it in the G-Buffers
	//  - Use the provided stub function calc_pbs_illumination_in_vs()
	//  - Hint: You still want to add ambient and emissive illumination to the result, same as with Blinn-Phong

	// TODO Bonus Task 2:
	// If (and only if) uboMatricesAndUserInput.mIblEnabled is true, then:
	//  - Instead of performing Blinn-Phong lighting, perform Image Based Illumination
	//  - Sample roughness and metallic values from wherever you stored it in the G-Buffers
	//  - Use the provided stub function calc_ibl_illumination_in_ws()
	//    Hint: calc_ibl_illumination_in_ws() works in world-space, but your all your vectors are in view-space.
	//          So you need to transform them to world-space before calling the function.
	//  - Hint: You should *not* add ambient illumination to the result.
	//
	//  If you want, you can combine IBL with PBS, but that is not required.
	//  In case you want to try it, just calculate PBS as usual, but replace the ambient illumination by IBL.

//	oFragColor = subpassLoad(iColor).rgba;
}
// -------------------------------------------------------

