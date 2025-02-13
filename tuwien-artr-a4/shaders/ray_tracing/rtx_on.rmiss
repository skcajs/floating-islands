#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable

// ###### RAY PAYLOAD FROM rtx_on.rchit ##################
layout(location = 0) rayPayloadInEXT vec3 hitMissColor;
// -------------------------------------------------------

// ###### MISS SHADER MAIN ###############################
void main()
{
    // Just set a solid color if the ray didn't hit anything:
    hitMissColor = vec3(118.0/255.0, 185.0/255.0, 0.0);

	//
	// TODO Bonus Task 4 RTX ON: Do not set a solid color but instead, the same color value that sky_gradient.frag would set!
	//
}
// -------------------------------------------------------

