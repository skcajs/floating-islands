//? #version 460
// above line is just for the VS GLSL language integration plugin

#ifndef SPHERE_MAPPING_GLSL
#define SPHERE_MAPPING_GLSL 1

// utility functions to convert directions to/from equirectangular mapping uv coordinates
// see http://paulbourke.net/geometry/transformationprojection/

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// v has to be in world space, and normalized!
vec2 sphere_map_direction_to_uv(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= vec2(0.5 / M_PI, 1.0 / M_PI);
    uv += 0.5;
    return uv;
}

// returned direction is in world space, and normalized
vec3 sphere_map_uv_to_direction(vec2 uv)
{
	uv -= 0.5;
	uv *= vec2(2.0 * M_PI, M_PI);
	float theta = uv.x;
	//float phi = acos(sin(uv.y)); // TODO: can we simplify this?
	//return vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
	float cosPhi = sin(uv.y);
	float sinPhi = sqrt(1.0f - cosPhi*cosPhi);
	return vec3(sinPhi * cos(theta), cosPhi, sinPhi * sin(theta));
}

#endif

