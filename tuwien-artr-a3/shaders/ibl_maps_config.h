#ifndef IBL_MAPS_CONFIG_H
#define IBL_MAPS_CONFIG 1

// the irradiance map can be quite small
#define IRRADIANCE_MAP_WIDTH		360
#define IRRADIANCE_MAP_HEIGHT		180

// the prefiltered environment map should be large enough to allow for a couple of mip levels (aka. different roughness values)
// the following is just enough for 11 levels
#define PREFILTERED_ENV_MAP_WIDTH	1600
#define PREFILTERED_ENV_MAP_HEIGHT	800

// BRDF lookup table
#define BRDF_LUT_WIDTH				512
#define BRDF_LUT_HEIGHT				512

#define ALL_IBL_MAPS_FORMAT			vk::Format::eR16G16B16A16Sfloat
#define ALL_IBL_MAPS_SHADERFORMAT	rgba16f

// performance (vs) quality options
//#define IRRADIANCE_MAP_NUM_SAMPLES	1024*16	// on weaker GPUs this "crashes" (i.e. the compute shader takes too long and gets killed by Windows TDR (Timeout Detection and Recovery) after 2 seconds)
#define IRRADIANCE_MAP_NUM_SAMPLES		1024*8	// this is enough anyway (and does not crash)
#define PREFILTERED_ENV_MAP_NUM_SAMPLES	1024
#define BRDF_LUT_NUM_SAMPLES			1024

#endif

