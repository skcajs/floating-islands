//? #version 460
// above line is just for the VS GLSL language integration plugin

#ifndef CUSTOM_PACKING_GLSL
#define CUSTOM_PACKING_GLSL 1

// Pack/unpack material index and texture gradients to/from a 16-bit RGBA uint texture

uvec4 pack_material_and_texture_gradients(uint matId, vec4 ddx_ddy) {
	// we have 4*16 = 64 bits available
	// max material index currently is 22-ish, so 5 bits would be sufficient (-> max 32 materials)
	// leaves 59 bits for the gradients -> 4*14 = 56 (so still 3 bits left if we pack the grads equally)
	// so:
	// material: 8 bits
	// each grad component: 14 bits

	// assumed: grad components are in [-1,1] range
	// bring them to [1,1] range
	// ddx_ddy = ddx_ddy * 0.5 + 0.5;

	// just to be on the safe side, assume gradients can be in [-2,2] range (e.g. with extreme uv coords -1 and 1)
	// bring them to [1,1] range
	ddx_ddy = ddx_ddy * 0.25 + 0.5;

	uvec4 scaled_grads = uvec4(round(clamp(ddx_ddy, 0.0, 1.0) * 16383.0)); // * (2^14 - 1)

	matId = matId & 0xFF;
	scaled_grads = scaled_grads & uvec4(0x3FFF);

	//         0                1                2                3          <- word #
	//  fedcba9876543210 fedcba9876543210 fedcba9876543210 fedcba9876543210	 <- bit #
	// |mmmmmmmmaaaaaaaa|aaaaaabbbbbbbbbb|bbbbcccccccccccc|ccdddddddddddddd|

	uvec4 result;
	result[0] = ((matId           & 0xFF) <<  8) | (scaled_grads[0] >> 6);
	result[1] = ((scaled_grads[0] & 0x3F) << 10) | (scaled_grads[1] >> 4);
	result[2] = ((scaled_grads[1] & 0x0F) << 12) | (scaled_grads[2] >> 2);
	result[3] = ((scaled_grads[2] & 0x03) << 14) |  scaled_grads[3]      ;

	return result;
}

void unpack_material_and_texture_gradients(uvec4 packed, out uint matId, out vec4 ddx_ddy) {
	//         0                1                2                3          <- word #
	//  fedcba9876543210 fedcba9876543210 fedcba9876543210 fedcba9876543210	 <- bit #
	// |mmmmmmmmaaaaaaaa|aaaaaabbbbbbbbbb|bbbbcccccccccccc|ccdddddddddddddd|

	uvec4 scaled_grads = uvec4(0);
	matId = (packed[0] >> 8) & 0xFF;
	scaled_grads[0] = ((packed[0] & 0x0FF) << 6) | ((packed[1] >> 10) & 0x3F);
	scaled_grads[1] = ((packed[1] & 0x3FF) << 4) | ((packed[2] >> 12) & 0x0F);
	scaled_grads[2] = ((packed[2] & 0xFFF) << 2) | ((packed[3] >> 14) & 0x03);
	scaled_grads[3] =   packed[3] & 0x3FFF;

	ddx_ddy = vec4(scaled_grads) / 16383.0;
	//ddx_ddy = ddx_ddy * 2.0 - 1.0;
	ddx_ddy = ddx_ddy * 4.0 - 2.0;
}

#endif

