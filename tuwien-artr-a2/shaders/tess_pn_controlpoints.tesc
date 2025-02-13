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

// projects the center point onto the plane defined by averages of the points and planePoints 
vec3 compute_center_point(vec3 P1, vec3 P2, vec3 P3, vec3 b021, vec3 b012, vec3 b102, vec3 b201, vec3 b210, vec3 b120)
{
	vec3 V = (P1 + P2 + P3) / 3.0f, E = (b021 + b012 + b102 + b201 + b210 + b120) / 6.0;
	vec3 VE = E - V;
	return VE/2 + E;
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

bool fustrumCheckX(vec4 P)
{
	return !(P.y <= (P.w + 1) && P.y >= -(P.w + 1) && P.z <= (P.w + 1) && P.z >= -(P.w + 1));
}

bool fustrumCheckY(vec4 P)
{
	return !(P.x <= (P.w + 1) && P.x >= -(P.w + 1) && P.z <= (P.w + 1) && P.z >= -(P.w + 1));
}

bool fustrumCheckZ(vec4 P)
{
	return !(P.x <= (P.w + 1) && P.x >= -(P.w + 1) && P.y <= (P.w + 1) && P.y >= -(P.w + 1));
		
}

bool fustrumCheck(vec4 P[3], int x, int i)
{
	return 
		(P[i][x] <= (P[i].w + 1) && P[i][x] >= -(P[i].w + 1))  ||
		(P[i][x] <= -P[i].w && P[(i+1)%3][x] >=	 P[(i+1)%3].w) ||
		(P[i][x] >=  P[i].w && P[(i+1)%3][x] <= -P[(i+1)%3].w) ||
		(P[i][x] <= -P[i].w && P[(i+2)%3][x] >=  P[(i+2)%3].w) ||
		(P[i][x] >=  P[i].w && P[(i+2)%3][x] <= -P[(i+2)%3].w);
}

bool viewFustrumCheck(vec3 P1, vec3 P2, vec3 P3, mat4 projMatrix) 
{
	vec4 P[3];

	P[0] = projMatrix * vec4(P1, 1);
	P[1] = projMatrix * vec4(P2, 1);
	P[2] = projMatrix * vec4(P3, 1);

	bool isInsideFustrum = false;

	for (int i = 0; i < 3; ++i)
	{

		if (!isInsideFustrum){
			isInsideFustrum = fustrumCheck(P, 0, i) && 
				fustrumCheck(P, 1, i) && 
				fustrumCheck(P, 2, i);
		}

	}

	return isInsideFustrum;
}

// Only taking the outer three triangles created by the bezier control points
// If more triangles needed, I will add the midpoint, and check further normals.
bool backfaceCheck(vec3 p0, vec3 p1, vec3 p2, vec3 p3, vec3 p4, vec3 p5, vec3 p6, vec3 p7, vec3 p8)
{
	bool frontFacing = false;

	// b300 p0,		b030 p1,	b003 p2,	b021 p3,	b012 p4,	b102 p5,	b201 p6,	b210 p7,	b120 p8

	const vec3 p[4][3] = {{p0, p2, p1}, {p0, p6, p7}, {p8, p3, p1}, {p5, p2, p4}};

	for (int i = 0; i < 4; ++i)
	{
		if (!frontFacing && dot(p[i][0], cross(p[i][1]-p[i][0], p[i][2]-p[i][0])) > 0)
		{
			frontFacing = true;
		}
	}
	// After checking these triangles, most, if not all, of the false positives seem to have gone!
	return frontFacing;
}

void adaptiveTessellationModeDistanceBased(vec3 p0, vec3 p1, vec3 p2, float maxTessLevel) 
{
	const int MIN_TESS_LEVEL = 1;
    const float MAX_TESS_LEVEL = maxTessLevel;

    const float MIN_DISTANCE = 16;
    const float MAX_DISTANCE = 64;

	vec3 P0 = (p0+p1)/2;
	vec3 P1 = (p1+p2)/2;
	vec3 P2 = (p2+p0)/2;

    float distance0 = clamp((abs(P0.z)-MIN_DISTANCE) / (MAX_DISTANCE-MIN_DISTANCE), 0.0, 1.0);
    float distance1 = clamp((abs(P1.z)-MIN_DISTANCE) / (MAX_DISTANCE-MIN_DISTANCE), 0.0, 1.0);
    float distance2 = clamp((abs(P2.z)-MIN_DISTANCE) / (MAX_DISTANCE-MIN_DISTANCE), 0.0, 1.0);
	 
	float tessLevel0 = mix( MAX_TESS_LEVEL, MIN_TESS_LEVEL, distance0 );
    float tessLevel1 = mix( MAX_TESS_LEVEL, MIN_TESS_LEVEL, distance1 );
    float tessLevel2 = mix( MAX_TESS_LEVEL, MIN_TESS_LEVEL, distance2 );

	// Getting the levels in the correct order took some guess work
	gl_TessLevelOuter[0] = tessLevel1;
    gl_TessLevelOuter[1] = tessLevel2;
    gl_TessLevelOuter[2] = tessLevel0;

	gl_TessLevelInner[0] = (tessLevel0 + tessLevel1 + tessLevel2) / 3;
}

void adaptiveTessellationModeAngles(vec3 p0, vec3 p1, vec3 p2, float maxTessLevel) 
{
	const int MIN_TESS_LEVEL = 1;
    const float MAX_TESS_LEVEL = maxTessLevel;

	vec3 P0 = (p0+p1)/2;
	vec3 P1 = (p1+p2)/2;
	vec3 P2 = (p2+p0)/2;

	vec3 normal = vec3(0.0,1.0,0.0);

	float angle0 = clamp(acos(dot(normal, P0)), 0.0, 1.0);
    float angle1 = clamp(acos(dot(normal, P1)), 0.0, 1.0);
    float angle2 = clamp(acos(dot(normal, P2)), 0.0, 1.0);


	float tessLevel0 = mix( MAX_TESS_LEVEL, MIN_TESS_LEVEL, angle0 );
	float tessLevel1 = mix( MAX_TESS_LEVEL, MIN_TESS_LEVEL, angle1 );
	float tessLevel2 = mix( MAX_TESS_LEVEL, MIN_TESS_LEVEL, angle2 );
	
	gl_TessLevelOuter[0] = tessLevel1;
	gl_TessLevelOuter[1] = tessLevel2;
	gl_TessLevelOuter[2] = tessLevel0;

	gl_TessLevelInner[0] = (tessLevel0 + tessLevel1 + tessLevel2) / 3;
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
		
		// TODO Task 5: Implement one or multiple adaptive tessellation modes!
		//

		// The mViewToDebugviewMatrix transforms from the current view space to the debug view space!
		// Make use of this transformation to debug your code for tasks 3, 4 and 5!
		// If the "Debug camera" is deactivated, mViewToDebugviewMatrix is set to the identity matrix => no effect.
		// If the "Debug camera" is activated, mViewToDebugviewMatrix transforms from the main camera's
		//                                     view space into the debug camera's view space.
		//
		// Hint: Use positionVS[1-3] for culling and LOD computations!

		bool debug = true;

		vec4 positionVS1 = vec4(tc_in[0].positionVS, 1.0);
		vec4 positionVS2 = vec4(tc_in[1].positionVS, 1.0);
		vec4 positionVS3 = vec4(tc_in[2].positionVS, 1.0);

		if (debug)
		{
			positionVS1 = uboMatricesAndUserInput.mViewToDebugviewMatrix * vec4(tc_in[0].positionVS, 1.0);
			positionVS2 = uboMatricesAndUserInput.mViewToDebugviewMatrix * vec4(tc_in[1].positionVS, 1.0);
			positionVS3 = uboMatricesAndUserInput.mViewToDebugviewMatrix * vec4(tc_in[2].positionVS, 1.0);
		}

		bool performTessellation = true;

		// TODO Task 3: Perform view frustum culling!
		//
		if (uboMatricesAndUserInput.mFrustumCullingBeforeTess) 
			performTessellation = viewFustrumCheck(positionVS1.xyz, positionVS2.xyz, positionVS3.xyz, uboMatricesAndUserInput.mProjMatrix);

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

		//
		// TODO Task 2: Instead of the following, calculate displaced control points
		//              according to the PN-Triangles technique!
		//

		// Using the helper functions
		vec3 b300 = P1;
		vec3 b030 = P2;
		vec3 b003 = P3;
		vec3 b021 = project_point_to_plane(m021, b030, N2);
		vec3 b012 = project_point_to_plane(m012, b003, N3);
		vec3 b102 = project_point_to_plane(m102, b003, N3);
		vec3 b201 = project_point_to_plane(m201, b300, N1);
		vec3 b210 = project_point_to_plane(m210, b300, N1);
		vec3 b120 = project_point_to_plane(m120, b030, N2);
		vec3 b111 = compute_center_point(P1, P2, P3, b021, b012, b102, b201, b210, b120);

		// Compute Bezier control points for normals:
		vec3 m200 = interpolate2(2, 0, 0,   N1, N2, N3);
		vec3 m020 = interpolate2(0, 2, 0,   N1, N2, N3);
		vec3 m002 = interpolate2(0, 0, 2,   N1, N2, N3);
		vec3 m011 = interpolate2(0, 1, 1,   N1, N2, N3);
		vec3 m101 = interpolate2(1, 0, 1,   N1, N2, N3);
		vec3 m110 = interpolate2(1, 1, 0,   N1, N2, N3);

		// The implimentation by Vlachos et al
		vec3 n200 = N1;
		vec3 n020 = N2;
		vec3 n002 = N3;
		float v12 = 2. * (dot(P2 - P1, N1 + N2) / dot(P2 - P1, P2 - P1));
		float v23 = 2. * (dot(P3 - P2, N2 + N3) / dot(P3 - P2, P3 - P2));
		float v31 = 2. * (dot(P1 - P3, N3 + N1) / dot(P1 - P3, P1 - P3));
		vec3 n110 = normalize(N1 + N2 - v12 * (P2 - P1));
		vec3 n011 = normalize(N2 + N3 - v23 * (P3 - P2));
		vec3 n101 = normalize(N3 + N1 - v31 * (P1 - P3));

		// Quick transformed bezier control points
		mat4 dvmMatrix = uboMatricesAndUserInput.mViewToDebugviewMatrix * uboMatricesAndUserInput.mViewMatrix * pushConstants.mModelMatrix;
		vec3 p[] = vec3[](b300, b030, b003, b021, b012, b102, b201, b210, b120, b111);
		vec3 n[] = vec3[](N1, N2, N3);

		for (int i = 0; i < 10; ++i)
		{
			p[i] = (dvmMatrix * vec4(p[i], 1.0)).xyz;
		}

		for (int i = 0; i < 3; ++i)
		{
			n[i] = normalize((dvmMatrix * vec4(n[i], 1.0)).xyz);
		}

		// TODO Task 4: Perform backface culling!
		//

		if(performTessellation && uboMatricesAndUserInput.mBackfaceCullingBeforeTess) 
			performTessellation = backfaceCheck(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);

		// --- COMPUTE TESSELLATION LEVELS ---

		// TODO Task 5: Implement one or multiple adaptive tessellation modes!
		//

		const int matIndex = pushConstants.mMaterialIndex;
		// A mesh will either be tessellated if a flag is set in its material data (will be the case for terrain),
		// or it can also be forced through the push constants of the current draw call:
		const float doTessellateMesh = max(materialsBuffer.materials[matIndex].mCustomData[0], pushConstants.mEnforceTessellation ? 1.0 : 0.0);
		// => If doTessellateMesh is 0, it shall not be tessellated;
		//    If doTessellateMesh is 1, apply the user specified tessellation level.
		const float userDefinedTessLevel = uboMatricesAndUserInput.mUserInput[0];
		const float tessLevel =
			1.0 + // A tess level of 0.0 would mean that the patch is culled. Generally, we want to render patches => therefore, 1.0.
			(userDefinedTessLevel - 1.0) * doTessellateMesh; // The patches to be tessellated might get higher tess levels applied.

		// Set the tessellation levels
		switch (uboMatricesAndUserInput.mAdaptiveTessellationMode)
		{
			case 1:
				adaptiveTessellationModeDistanceBased(p[0], p[1], p[2], tessLevel);
				break;
			case 2:
				adaptiveTessellationModeAngles(p[0], p[1], p[2], tessLevel);
				break;
			case 0:
			default:
				gl_TessLevelOuter[0] = tessLevel;
				gl_TessLevelOuter[1] = tessLevel;
				gl_TessLevelOuter[2] = tessLevel;
				gl_TessLevelInner[0] = tessLevel;
				break;
		}


		if (performTessellation)
		{
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
		
			// Store the calculated values inside the patch out data structure:
			patch_data.n200 = normalize(n200);
			patch_data.n020 = normalize(n020);
			patch_data.n002 = normalize(n002);
			patch_data.n011 = normalize(n011);
			patch_data.n101 = normalize(n101);
			patch_data.n110 = normalize(n110);
		}
		else 
		{
			gl_TessLevelOuter[0] = 0;
		}

	}
}
// -------------------------------------------------------

