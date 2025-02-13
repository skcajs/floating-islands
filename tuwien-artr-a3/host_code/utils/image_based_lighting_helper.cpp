#include "precompiled_headers.hpp"
#include "image_based_lighting_helper.hpp"
#include "../../shaders/ibl_maps_config.h"

#define IBL_HELPER_TEXPATH "assets/ibl/"
#define IBL_HELPER_TEX_BG  "Newport_Loft_8k.jpg"
#define IBL_HELPER_TEX_HDR "Newport_Loft_Ref.hdr"

IblHelper::IblHelper(avk::queue& aQueue)
	: mQueue{ &aQueue }
	, mSphere{ &aQueue }
{
}

void IblHelper::initialize(std::vector<helpers::data_for_draw_call>& aDrawCalls, helpers::LoadedMaterialsInfo aLoadedMaterials
	, avk::queue* aQueue
)
{
	using namespace avk;

	// create dummy IBL maps, so they can be passed to lighting shaders even though the real maps have not been built yet
	create_dummy_maps();

	mMaterialInfo = aLoadedMaterials;
	//mMaterialInfo.printDebugInfo(); // debug: print names of loaded materials

	// extract the demo object for rendering
	{
		std::vector<helpers::data_for_draw_call> tmpDrawCalls;
		helpers::separate_draw_calls(1, aDrawCalls, tmpDrawCalls);
		std::vector<ExtendedDrawCallData> tmpDrawCallSet;
		for (auto& dc : tmpDrawCalls) {
			ExtendedDrawCallData edc;
			edc.drawCall = std::move(dc);
			edc.hasPbsOverride = false;
			tmpDrawCallSet.push_back(std::move(edc));
		}
		mDrawCallsSets.push_back(std::move(tmpDrawCallSet));
	}

	// create a sphere geometry
	mSphere.set_flags(simple_geometry::flags::all).create_sphere(40, 80);

	// create drawcalls for an array of spheres
	{
		helpers::data_for_draw_call baseDrawCall;
		baseDrawCall.mIndexBuffer      = mSphere.mIndexBuffer;
		baseDrawCall.mPositionsBuffer  = mSphere.mPositionsBuffer;
		baseDrawCall.mTexCoordsBuffer  = mSphere.mTexCoordsBuffer;
		baseDrawCall.mNormalsBuffer    = mSphere.mNormalsBuffer;
		baseDrawCall.mTangentsBuffer   = mSphere.mTangentsBuffer;
		baseDrawCall.mBitangentsBuffer = mSphere.mBitangentsBuffer;
		baseDrawCall.mMaterialIndex = 7; // not used

		std::vector<ExtendedDrawCallData> tmpDrawCallSet;

		int numZ = 5, numY = 5;
		float size = 1.0f, dist = size * 1.25f;
		glm::vec3 baseTranslation = glm::vec3(0, numY * -0.5f * dist, numZ * -0.5f * dist) + glm::vec3(3, 1.5, 0);
		for (int iZ = 0; iZ < numZ; ++iZ) {
			for (int iY = 0; iY < numY; ++iY) {
				glm::mat4 matRot = glm::mat4(1);
				matRot = glm::rotate(glm::radians(-20.f), glm::vec3(0,0,1)) * matRot;
				matRot = glm::rotate(glm::radians( 20.f), glm::vec3(1,0,0)) * matRot;

				ExtendedDrawCallData edc;
				edc.drawCall = baseDrawCall; // copy
				edc.drawCall.mModelMatrix = glm::mat4(1);
				edc.mTransformAfterRotate = glm::translate(baseTranslation + glm::vec3(0, iY * dist, iZ * dist)) * glm::scale(glm::vec3(size * 0.5f)) * matRot;
				edc.hasPbsOverride = true;
				edc.roughness = float(iZ) / float(numZ - 1);
				edc.metallic = float(iY) / float(numY - 1);
				tmpDrawCallSet.push_back(std::move(edc));
			}
		}
		mDrawCallsSets.push_back(std::move(tmpDrawCallSet));
	}

	// init material index we want to use
	{
		bool found = false;
		for (auto i = 0; i < mMaterialInfo.mMaterialNames.size(); ++i) {
			if (mMaterialInfo.mMaterialNames[i] == mDefaultMaterialName) {
				found = true;
				mMaterialIndexToUse = int(i);
			}
		}
		if (!found) LOG_WARNING(std::format("Default material \"{}\" not found", mDefaultMaterialName));
	}
	
	if (!std::filesystem::exists(IBL_HELPER_TEXPATH IBL_HELPER_TEX_BG) || !std::filesystem::exists(IBL_HELPER_TEXPATH IBL_HELPER_TEX_HDR)) {
		throw avk::runtime_error("Error: IBL images not found!");
		//LOG_ERROR("IBL images not found! - disabling IBL");
		//return;
	}

	// load IBL images
	auto [imgBg,  cmdsBg ] = create_image_from_file(IBL_HELPER_TEXPATH IBL_HELPER_TEX_BG, true, true, true, 4, layout::shader_read_only_optimal);
	context().record_and_submit_with_fence({ cmdsBg }, *mQueue)->wait_until_signalled();
	auto imgHdr = load_hdr_img(
		aQueue,
		IBL_HELPER_TEXPATH IBL_HELPER_TEX_HDR, 1u, layout::shader_read_only_optimal // reflection map - aka HDR "environment map"
	); 

	// create background image sampler
	mBackgroundImageSampler = context().create_image_sampler(
		context().create_image_view(std::move(imgBg)),
		context().create_sampler(filter_mode::bilinear, border_handling_mode::repeat, 0.0f) // always use mip level 0, otherwise we get a seam artefact)
	);

	// create hdr environment map image sampler
	mHdrEnvironmentMapImageSampler = context().create_image_sampler(
		context().create_image_view(std::move(imgHdr)), context().create_sampler(filter_mode::bilinear, border_handling_mode::repeat)
	);

	// create a pipeline for creating the maps
	mPipelineBuildIblMap = avk::context().create_compute_pipeline_for(
		compute_shader("shaders/build_ibl_maps.comp"),
		push_constant_binding_data{ shader_type::compute, 0, sizeof(PushConstantsForMapBuildingShader) },
		descriptor_binding(0, 0, mHdrEnvironmentMapImageSampler->as_combined_image_sampler(layout::shader_read_only_optimal) ),	// input: environment map
		descriptor_binding<image_view_as_storage_image>(0, 1, 1u)	// output: irradiance map / prefiltered environment map / brdf lookup table
	);

	mInitialized = true;
}

void IblHelper::assert_initialized()
{
	if (!mInitialized) throw avk::runtime_error("IblHelper is not initialized!");
}

const avk::image_sampler& IblHelper::get_background_image_sampler()
{
	assert_initialized();
	return mBackgroundImageSampler;
}

void IblHelper::render_geometry(avk::command_buffer_t& cb, glm::vec4 aMainPbsOverride, std::function<void(const glm::mat4&aModelMatrix, const glm::vec4&aPbsOverride, int aMaterialIndex)> aSetPushconstantsFunction)
{
	int whichSet = get_geometry_to_render();
	if (whichSet < 0 || whichSet >= mDrawCallsSets.size()) whichSet = 0;

	bool isArraySet = (whichSet == 1);

	glm::mat4 mRotMatrix = glm::mat4(1);
	if (mRotate) {
		constexpr float rotSpeed = glm::radians(45.0f); // per sec
		float angle = fmod(static_cast<float>(avk::context().get_time() * rotSpeed), glm::two_pi<float>());
		mRotMatrix = glm::rotate(angle, glm::vec3(0, 1, 0));
	}

	for (const auto& extDrawCall : mDrawCallsSets[whichSet]) {
		int matIndex = (whichSet == 0) ? extDrawCall.drawCall.mMaterialIndex : get_material_index_to_use();
		glm::mat4 modelMatrix = extDrawCall.mTransformAfterRotate * mRotMatrix * extDrawCall.drawCall.mModelMatrix;
		glm::vec4 pbsOverride;
		if (extDrawCall.hasPbsOverride && !mUseTexturePbrData) {
			pbsOverride = glm::vec4(extDrawCall.metallic, extDrawCall.roughness, extDrawCall.hasPbsOverride ? 1.0f : 0.0f, 0.0f);
		} else {
			pbsOverride = aMainPbsOverride;
		}
		//glm::vec4 pbsOverride = glm::vec4(extDrawCall.metallic, extDrawCall.roughness, extDrawCall.hasPbsOverride ? 1.0f : 0.0f, 0.0f);

		aSetPushconstantsFunction(modelMatrix, pbsOverride, matIndex);
		cb.record(avk::command::draw_indexed(
			extDrawCall.drawCall.mIndexBuffer.as_reference(),     // Index buffer
			extDrawCall.drawCall.mPositionsBuffer.as_reference(), // Vertex buffer at index #0
			extDrawCall.drawCall.mTexCoordsBuffer.as_reference(), // Vertex buffer at index #1
			extDrawCall.drawCall.mNormalsBuffer.as_reference(),   // Vertex buffer at index #2
			extDrawCall.drawCall.mTangentsBuffer.as_reference(),  // Vertex buffer at index #3
			extDrawCall.drawCall.mBitangentsBuffer.as_reference() // Vertex buffer at index #4
		));
	}
}

avk::image IblHelper::load_hdr_img(
	avk::queue* aQueue,
	std::string filename, uint32_t mipLevels, avk::layout::image_layout aImageLayout
)
{
	stbi_set_flip_vertically_on_load(true);
	int width, height, nrComponents;
	float *data = stbi_loadf(filename.c_str(), &width, &height, &nrComponents, 4);

	int nrChannels = 4;
	int bytesPerChannel = 4;

	int data_size = width * height * nrChannels * bytesPerChannel;

	//auto format = vk::Format::eR16G16B16A16Sfloat; // <- not working!
	auto format = vk::Format::eR32G32B32A32Sfloat;
	avk::image_usage aImageUsage = avk::image_usage::general_texture;
	auto img = avk::context().create_image(width, height, format, 1, avk::memory_usage::device, aImageUsage,
		[&](avk::image_t& image) {
			image.create_info().mipLevels = mipLevels;
		}
	);

	avk::buffer staging = avk::context().create_buffer(
		AVK_STAGING_BUFFER_MEMORY_USAGE,
		vk::BufferUsageFlagBits::eTransferSrc,
		avk::generic_buffer_meta::create_from_size(data_size)
	);
	staging->fill(data, 0);

	stbi_image_free(data);

	auto fence = avk::context().record_and_submit_with_fence({
		avk::sync::image_memory_barrier(img.as_reference(),
		avk::stage::none  >> avk::stage::copy,
		avk::access::none >> avk::access::transfer_read | avk::access::transfer_write
		).with_layout_transition(avk::layout::undefined >> avk::layout::transfer_dst),

		avk::copy_buffer_to_image(staging.as_reference(), img.as_reference(), avk::layout::transfer_dst),

		avk::sync::image_memory_barrier(img.as_reference(),
			avk::stage::copy            >> avk::stage::none,
			avk::access::transfer_write >> avk::access::none
		).with_layout_transition(avk::layout::transfer_dst >> aImageLayout)
	}, *mQueue);
	fence->wait_until_signalled();

	if (img->create_info().mipLevels > 1)
	{
		auto fen = avk::context().record_and_submit_with_fence({
			img->generate_mip_maps( aImageLayout >> aImageLayout)
		}, *aQueue);
		fen->wait_until_signalled();
	}

	return std::move(img);
}

void IblHelper::build_maps(avk::queue* aQueue, avk::command_pool &aCommandPool, avk::descriptor_cache &aDescriptorCache)
{
	using namespace avk;

	bool captureInRdoc = true;

	float t0 = static_cast<float>(context().get_time());
	LOG_INFO("---------- Rebuilding IBL maps...");

	// compute the max number of mip levels we can use for the prefiltered environment map
	uint32_t pfeMaxPossibleMipLevels = static_cast<uint32_t>(ceil(log2f(float(std::max(PREFILTERED_ENV_MAP_WIDTH, PREFILTERED_ENV_MAP_HEIGHT)))));
	uint32_t pfeSkipMipLevels = 3; // these would be too low resolution, resulting in artefacts
	uint32_t pfeMipLevels = (int(pfeMaxPossibleMipLevels) - int(pfeSkipMipLevels)) > 1 ? pfeMaxPossibleMipLevels - pfeSkipMipLevels : pfeMaxPossibleMipLevels;
	LOG_INFO(std::format("... using {} mip levels (of max possible {}) for the prefiltered environment map", pfeMipLevels, pfeMaxPossibleMipLevels));

	// create image view for irradiance map
	auto irrMapImgView = avk::context().create_image_view(
		avk::context().create_image(IRRADIANCE_MAP_WIDTH, IRRADIANCE_MAP_HEIGHT, ALL_IBL_MAPS_FORMAT, 1, memory_usage::device, image_usage::general_storage_image)
	);
	//rdoc::labelImage(irrMapImgView->get_image().handle(), "Irradiance Map");

	// create image for prefiltered environment map with multiple MIP levels
	auto pfeMapImg = avk::context().create_image(PREFILTERED_ENV_MAP_WIDTH, PREFILTERED_ENV_MAP_HEIGHT, ALL_IBL_MAPS_FORMAT, 1, memory_usage::device, image_usage::general_storage_image,
		[&pfeMipLevels](image_t &img) {
			img.create_info().mipLevels = pfeMipLevels;
		}
	);
	//rdoc::labelImage(pfeMapImg->handle(), "Prefiltered EnvMap");

	// create temporary image views - one for accessing each mip level of pfeMapImg
	std::vector<image_view> pfeMapImgViewsPerMip;
	for (uint32_t i = 0; i < pfeMipLevels; ++i) {
		auto view = context().create_image_view(pfeMapImg, std::nullopt, {},
			[&i](image_view_t &v) {
				v.create_info().subresourceRange.setBaseMipLevel(i).setLevelCount(1);
			}
		);
		pfeMapImgViewsPerMip.push_back(std::move(view));
	}

	// create image view for BRDF lookup table
	auto brdfLutImgView = avk::context().create_image_view(
		avk::context().create_image(BRDF_LUT_WIDTH, BRDF_LUT_HEIGHT, ALL_IBL_MAPS_FORMAT, 1, memory_usage::device, image_usage::general_storage_image)
	);
	//rdoc::labelImage(brdfLutImgView->get_image().handle(), "BRDF Lookup Table");


	// accumulate all the commands and sync instructions to record
	std::vector<avk::recorded_commands_t> recordedCmds;

	// sync and layout transitions
	recordedCmds.push_back(
		sync::image_memory_barrier(irrMapImgView->get_image(),
			stage::none                    >> stage::compute_shader,
			access::none                   >> access::shader_storage_write
		).with_layout_transition(layout::undefined >> layout::general)
	);
	for (auto &v : pfeMapImgViewsPerMip) {
		recordedCmds.push_back(
			sync::image_memory_barrier(v->get_image(),
				stage::none                    >> stage::compute_shader,
				access::none                   >> access::shader_storage_write
			).with_layout_transition(layout::undefined >> layout::general)
		);
	}
	recordedCmds.push_back(
		sync::image_memory_barrier(brdfLutImgView->get_image(),
			stage::none                    >> stage::compute_shader,
			access::none                   >> access::shader_storage_write
		).with_layout_transition(layout::undefined >> layout::general)
	);

	// compute shader invocations
	recordedCmds.push_back(
		command::custom_commands([&,this](avk::command_buffer_t& cb) {
			PushConstantsForMapBuildingShader pushConstants = {};
			cb.record(avk::command::bind_pipeline(mPipelineBuildIblMap.as_reference()));

			// Irradiance Map
			uint32_t w = irrMapImgView->get_image().width();
			uint32_t h = irrMapImgView->get_image().height();
			cb.record(avk::command::bind_descriptors(mPipelineBuildIblMap->layout(), aDescriptorCache->get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mHdrEnvironmentMapImageSampler->as_combined_image_sampler(layout::shader_read_only_optimal)), // input: environment map
				descriptor_binding(0, 1, irrMapImgView->as_storage_image(layout::general)) // output: irradiance map
			})));
			pushConstants.mMapToBuild = 0;
			pushConstants.mRoughness = 0.0f; // not used here
			cb.record(avk::command::push_constants(mPipelineBuildIblMap->layout(), pushConstants));
			cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);

			// Prefiltered Environment Map
			w = pfeMapImg->width();
			h = pfeMapImg->height();
			for (int mip = 0; mip < int(pfeMapImgViewsPerMip.size()); ++mip) {
				cb.record(avk::command::bind_descriptors(mPipelineBuildIblMap->layout(), aDescriptorCache->get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mHdrEnvironmentMapImageSampler->as_combined_image_sampler(layout::shader_read_only_optimal)), // input: environment map
					descriptor_binding(0, 1, pfeMapImgViewsPerMip[mip]->as_storage_image(layout::general)) // output: prefiltered environment map, mip level mip
				})));
				pushConstants.mMapToBuild = 1;
				pushConstants.mRoughness = float(mip) / float(pfeMipLevels - 1);
				cb.record(avk::command::push_constants(mPipelineBuildIblMap->layout(), pushConstants));
				cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);

				w = std::max(1u, w / 2); h = std::max(1u, h / 2); // adjust dispatch size for next mip level
			}

			// BRDF Lookup Table
			w = brdfLutImgView->get_image().width();
			h = brdfLutImgView->get_image().height();
			cb.record(avk::command::bind_descriptors(mPipelineBuildIblMap->layout(), aDescriptorCache->get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mHdrEnvironmentMapImageSampler->as_combined_image_sampler(layout::shader_read_only_optimal)), // input: environment map
				descriptor_binding(0, 1, brdfLutImgView->as_storage_image(layout::general)) // output: brdf lookup table
			})));
			pushConstants.mMapToBuild = 2;
			pushConstants.mRoughness = 0.0f; // not used here
			cb.record(avk::command::push_constants(mPipelineBuildIblMap->layout(), pushConstants));
			cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
		})
	);

	// do it!
	//if (captureInRdoc) rdoc::start_capture();
	auto cmdBfr = aCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	auto fence = context().create_fence();
	context().record(recordedCmds)
		.into_command_buffer(cmdBfr)
		.then_submit_to(*aQueue)
		.signaling_upon_completion(fence)
		.submit();
	fence->wait_until_signalled();
	//if (captureInRdoc) rdoc::end_capture();

	// transition the layouts again to shader read optimal access
	recordedCmds.clear();
	recordedCmds.push_back(sync::image_memory_barrier(irrMapImgView->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal));
	for (auto &v : pfeMapImgViewsPerMip) {
		recordedCmds.push_back(sync::image_memory_barrier(v->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal));
	}
	recordedCmds.push_back(sync::image_memory_barrier(brdfLutImgView->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal));
	auto fence2 = context().record_and_submit_with_fence(recordedCmds, *aQueue);
	fence2->wait_until_signalled();

	// create image-samplers
	mIrradianceMapImageSampler = context().create_image_sampler(
		std::move(irrMapImgView),
		context().create_sampler(filter_mode::bilinear, border_handling_mode::repeat)
	);
	mPrefilteredEnvMapImageSampler = context().create_image_sampler(
		context().create_image_view(std::move(pfeMapImg)),
		context().create_sampler(filter_mode::trilinear, border_handling_mode::repeat)
	);
	mBrdfLookupTableImageSampler = context().create_image_sampler(
		std::move(brdfLutImgView),
		context().create_sampler(filter_mode::bilinear, border_handling_mode::clamp_to_edge)
	);

	float t = static_cast<float>(context().get_time()) - t0;
	LOG_INFO(std::format("---------- ...done in {} sec", t));

	// clear the descriptor cache, to remove any bound references to the old maps
	aDescriptorCache->cleanup();

	mMapsInitialized = true;
}

void IblHelper::create_dummy_maps()
{
	using namespace avk;

	auto [tex, cmds] = create_1px_texture_cached({ 255, 255, 255, 255 }, layout::shader_read_only_optimal, vk::Format::eR8G8B8A8Unorm);
	context().record_and_submit_with_fence({ std::move(cmds) }, *mQueue)->wait_until_signalled();

	mIrradianceMapImageSampler = context().create_image_sampler(
		context().create_image_view(tex),
		context().create_sampler(filter_mode::bilinear, border_handling_mode::repeat)
	);
	mPrefilteredEnvMapImageSampler = context().create_image_sampler(
		context().create_image_view(tex),
		context().create_sampler(filter_mode::trilinear, border_handling_mode::repeat)
	);
	mBrdfLookupTableImageSampler = context().create_image_sampler(
		context().create_image_view(tex),
		context().create_sampler(filter_mode::bilinear, border_handling_mode::clamp_to_edge)
	);
}

