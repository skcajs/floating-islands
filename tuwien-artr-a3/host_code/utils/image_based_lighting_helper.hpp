#pragma once

class IblHelper {
public:
	/** Constructor
	 *	@param	aQueue	Stores an avk::queue* internally for future use, which has been created previously.
	 */
	IblHelper(avk::queue& aQueue);

	void initialize(std::vector<helpers::data_for_draw_call>& aDrawCalls, helpers::LoadedMaterialsInfo aLoadedMaterials
		, avk::queue* aQueue
	);
	const avk::image_sampler& get_background_image_sampler();

	void render_geometry(avk::command_buffer_t& cb, glm::vec4 aMainPbsOverride, std::function<void(const glm::mat4 &aModelMatrix, const glm::vec4 &aPbsOverride, int aMaterialIndex)> aSetPushconstantsFunction);

	void set_geometry_to_render(int aGeo) { mGeometryToRender = aGeo; }
	int  get_geometry_to_render() { return mGeometryToRender; }

	void set_use_texture_pbr_data(bool aUseIt) { mUseTexturePbrData = aUseIt; }
	bool get_use_texture_pbr_data() { return mUseTexturePbrData; }

	void set_material_index_to_use(int matIdx) { mMaterialIndexToUse = matIdx; }
	int get_material_index_to_use() { return mMaterialIndexToUse; }

	void set_rotate(bool aRotate) { mRotate = aRotate; }
	bool get_rotate() { return mRotate; }

	bool is_initialized() { return mInitialized; }
	void make_shaders_hot_reloadable(std::optional<avk::updater> aUpdater) {
		if (!mInitialized) return;
		if (!aUpdater.has_value()) return;
		aUpdater->on(avk::shader_files_changed_event(mPipelineBuildIblMap.as_reference()))
			.update(mPipelineBuildIblMap);
	}

	void build_maps(avk::queue* aQueue, avk::command_pool &aCommandPool, avk::descriptor_cache &aDescriptorCache);

	bool are_maps_initialized() { return mMapsInitialized; }
	void invalidate_maps() { mMapsInitialized = false; }

	// These image samplers are always valid (after initialize() has been called), regardless whether build_maps() has been called or not
	const avk::image_sampler& get_irradiance_map() { return mIrradianceMapImageSampler; }
	const avk::image_sampler& get_prefiltered_environment_map() { return mPrefilteredEnvMapImageSampler; }
	const avk::image_sampler& get_brdf_lookup_table() { return mBrdfLookupTableImageSampler; }
private:
	void assert_initialized();
	avk::image load_hdr_img(
		avk::queue* aQueue,
		std::string filename, uint32_t mipLevels = 1, avk::layout::image_layout aImageLayout = avk::layout::general);
	void make_geometry_buffers_shared(simple_geometry &g);
	void create_dummy_maps();

	const std::string mDefaultMaterialName = "a3_ibl_spheres"; // "fabric_g";

	struct ExtendedDrawCallData {
		helpers::data_for_draw_call drawCall;
		bool hasPbsOverride = false;
		float roughness = 0.0f, metallic = 0.0f;
		glm::mat4 mTransformAfterRotate = glm::mat4(1);
	};

	struct PushConstantsForMapBuildingShader {
		int   mMapToBuild; // 0: irradiance map, 1: pre-filtered environment map, 2: brdf lookup table
		float mRoughness;  // roughness for (the current mip level of) the pre-filtered environment map
	};

	avk::queue* mQueue;

	std::vector<std::vector<ExtendedDrawCallData>> mDrawCallsSets;
	avk::image_sampler mBackgroundImageSampler, mHdrEnvironmentMapImageSampler;
	avk::compute_pipeline mPipelineBuildIblMap;

	// the image-samplers computed in build_maps:
	avk::image_sampler mIrradianceMapImageSampler, mPrefilteredEnvMapImageSampler, mBrdfLookupTableImageSampler;

	int mGeometryToRender = 1;
	simple_geometry mSphere;
	helpers::LoadedMaterialsInfo mMaterialInfo;
	int mMaterialIndexToUse = 7;
	bool mUseTexturePbrData = false;
	bool mRotate = false;

	bool mInitialized = false;
	bool mMapsInitialized = false;
};

