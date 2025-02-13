#include "imgui_utils.h"

// This class handles the reflection post-processing effect(s).
class reflections : public avk::invokee
{

	struct push_constants_data {
		int mMaxSteps;
		float mStepSize;
		float mEpsilon;
	};

public:
	/**	An invokee which adds ambient occlusion as a post processing effect
	 */
	reflections() : invokee("Reflections Post Processing Effects", true)
	{ }

	// Execution order of 40 => Execute after ambient occlusion and before tone mapping
	int execution_order() const override { return 40; }
	
	/**	Method to configure this invokee, intended to be invoked BEFORE this invokee's invocation of initialize()
	 *	@param	aQueue							Stores an avk::queue* internally for future use, which has been created previously.
	 *	@param	aDescriptorCache				A descriptor cache that shall be used (possibly allowing descriptor re-use from other invokees)
	 *	@param	aUniformsBuffer					A buffer containing user input and current frame's data
	 *	@param	aSourceColor					Rendered results from previous steps where ambient occlusion shall be added,
	 *											expected to be given in GENERAL layout.
	 *	@param	aSourceDepth					G-Buffer depth values associated to the color values in aSourceColors
	 *	@param	aSourceUvNormal					G-Buffer attachment containing UV coordinates in .rg and spherical normals in .ba
	 *	@param	aSourceMatId					G-Buffer attachment containing the material id
	 *	@param	aDestinationImageView			Destination image view which shall receive the rendered results after the ambient occlusion effect has been added
	 *											Expected to be given in THE SAME FORMAT as aIntermediateImage and in GENERAL layout.
	 *	@param	aMaterialsBuffer				A buffer containing all the materials of the scene
	 *	@param	aImageSamplerDescriptorInfos	A vector containing all the image samplers that are used/referenced in the data of aMaterialsBuffer
	 */
	void config(avk::queue& aQueue, avk::descriptor_cache aDescriptorCache, avk::buffer aUniformsBuffer,
		avk::image_view aSourceColor, avk::image_view aSourceDepth, avk::image_view aSourceUvNormal, avk::image_view aSourceMatId,
		avk::image_view aDestinationImageView,
		avk::buffer aMaterialsBuffer, std::vector<avk::combined_image_sampler_descriptor_info> aImageSamplerDescriptorInfos)
	{
		using namespace avk;

		mQueue = &aQueue;
		mDescriptorCache = std::move(aDescriptorCache);
		mUniformsBuffer = std::move(aUniformsBuffer);
		mSrcColor = std::move(aSourceColor);
		mSrcDepth = std::move(aSourceDepth);
		mSrcUvNrm = std::move(aSourceUvNormal);
		mSrcMatId = std::move(aSourceMatId);
		mDstResults = std::move(aDestinationImageView);
		mMaterials = std::move(aMaterialsBuffer);
		mImageSamplerDescriptorInfos = std::move(aImageSamplerDescriptorInfos);

		mIntermediateImage = context().create_image_view_from_template(mDstResults.get());
		auto fen = context().record_and_submit_with_fence(command::gather(
			// Transition the storage image into GENERAL layout and keep it in that layout forever:
			sync::image_memory_barrier(mIntermediateImage->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general)
		), *mQueue);
		fen->wait_until_signalled();
	}

	/**	Method to configure this invokee for ray traced reflections, intended to be invoked BEFORE this invokee's invocation of initialize()
	 *	@param	aLightsBuffer							A buffer containing light source data
	 *	@param	aIndexBufferUniformTexelBufferViews		A vector of descriptors to uniform texel buffers, containing indices data
	 *	@param	aNormalBufferUniformTexelBufferViews	A vector of descriptors to uniform texel buffers, containing normals data
	 *	@param	aTopLevelAS								A top level acceleration structure for ray tracing, it shall contain the whole scene
	 */
	void config_rtx_on(
		avk::buffer aLightsBuffer,
		std::vector<avk::buffer_view_descriptor_info> aIndexBufferUniformTexelBufferViews, std::vector<avk::buffer_view_descriptor_info> aNormalBufferUniformTexelBufferViews,
		avk::top_level_acceleration_structure aTopLevelAS)
	{
		mRtxOn.emplace();

		mLightsBuffer = std::move(aLightsBuffer);
		mIndexBufferUniformTexelBufferViews = std::move(aIndexBufferUniformTexelBufferViews);
		mNormalBufferUniformTexelBufferViews = std::move(aNormalBufferUniformTexelBufferViews);
		mTopLevelAS = std::move(aTopLevelAS);
		//
		// TODO Bonus Task 3 RTX ON: Pass additional required resources and use them!
		//
	}
	
	/**	Returns the result of the GPU timer query, which indicates how long the reflections effect approximately took.
	 */
	float duration()
	{
		if (!mReflectionsEnabled) {
			return 0.0f;
		}
		return helpers::get_timing_interval_in_ms(std::format("reflections {}", mPingPong));
	}

	// Create all the compute (and ray-tracing) pipelines used for the post processing effect(s),
	// prepare some command buffers with pipeline barriers to synchronize with subsequent commands,
	// create a new ImGui window that allows to enable/disable reflections, and to modify parameters:
	void initialize() override 
	{
		using namespace avk;

		// Create a command pool for allocating single-use (hence, transient) command buffers:
		mCommandPool = context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		// Use this invokee's updater to enable shader hot reloading
		mUpdater.emplace();

		mGenerateReflectionsPipeline = context().create_compute_pipeline_for(
			"shaders/ssr.comp",
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_data) },
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 1, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 2, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 3, 1u),
			descriptor_binding(1, 0, mUniformsBuffer),
			descriptor_binding<image_view_as_storage_image>(2, 0, 1u)
		);

		mUpdater->on(shader_files_changed_event(mGenerateReflectionsPipeline.as_reference()))
			.update(mGenerateReflectionsPipeline);

		mApplyReflectionsPipeline = context().create_compute_pipeline_for(
			"shaders/apply_reflections.comp",
			descriptor_binding(0, 0, mMaterials),
			descriptor_binding(0, 1, mImageSamplerDescriptorInfos),
			descriptor_binding<image_view_as_sampled_image>(1, 0, 1u),
			descriptor_binding<image_view_as_sampled_image>(1, 1, 1u),
			descriptor_binding<image_view_as_sampled_image>(1, 2, 1u),
			descriptor_binding<image_view_as_sampled_image>(1, 3, 1u),
			descriptor_binding<image_view_as_sampled_image>(2, 0, 1u),
			descriptor_binding<image_view_as_storage_image>(2, 1, 1u)
		);

		mUpdater->on(shader_files_changed_event(mApplyReflectionsPipeline.as_reference()))
			.update(mApplyReflectionsPipeline);

		// Create a new ImGui window:
		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if(nullptr != imguiManager) {
			imguiManager->add_callback([this](){
				ImGui::Begin("Reflections Settings");
				ImGui::SetWindowPos(ImVec2(295.0f, 180.0f), ImGuiCond_FirstUseEver);
				ImGui::SetWindowSize(ImVec2(220.0f, 115.0f), ImGuiCond_FirstUseEver);
				ImGui::Checkbox("enabled", &mReflectionsEnabled);
				static const char* sOcclusionItems[] = { "display reflections", "apply reflections" };
				ImGui::Combo("apply?", &mApplyReflections, sOcclusionItems, IM_ARRAYSIZE(sOcclusionItems));
				ImGui::SliderInt("max steps", &mMaxSteps, 10, 200);
				ImGui::SliderFloat("step size", &mStepSize, 0.1, 1);
				ImGui::SliderFloat("epsilon", &mEpsilon, 0.01, 0.1);
				if (mRtxOn.has_value()) {
					static const char* sRtxOffOn[] = { "RTX OFF (use Screen Space Reflections)", "RTX ON" };
					ImGui::Combo("type", &mRtxOn.value(), sRtxOffOn, IM_ARRAYSIZE(sRtxOffOn));
				}
				ImGui::End();
			});
		}
		else {
			LOG_ERROR("Failed to init GUI, because composition does not contain an element of type avk::imgui_manager.");
		}

		// If a top level acceleration structure has been passed, ...
		if (mTopLevelAS.has_value()) {

			// If any of these asserts fails: Have you passed everything properly to config_rtx_on()?
			assert(mUniformsBuffer.has_value());
			assert(mLightsBuffer.has_value());
			assert(!mIndexBufferUniformTexelBufferViews.empty());
			assert(!mNormalBufferUniformTexelBufferViews.empty());

			// ... create a ray tracing pipeline:
			mRayTracingPipeline = context().create_ray_tracing_pipeline_for(
				// Specify all the shaders which participate in rendering in a shader binding table (the order matters):
				//
				// TODO Bonus Task 5 RTX ON: Add further shaders for a recursive shadow pass to the shader binding table and 
				//                           invoke them recursively at an appropriate location in one of the existing shaders!
				//
				define_shader_table(
					ray_generation_shader("shaders/ray_tracing/rtx_on.rgen"),
					miss_shader("shaders/ray_tracing/rtx_on.rmiss"),
					triangles_hit_group::create_with_rchit_only("shaders/ray_tracing/rtx_on.rchit")
				),
				//
				// TODO Bonus Task 5 RTX ON: Configure the ray tracing pipeline to enable recursive traceRayEXT calls!
				//
				max_recursion_depth::disable_recursion(), // No need for recursions
				descriptor_binding(0, 0, mMaterials),
				descriptor_binding(0, 1, mImageSamplerDescriptorInfos),
				descriptor_binding(1, 0, mUniformsBuffer),
				descriptor_binding(1, 1, mLightsBuffer),
				descriptor_binding<image_view_as_sampled_image>(2, 0, 1u),
				descriptor_binding<image_view_as_sampled_image>(2, 1, 1u),
				descriptor_binding<image_view_as_sampled_image>(2, 2, 1u),
				descriptor_binding<image_view_as_sampled_image>(2, 3, 1u),
				descriptor_binding<image_view_as_storage_image>(3, 0, 1u),
				descriptor_binding(4, 0, mIndexBufferUniformTexelBufferViews),
				descriptor_binding(4, 1, mNormalBufferUniformTexelBufferViews),
				//
				// TODO Bonus Task 3 RTX ON: Define additional resource bindings in the pipeline layout:
				//
				descriptor_binding(5, 0, mTopLevelAS)
			);

			mUpdater->on(shader_files_changed_event(mRayTracingPipeline.as_reference()))
				.update(mRayTracingPipeline);

			// Print the structure of our shader binding table, also displaying the offsets:
			mRayTracingPipeline->print_shader_binding_table_groups();
		}
	}
	
	void update() override
	{
		mPushConstants.mMaxSteps = mMaxSteps;
		mPushConstants.mStepSize = mStepSize;
		mPushConstants.mEpsilon = mEpsilon;
	}

	// Create a new command buffer every frame, record instructions into it, and submit it to the graphics queue:
	void render() override 
	{
		using namespace avk;

		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		context().record(command::gather( // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):

			command::conditional([this] { return mReflectionsEnabled; },
				[this] { return command::custom_commands([this](avk::command_buffer_t& cb) {

					mPingPong = 1 - mPingPong;
					helpers::record_timing_interval_start(cb.handle(), std::format("reflections {}", mPingPong));

					const auto w = mDstResults->get_image().width();
					const auto h = mDstResults->get_image().height();
					cb.record(sync::global_memory_barrier(stage::color_attachment_output >> stage::compute_shader, 
					                                      access::color_attachment_write >> access::shader_sampled_read));

					// ------> 1st step: Generate reflections
					if (0 == mRtxOn.value_or(0)) {
						// =================  vvv   SSR   vvv  ================
						// Generate reflections using screen space reflections:
						cb.record(avk::command::bind_pipeline(mGenerateReflectionsPipeline.as_reference()));
						cb.record(avk::command::bind_descriptors(mGenerateReflectionsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mSrcDepth->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(0, 1, mSrcUvNrm->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(0, 2, mSrcMatId->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(0, 3, mSrcColor->as_sampled_image(layout::general)),
							descriptor_binding(1, 0, mUniformsBuffer),
							descriptor_binding(2, 0, mIntermediateImage->as_storage_image(layout::general)) 
						})));
						cb.record(avk::command::push_constants(mGenerateReflectionsPipeline->layout(), mPushConstants));
						cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
						// =================  ^^^   SSR   ^^^  ================
					}
					else {
						// =================  vvv  RTX ON  vvv  ================
						// Generate reflections using ray tracing (if it has been enabled in assignment4.cpp):
						if (mRayTracingPipeline.has_value()) {
							cb.record(avk::command::bind_pipeline(mRayTracingPipeline.as_reference()));
							cb.record(avk::command::bind_descriptors(mRayTracingPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
								descriptor_binding(0, 0, mMaterials),
								descriptor_binding(0, 1, mImageSamplerDescriptorInfos),
								descriptor_binding(1, 0, mUniformsBuffer),
								descriptor_binding(1, 1, mLightsBuffer),
								descriptor_binding(2, 0, mSrcDepth->as_sampled_image(layout::shader_read_only_optimal)),
								descriptor_binding(2, 1, mSrcUvNrm->as_sampled_image(layout::shader_read_only_optimal)),
								descriptor_binding(2, 2, mSrcMatId->as_sampled_image(layout::shader_read_only_optimal)),
								descriptor_binding(2, 3, mSrcColor->as_sampled_image(layout::general)),
								descriptor_binding(3, 0, mIntermediateImage->as_storage_image(layout::general)),
								descriptor_binding(4, 0, mIndexBufferUniformTexelBufferViews),
								descriptor_binding(4, 1, mNormalBufferUniformTexelBufferViews),
								//
								// TODO Bonus Task 3 RTX ON: Pass additional resources to the ray tracing pipeline!
								//
								descriptor_binding(5, 0, mTopLevelAS)
							})));
							cb.record(avk::command::trace_rays(
								vk::Extent3D{ w, h, 1u },
								mRayTracingPipeline->shader_binding_table(),
								avk::using_raygen_group_at_index(0),
								avk::using_miss_group_at_index(0),
								avk::using_hit_group_at_index(0)
							));
						}
						else {
							LOG_ERROR("mRayTracingPipeline has not been created. Cannot use it.");
						}
						// =================  ^^^  RTX ON  ^^^  ================
					}

					auto srcStages = 1 == mRtxOn.value_or(0) ? stage::ray_tracing_shader : stage::compute_shader;
					if (1 == mApplyReflections) {
						// ------> 2nd step: Apply the reflections
						cb.record(sync::image_memory_barrier(mIntermediateImage->get_image(),
							// We have to wait for previous operation that wrote reflections into mIntermediateImage before using it in a compute shader:
							srcStages                      >>   stage::compute_shader,
							// Writes to the image must be made available and visible to reads in the subsequent compute shader:
							access::shader_storage_write   >>   access::shader_read
						));
						
						cb.record(avk::command::bind_pipeline(mApplyReflectionsPipeline.as_reference()));
						cb.record(avk::command::bind_descriptors(mApplyReflectionsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mMaterials),
							descriptor_binding(0, 1, mImageSamplerDescriptorInfos),
							descriptor_binding(1, 0, mSrcDepth->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(1, 1, mSrcUvNrm->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(1, 2, mSrcMatId->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(1, 3, mSrcColor->as_sampled_image(layout::general)),
							descriptor_binding(2, 0, mIntermediateImage->as_sampled_image(layout::general)),
							descriptor_binding(2, 1, mDstResults->as_storage_image(layout::general))
						})));
						cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
					}
					else {
						// ------> Alternative 2nd step: For debug purposes, display the reflected values instead of applying them to the input image
						cb.record(sync::image_memory_barrier(mIntermediateImage->get_image(),
							// We have to wait for previous operation that wrote reflections into mIntermediateImage before using it in the transfer operation:
							srcStages                     >>  stage::copy,
							// Writes to the image must be made available and visible to reads of the transfer operation:
							access::shader_storage_write  >>  access::transfer_read
						));
						cb.record(copy_image_to_another(mIntermediateImage->get_image(), layout::general, mDstResults->get_image(), layout::general));
					}

					helpers::record_timing_interval_end(cb.handle(), std::format("reflections {}", mPingPong));
				}); },
				[this] { 
					return copy_image_to_another(mSrcColor->get_image(), layout::general, mDstResults->get_image(), layout::general); 
				}
			),
			
			// In any case, sync with subsequent compute or transfer commands:
			sync::global_memory_barrier(
				// Compute or transfer operations must complete before other compute or transfer operations may proceed:
				stage::compute_shader | stage::transfer               >> stage::compute_shader | stage::transfer,
				// Particularly write accesses must be made available, and we just assume here that next access for resources will be read:
				access::shader_storage_write | access::transfer_write >> access::shader_read | access::transfer_read
			)
		)) // End of command recording
		.into_command_buffer(cmdBfr)
		.then_submit_to(*mQueue)
		.submit();

		// Use a convenience function of avk::window to take care of the command buffer's lifetime:
		// It will get deleted in the future after #concurrent-frames have passed by.
		context().main_window()->handle_lifetime(std::move(cmdBfr));
	}
	
private:
	/** One single queue to submit all the commands to: */
	avk::queue* mQueue;

	/** One descriptor cache to use for allocating all the descriptor sets from: */
	avk::descriptor_cache mDescriptorCache;

	/** A command pool for allocating (single-use) command buffers from: */
	avk::command_pool mCommandPool;

	int mPingPong{ 1 };

	// Source image views:
	avk::image_view mSrcDepth;
	avk::image_view mSrcUvNrm;
	avk::image_view mSrcMatId;
	avk::image_view mSrcColor;
	// Destination image view:
	avk::image_view mDstResults;
	// Images for intermediate results:
	avk::image_view mIntermediateImage;
	// Buffer containing all the different materials as loaded from 3D models/ORCA scenes:
	avk::buffer mMaterials;
	// Set of image samplers which are referenced by the materials in mMaterials:
	std::vector<avk::combined_image_sampler_descriptor_info> mImageSamplerDescriptorInfos;
	// Buffer containing the user input and matrices:
	avk::buffer mUniformsBuffer;
	// Buffer containing the light source data:
	avk::buffer mLightsBuffer;
	
	// Settings, which can be modified via ImGui:
	bool mReflectionsEnabled = true;
	int mApplyReflections = 1;
	std::optional<int> mRtxOn;

	int mMaxSteps = 100;
	float mStepSize = 0.1;
	float mEpsilon = 0.05;

	push_constants_data mPushConstants;

	avk::compute_pipeline mGenerateReflectionsPipeline;

	avk::top_level_acceleration_structure mTopLevelAS;
	std::vector<avk::buffer_view_descriptor_info> mIndexBufferUniformTexelBufferViews;
	std::vector<avk::buffer_view_descriptor_info> mNormalBufferUniformTexelBufferViews;

	avk::ray_tracing_pipeline mRayTracingPipeline;

	avk::compute_pipeline mApplyReflectionsPipeline;
};

