#include "imgui_utils.h"

// This class handles the ambient occlusion post-processing effect(s).
class ambient_occlusion : public avk::invokee
{
	struct push_constants_for_ssao {
		float mSampleRadius;
		float mDarkeningFactor;
		int mNumSamples;
	};

	struct push_constants_for_blur {
		float mSpatial;
		float mIntensity;
		int mKernelSize;
	};

public:
	/**	An invokee which adds ambient occlusion as a post processing effect
	 */
	ambient_occlusion() : invokee("Ambient Occlusion Post Processing Effect", true)
	{}

	// Execution order of 20 => Execute after assignment4 and before reflections
	int execution_order() const override { return 20; }

	/**	Method to configure this invokee, intended to be invoked BEFORE this invokee's invocation of initialize()
	 *	@param	aQueue					Stores an avk::queue* internally for future use, which has been created previously.
	 *	@param	aDescriptorCache		A descriptor cache that shall be used (possibly allowing descriptor re-use from other invokees)
	 *	@param	aUniformsBuffer			A buffer containing user input and current frame's data
	 *	@param	aSourceColor			Rendered results from previous steps where ambient occlusion shall be added,
	 *									expected to be given in SHADER_READ_ONLY_OPTIMAL layout.
	 *	@param	aSourceDepth			G-Buffer depth values associated to the color values in aSourceColors
	 *	@param	aSourceUvNormal			G-Buffer attachment containing UV coordinates in .rg and spherical normals in .ba
	 *	@param	aDestinationImageView	Destination image view which shall receive the rendered results after the ambient occlusion effect has been added
	 *									Expected to be given in THE SAME FORMAT as aIntermediateImage and in GENERAL layout.
	 */
	void config(avk::queue& aQueue, avk::descriptor_cache aDescriptorCache, avk::buffer aUniformsBuffer, avk::image_view aSourceColor, avk::image_view aSourceDepth, avk::image_view aSourceUvNormal, avk::image_view aDestinationImageView)
	{
		using namespace avk;

		mQueue = &aQueue;
		mDescriptorCache = std::move(aDescriptorCache);
		mUniformsBuffer = std::move(aUniformsBuffer);
		mSrcColor = std::move(aSourceColor);
		mSrcDepth = std::move(aSourceDepth);
		mSrcUvNrm = std::move(aSourceUvNormal);
		mDstResults = std::move(aDestinationImageView);

		mIntermediateImage = context().create_image_view_from_template(mDstResults.get());
		auto fen = context().record_and_submit_with_fence(command::gather(
			// Transition the storage image into GENERAL layout and keep it in that layout forever:
			sync::image_memory_barrier(mIntermediateImage->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general)
		), *mQueue);
		fen->wait_until_signalled();
	}

	/**	Returns the result of the GPU timer query, which indicates how long the SSAO effect approximately took.
	 */ 
	float duration()
	{
		if (!mSsaoEnabled) {
			return 0.0f;
		}
		return helpers::get_timing_interval_in_ms(std::format("ssao {}", mPingPong));
	}

	// Return offsets for sampling the neighborhood during SSAO:
	std::vector<glm::vec4> generate_ambient_occlusion_samples()
	{
		//
		// TODO Task 1: You might want to generate better samples for ambient occlusion
		//

		std::default_random_engine generator;
		generator.seed(345);
		std::uniform_real_distribution<float> distribution(0.0f, 1.0f); // generates floats in the range -1..1

		std::vector<glm::vec4> samplesData;
		samplesData.reserve(144);
		for (int i = 0; i < 128; ++i)
		{
			// Generate random samples on the surface of a sphere:
			glm::vec3 rnd_sample{distribution(generator) * 2 - 1, distribution(generator) * 2 - 1, distribution(generator)};
			auto on_hemisphere_surface = glm::vec4(glm::normalize(rnd_sample), 0.0);
			on_hemisphere_surface *= distribution(generator);

			float scale = (float)i / 128.0;
			on_hemisphere_surface *= std::lerp(0.1f, 1.0f, scale * scale);
			on_hemisphere_surface *= scale;

			samplesData.push_back(on_hemisphere_surface);
		}

		std::shuffle(samplesData.begin(), samplesData.begin() + 128, generator);

		return samplesData;
	}

	std::vector<glm::vec4> generate_noise()
	{
		std::default_random_engine generator;
		generator.seed(345);
		std::uniform_real_distribution<float> distribution(0.0f, 1.0f); // generates floats in the range -1..1

		std::vector<glm::vec4> noiseData;
		noiseData.reserve(64);
		for (int i = 0; i < 64; ++i)
		{
			glm::vec4 noise{ distribution(generator), distribution(generator) , 0.0f, 0.0f };
			noiseData.push_back(noise);
		}

		return noiseData;
	}
	
	// Create all the compute pipelines used for the post processing effect(s),
	// prepare some command buffers with pipeline barriers to synchronize with subsequent commands,
	// create a new ImGui window that allows to enable/disable ambient occlusion, and to modify parameters:
	void initialize() override 
	{
		using namespace avk;

		// Create a command pool for allocating single-use (hence, transient) command buffers:
		mCommandPool = context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		// Use this invokee's updater to enable shader hot reloading
		mUpdater.emplace();

		auto samplesData = generate_ambient_occlusion_samples();
		mRandomSamplesBuffer = context().create_buffer(
			memory_usage::device, {},
			uniform_buffer_meta::create_from_data(samplesData)
		);
		auto fen = context().record_and_submit_with_fence({
			mRandomSamplesBuffer->fill(samplesData.data(), 0)
		}, *mQueue);
		fen->wait_until_signalled();

		// Noise ---------------------------------------------

		auto noiseData = generate_noise();

		mNoiseBuffer = context().create_buffer(
			memory_usage::device, { vk::BufferUsageFlagBits::eTransferSrc },
			uniform_buffer_meta::create_from_data(noiseData)
		);

		fen = context().record_and_submit_with_fence({
			mNoiseBuffer->fill(noiseData.data(), 0)
			}, *mQueue);
		fen->wait_until_signalled();

		mOcclusionFactorsPipeline = context().create_compute_pipeline_for(
			"shaders/ssao.comp",
			push_constant_binding_data { shader_type::compute, 0, sizeof(push_constants_for_ssao) },
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 1, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 2, 1u),
			descriptor_binding(1, 0, mUniformsBuffer),
			descriptor_binding(2, 0, mRandomSamplesBuffer),
			descriptor_binding(3, 0, mNoiseBuffer)
		);

		mUpdater->on(shader_files_changed_event(mOcclusionFactorsPipeline.as_reference()))
			.update(mOcclusionFactorsPipeline);

		//
		// TODO Task 2: Add a pipeline to blur the occlusion factors
		//

		mBlurOcclusionFactorsPipeline = context().create_compute_pipeline_for(
			"shaders/blur_occlusion_factors.comp",
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_blur) },
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 1, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 2, 1u)
		);

		mUpdater->on(shader_files_changed_event(mBlurOcclusionFactorsPipeline.as_reference()))
			.update(mBlurOcclusionFactorsPipeline);

		mApplyOcclusionFactorsPipeline = context().create_compute_pipeline_for(
			"shaders/apply_occlusion_factors.comp",
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 1, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 2, 1u)
		);

		mUpdater->on(shader_files_changed_event(mApplyOcclusionFactorsPipeline.as_reference()))
			.update(mApplyOcclusionFactorsPipeline);
		
		// Create a new ImGui window:
		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (nullptr == imguiManager) {
			LOG_ERROR("Failed to install UI callback for ambient_occlusion, because composition does not contain an element of type avk::imgui_manager.");
			return;
		}

		imguiManager->add_callback([this](){
			ImGui::Begin("Ambient Occlusion Settings");
			ImGui::SetWindowPos(ImVec2(295.0f, 10.0f), ImGuiCond_FirstUseEver);
			ImGui::SetWindowSize(ImVec2(220.0f, 160.0f), ImGuiCond_FirstUseEver);
			ImGui::Checkbox("enabled", &mSsaoEnabled);
			ImGui::SliderInt("#samples", &mNumSamples, 1, 128);
			ImGui::SliderFloat("radius", &mSampleRadius, 0.0f, 6.0f);
			ImGui::SliderFloat("darkening factor", &mDarkeningFactor, 0.0f, 5.0f);
			static const char* sOcclusionItems[] = { "display occlusion factors", "apply occlusion factors" };
			ImGui::Combo("occlusion factors", &mApplyOcclusionFactors, sOcclusionItems, IM_ARRAYSIZE(sOcclusionItems));
			static const char* sBlurItems[] = { "don't blur occlusion factors", "blur occlusion factors" };
			ImGui::Combo("blur", &mBlurOcclusionFactors, sBlurItems, IM_ARRAYSIZE(sBlurItems));
			ImGui::SliderFloat("sigma intensity", &mIntensity, 0.1f, 10.0f);
			ImGui::SliderFloat("sigma spatial", &mSpatial, 0.1f, 10.0f);
			ImGui::End();
		});
	}

	// Update the push constant data that will be used in render():
	void update() override 
	{
		using namespace avk;

	}

	// Create a new command buffer every frame, record instructions into it, and submit it to the graphics queue:
	void render() override 
	{
		using namespace avk;

		mOcclusionFactorsPushConstants.mNumSamples = mNumSamples;
		mOcclusionFactorsPushConstants.mSampleRadius = mSampleRadius;
		mOcclusionFactorsPushConstants.mDarkeningFactor = mDarkeningFactor;

		mBlurPushConstants.mIntensity = mIntensity;
		mBlurPushConstants.mSpatial = mSpatial;
		mBlurPushConstants.mKernelSize = 5;

		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		context().record({ // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):

			command::custom_commands([this](avk::command_buffer_t& cb) {

				if (mSsaoEnabled) {
					// ---------------------- If SSAO is enabled perform the following actions --------------------------

					mPingPong = 1 - mPingPong;
					helpers::record_timing_interval_start(cb.handle(), std::format("ssao {}", mPingPong));

					const auto w = mDstResults->get_image().width();
					const auto h = mDstResults->get_image().height();

					// ------> 1st step (and also SSAO's main step): Generate the occlusion factors
					cb.record(avk::command::bind_pipeline(mOcclusionFactorsPipeline.as_reference()));
					cb.record(avk::command::bind_descriptors(mOcclusionFactorsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
						descriptor_binding(0, 0, mSrcDepth->as_sampled_image(layout::shader_read_only_optimal)),
						descriptor_binding(0, 1, mSrcUvNrm->as_sampled_image(layout::shader_read_only_optimal)),
						descriptor_binding(0, 2, mIntermediateImage->as_storage_image(layout::general)),
						descriptor_binding(1, 0, mUniformsBuffer),
						descriptor_binding(2, 0, mRandomSamplesBuffer),
						descriptor_binding(3, 0, mNoiseBuffer),
					})));
					cb.record(avk::command::push_constants(mOcclusionFactorsPipeline->layout(), mOcclusionFactorsPushConstants));
					cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);

					if (mBlurOcclusionFactors) {
						// ------> 2nd step: Blur the occlusion factors
						//
						// TODO Task 2: Blur the occlusion factors by using compute shader(s)!
						//              Make sure to write the blurred results into mIntermediateImage!
						//              Furthermore, establish proper synchronization! (It is okay to use a global memory barrier, you don't have to use image memory barriers.)
						//

						cb.record(sync::global_memory_barrier(
							stage::compute_shader >> stage::compute_shader,
							access::shader_storage_write >> access::shader_read
						));

						cb.record(avk::command::bind_pipeline(mBlurOcclusionFactorsPipeline.as_reference()));
						cb.record(avk::command::bind_descriptors(mBlurOcclusionFactorsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mIntermediateImage->as_sampled_image(layout::general)), // uOcclusionFactors
							descriptor_binding(0, 1, mSrcDepth->as_sampled_image(layout::shader_read_only_optimal)), // uDepth
							descriptor_binding(0, 2, mIntermediateImage->as_storage_image(layout::general)), // uDst
						})));
						cb.record(avk::command::push_constants(mBlurOcclusionFactorsPipeline->layout(), mBlurPushConstants));
						cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
					}

					if (mApplyOcclusionFactors) {
						// ------> 3rd step: apply the occlusion factors
						cb.record(sync::global_memory_barrier(
							// We have to wait for previous compute finish, before we may invoke another compute shader that needs to access the same data:
							stage::compute_shader        >> stage::compute_shader,
							// The compute's image writes must be made available and visible to the next compute shader's read access:
							access::shader_storage_write >> access::shader_read
						));
						cb.record(avk::command::bind_pipeline(mApplyOcclusionFactorsPipeline.as_reference()));
						cb.record(avk::command::bind_descriptors(mApplyOcclusionFactorsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mSrcColor->as_sampled_image(layout::read_only_optimal)),
							descriptor_binding(0, 1, mIntermediateImage->as_sampled_image(layout::general)),
							descriptor_binding(0, 2, mDstResults->as_storage_image(layout::general)),
						})));
						cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
					}
					else {
						// Just copy over, to display the occlusion factors on the screen:
						cb.record(sync::global_memory_barrier(
							// The compute shader must complete execution before the transfer operation (copy) may resume:
							stage::compute_shader        >> stage::copy,
							// The compute's image writes must be made available and visible to the transfer operation's memory access type:
							access::shader_storage_write >> access::transfer_read
						));
						cb.record(copy_image_to_another(mIntermediateImage->get_image(), layout::general, mDstResults->get_image(), layout::general));
					}

					helpers::record_timing_interval_end(cb.handle(), std::format("ssao {}", mPingPong));
				}

				// -------------------------- If SSAO is disabled, do nothing but blit ------------------------------
				else {
					cb.record(
						sync::image_memory_barrier(mSrcColor->get_image(),
							// Any previous color attachment writes or compute shader writes must have completed before we may blit:
							stage::color_attachment_output | stage::compute_shader        >> stage::blit,
							// Writes must be made available and visible to the transfer read:
							access::color_attachment_write | access::shader_storage_write >> access::transfer_read
						).with_layout_transition(layout::shader_read_only_optimal >> layout::transfer_src)
					);
					cb.record(blit_image(mSrcColor->get_image(), layout::transfer_src, mDstResults->get_image(), layout::general));
					cb.record(
						sync::image_memory_barrier(mSrcColor->get_image(),
							// As soon as blit is done, change layout back, and assume that afterwards, there will be compute or transfer accessing this image:
							stage::blit   >>  stage::compute_shader  | stage::transfer,
							access::none  >>  access::none
						).with_layout_transition(layout::transfer_src >> layout::shader_read_only_optimal)
					);
				}

				// In any case, sync with subsequent compute or transfer commands:
				cb.record(sync::global_memory_barrier(
					// Compute or transfer operations must complete before other compute or transfer operations may proceed:
					stage::compute_shader | stage::transfer               >> stage::compute_shader | stage::transfer,
					// Particularly write accesses must be made available, and we just assume here that next access for resources will be read:
					access::shader_storage_write | access::transfer_write >> access::shader_read | access::transfer_read
				));
			})
			
		}) // End of command recording
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

	// Settings, which can be modified via ImGui:
	bool mSsaoEnabled = true;
	int mNumSamples = 32;
	float mSampleRadius = 2.0f;
	float mDarkeningFactor = 1.5;
	int mApplyOcclusionFactors = 1;
	int mBlurOcclusionFactors = 0;
	float mIntensity = 1.0f;
	float mSpatial = 1.0f;

	int mPingPong{ 1 };

	// Source image views:
	avk::image_view mSrcDepth;
	avk::image_view mSrcUvNrm;
	avk::image_view mSrcColor;

	// Destination image view:
	avk::image_view mDstResults;
	// Images for intermediate results:
	avk::image_view mIntermediateImage;
	// Buffer containing the user input and matrices:
	avk::buffer mUniformsBuffer;

	// Random samples which are used by SSAO:
	avk::buffer mRandomSamplesBuffer;
	avk::buffer mNoiseBuffer;
	
	// The "main SSAO" pipeline, that creates the occlusion factors:
	avk::compute_pipeline mOcclusionFactorsPipeline;
	// Push constants used in the mOcclusionFactorsPipeline:
	push_constants_for_ssao mOcclusionFactorsPushConstants;
	// Push constants used in the mBlurOcclusionFactorsPipeline:
	push_constants_for_blur mBlurPushConstants;

	avk::compute_pipeline mBlurOcclusionFactorsPipeline;

	// Pipeline which applies the occlusion factors to the source color image
	avk::compute_pipeline mApplyOcclusionFactorsPipeline;

};

