#include "imgui_utils.h"

// This class handles the tone mapping post-processing effect(s).
class tone_mapping : public avk::invokee
{
	struct push_constants_data {
		VkBool32 mToneMappingEnabled;
		VkBool32 mGradualAdaption;
		VkBool32 mUseMax;
		float mKey;
		float mDeltaTime;
		float mAdaptionSpeed;
	};

	struct lum_data {
		float mAvgLum;
		float mMaxLum;
		glm::vec2 _padding;
	};
	
public:
	tone_mapping() : invokee("Tone Mapping Post Processing Effect", true)
	{ }

	// Execute after reflections and before anti-aliasing:
	int execution_order() const { return 60; }

	/**	Method to configure this invokee, intended to be invoked BEFORE this invokee's invocation of initialize()
	 *	@param	aQueue				Stores an avk::queue* internally for future use, which has been created previously.
	 *	@param	aDescriptorCache	A descriptor cache that shall be used (possibly allowing descriptor re-use from other invokees)
	 *	@param	aSourceHdr			Input image in HDR format which contains the results to be tone mapped.
	 *	                            The image's layout is expected to be GENERAL.
	 *	@param	aDestinationLdr		Destination image which shall receive the LDR color values after tone mapping.
	 *	                            The image's layout is expected to be GENERAL.
	 */
	void config(avk::queue& aQueue, avk::descriptor_cache aDescriptorCache, 
		avk::image_view aSourceHdr, avk::image_view aDestinationLdr)
	{
		using namespace avk;

		mQueue = &aQueue;
		mDescriptorCache = std::move(aDescriptorCache);
		mSourceHdr = std::move(aSourceHdr);
		mDestinationLdr = std::move(aDestinationLdr);

		// Create some helper images:s
		const auto w = mSourceHdr->get_image().width();
		const auto h = mSourceHdr->get_image().height();
		auto avgLumImg = context().create_image(w, h, vk::Format::eR16Sfloat, 1, memory_usage::device, image_usage::general_storage_image | image_usage::mip_mapped);
		auto maxLumImg = context().create_image(w, h, vk::Format::eR16Sfloat, 1, memory_usage::device, image_usage::general_storage_image | image_usage::mip_mapped);

		for (auto level = 0u; level < avgLumImg->create_info().mipLevels; level++) {
			mAvgLogLumLevels.push_back(std::move(context().create_image_view(avgLumImg, std::nullopt, {}, [&level](avk::image_view_t& aImageView) { aImageView.create_info().subresourceRange.setBaseMipLevel(level).setLevelCount(1u); })));
			mMaxLogLumLevels.push_back(std::move(context().create_image_view(maxLumImg, std::nullopt, {}, [&level](avk::image_view_t& aImageView) { aImageView.create_info().subresourceRange.setBaseMipLevel(level).setLevelCount(1u); })));
		}
		
		auto fen = context().record_and_submit_with_fence(command::gather(
			// Transition the passed images and the internal images into GENERAL layout and keep it in that layout forever:
			sync::image_memory_barrier(avgLumImg.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
			sync::image_memory_barrier(maxLumImg.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
			sync::image_memory_barrier(mDestinationLdr->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general)
		), *mQueue);
		fen->wait_until_signalled();
	}

	/**	Returns the result of the GPU timer query, which indicates how long the tone mapping approximately took.
	 */
	float duration()
	{
		return helpers::get_timing_interval_in_ms(std::format("tone mapping {}", mPingPong));
	}

	// Create all the compute pipelines used for the post processing effect(s),
	// prepare some command buffers with pipeline barriers to synchronize with subsequent commands,
	// create a new ImGui window that allows to enable/disable tone mapping, and to modify parameters:
	void initialize() override 
	{
		using namespace avk;

		// Create a command pool for allocating single-use (hence, transient) command buffers:
		mCommandPool = context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		constexpr auto initialLumData = lum_data{ 1.0f, 10.0f, glm::vec2{0.0f, 0.0f} };
		mLumBuffer = context().create_buffer(
			memory_usage::device, {},
			storage_buffer_meta::create_from_data(initialLumData)
		);
		auto fen = context().record_and_submit_with_fence({
			mLumBuffer->fill(&initialLumData, 0)
		}, *mQueue);
		fen->wait_until_signalled();

		mToLogPipeline = context().create_compute_pipeline_for(
			"shaders/to_log.comp",
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 1, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 2, 1u)
		);

		mMaxPipeline = context().create_compute_pipeline_for(
			"shaders/max_mipmap.comp",
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 1, 1u)
		);

		mUpdateLumBufferPipeline = context().create_compute_pipeline_for(
			"shaders/update_lum_buffer.comp",
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_data) },
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 1, 1u),
			descriptor_binding(1, 0, mLumBuffer)
		);

		mToneMappingPipeline = context().create_compute_pipeline_for(
			"shaders/tone_mapping.comp",
			push_constant_binding_data { shader_type::compute, 0, sizeof(push_constants_data) },
			descriptor_binding<image_view_as_sampled_image>(0, 0, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 1, 1u),
			descriptor_binding(1, 0, mLumBuffer)
		);

		// Use this invokee's updater to enable shader hot reloading
		mUpdater.emplace();
		mUpdater->on(shader_files_changed_event(mToLogPipeline.get()))
			.update(mToLogPipeline);
		mUpdater->on(shader_files_changed_event(mMaxPipeline.get()))
			.update(mMaxPipeline);
		mUpdater->on(shader_files_changed_event(mUpdateLumBufferPipeline.get()))
			.update(mUpdateLumBufferPipeline);
		mUpdater->on(shader_files_changed_event(mToneMappingPipeline.get()))
			.update(mToneMappingPipeline);
		mUpdater->on(swapchain_changed_event(context().main_window()))
			.invoke([this]() {
			const auto w = mSourceHdr->get_image().width();
			const auto h = mSourceHdr->get_image().height();
			auto avgLumImg = context().create_image_from_template(mAvgLogLumLevels[0]->get_image(), [w, h](avk::image_t& aImage) { aImage.create_info().extent.setWidth(w).setHeight(h); });
			auto maxLumImg = context().create_image_from_template(mMaxLogLumLevels[0]->get_image(), [w, h](avk::image_t& aImage) { aImage.create_info().extent.setWidth(w).setHeight(h); });

			for (auto& old : mAvgLogLumLevels) {
				context().main_window()->handle_lifetime(std::move(old));
			}
			mAvgLogLumLevels.clear();
			for (auto& old : mMaxLogLumLevels) {
				context().main_window()->handle_lifetime(std::move(old));
			}
			mMaxLogLumLevels.clear();

			for (auto level = 0u; level < avgLumImg->create_info().mipLevels; level++) {
				mAvgLogLumLevels.push_back(std::move(context().create_image_view(avgLumImg, std::nullopt, {}, [&level](avk::image_view_t& aImageView) { aImageView.create_info().subresourceRange.setBaseMipLevel(level).setLevelCount(1u); })));
				mMaxLogLumLevels.push_back(std::move(context().create_image_view(maxLumImg, std::nullopt, {}, [&level](avk::image_view_t& aImageView) { aImageView.create_info().subresourceRange.setBaseMipLevel(level).setLevelCount(1u); })));
			}

			auto fen = context().record_and_submit_with_fence(command::gather(
				sync::image_memory_barrier(avgLumImg.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				sync::image_memory_barrier(maxLumImg.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general)
			), *mQueue);
			fen->wait_until_signalled();
				});

		// Create a new ImGui window:
		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if(nullptr != imguiManager) {
			imguiManager->add_callback([this](){
				ImGui::Begin("Tone Mapping Settings");
				ImGui::SetWindowPos(ImVec2(295.0f, 305.0f), ImGuiCond_FirstUseEver);
				ImGui::SetWindowSize(ImVec2(220.0f, 134.0f), ImGuiCond_FirstUseEver);
				ImGui::Checkbox("enabled", &mToneMappingEnabled);
				ImGui::Checkbox("gradual", &mGradualAdaption);
				ImGui::SliderFloat("speed", &mAdaptionSpeed, 0.1f, 10.0f);
				ImGui::Checkbox("use max", &mUseMax);
				ImGui::SliderFloat("key", &mKey, 0.0f, 1.0f);

				ImGui::End();
			});
		}
		else {
			LOG_ERROR("Failed to init GUI, because composition does not contain an element of type avk::imgui_manager.");
		}
	}

	// Update the push constant data that will be used in render():
	void update() override 
	{
		using namespace avk;

		mPushConstants.mToneMappingEnabled = mToneMappingEnabled ? VK_TRUE : VK_FALSE;
		mPushConstants.mGradualAdaption    = mGradualAdaption    ? VK_TRUE : VK_FALSE;
		mPushConstants.mUseMax             = mUseMax             ? VK_TRUE : VK_FALSE;
		mPushConstants.mKey                = mKey;
		mPushConstants.mDeltaTime          = time().delta_time();
		mPushConstants.mAdaptionSpeed	   = mAdaptionSpeed;
	}

	// Create a new command buffer every frame, record instructions into it, and submit it to the graphics queue:
	void render() override 
	{
		using namespace avk;

		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		context().record({ // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):

			command::custom_commands([this](avk::command_buffer_t& cb) {

				// Note: mToneMappingEnabled is evaluated in shader code!

				mPingPong = 1 - mPingPong;
				helpers::record_timing_interval_start(cb.handle(), std::format("tone mapping {}", mPingPong));

				const auto w = mDestinationLdr->get_image().width();
				const auto h = mDestinationLdr->get_image().height();
				auto mipW = w;
				auto mipH = h;

				// To log (compute shader)
				cb.record(avk::command::bind_pipeline(mToLogPipeline.as_reference()));
				cb.record(avk::command::bind_descriptors(mToLogPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mSourceHdr->as_sampled_image(layout::general)),
					descriptor_binding(0, 1, mAvgLogLumLevels.front()->as_storage_image(layout::general)),
					descriptor_binding(0, 2, mMaxLogLumLevels.front()->as_storage_image(layout::general))
				})));
				cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);

				cb.record(sync::global_memory_barrier(
					// The compute shader must complete execution before the transfer may resume:
					stage::compute_shader        >> stage::transfer,
					// The compute's image writes must be made available and visible to the mipmapping's access (which can be read and write):
					access::shader_storage_write >> access::transfer_read | access::transfer_write
				));

				// Compute average through mip mapping (transfer operation)
				cb.record(
					mAvgLogLumLevels[0]->get_image().generate_mip_maps(layout::general >> layout::general)
				);

				// Sync the transfer operations from generate_mip_maps with subsequent compute:
				cb.record(sync::global_memory_barrier(
					stage::transfer >> stage::compute_shader,
					access::transfer_write >> access::shader_sampled_read
				));

				// Compute max (compute shader):
				if (mUseMax) {
					cb.record(avk::command::bind_pipeline(mMaxPipeline.as_reference()));
					for (size_t level = 1; level < mMaxLogLumLevels.size(); level++) {
						mipW /= 2u;
						mipH /= 2u;
						cb.record(avk::command::bind_descriptors(mMaxPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mMaxLogLumLevels[level - 1]->as_sampled_image(layout::general)),
							descriptor_binding(0, 1, mMaxLogLumLevels[level    ]->as_storage_image(layout::general))
						})));
						dispatch_and_sync_with_subsequent_compute(std::max(1u, mipW), std::max(1u, mipH), cb);
					}
				}

				// Update luminance buffer with new avg and max lum (compute shader):
				cb.record(avk::command::bind_pipeline(mUpdateLumBufferPipeline.as_reference()));
				cb.record(avk::command::bind_descriptors(mUpdateLumBufferPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mAvgLogLumLevels.back()->as_sampled_image(layout::general)),
					descriptor_binding(0, 1, mMaxLogLumLevels.back()->as_sampled_image(layout::general)),
					descriptor_binding(1, 0, mLumBuffer)
				})));
				cb.record(avk::command::push_constants(mUpdateLumBufferPipeline->layout(), mPushConstants));
				dispatch_and_sync_with_subsequent_compute(1u, 1u, cb); // will dispatch one group - local_size = 1 instead of 16 doesn't matter

				// Invoke the tone mapping shader (compute shader):
				cb.record(avk::command::bind_pipeline(mToneMappingPipeline.as_reference()));
				cb.record(avk::command::bind_descriptors(mToneMappingPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mSourceHdr->as_sampled_image(layout::general)),
					descriptor_binding(0, 1, mDestinationLdr->as_storage_image(layout::general)),
					descriptor_binding(1, 0, mLumBuffer)
				})));
				cb.record(avk::command::push_constants(mToneMappingPipeline->layout(), mPushConstants));
				cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);

				helpers::record_timing_interval_end(cb.handle(), std::format("tone mapping {}", mPingPong));

				// Finally, sync with subsequent compute or transfer commands:
				cb.record(sync::global_memory_barrier(
					// Compute must complete before other compute or transfer operations may proceed:
					stage::compute_shader        >> stage::compute_shader | stage::transfer,
					// Particularly write accesses must be made available, and we just assume here that next access for resources will be read:
					access::shader_storage_write >> access::shader_read | access::transfer_read
				));
			}),
		}) // End of command recording
		.into_command_buffer(cmdBfr)
			.then_submit_to(*mQueue)
			.submit();

		// Use a convenience function of avk::window to take care of the command buffer's lifetime:
		// It will get deleted in the future after #concurrent-frames have passed by.
		context().main_window()->handle_lifetime(std::move(cmdBfr));
	}


private:
	static void dispatch_and_sync_with_subsequent_compute(uint32_t x, uint32_t y, avk::command_buffer_t& cb) {
		using namespace avk;
		cb.handle().dispatch((x + 15u) / 16u, (y + 15u) / 16u, 1);
		cb.record(sync::global_memory_barrier(
			// The compute shader must complete execution before the subsequent compute shader may proceed:
			stage::compute_shader        >> stage::compute_shader,
			// The compute's image writes must be made available and visible to subsequent read access:
			access::shader_storage_write >> access::shader_read
		));
	}

	// Settings which can be modified via ImGui:
	bool mToneMappingEnabled = true;
	bool mGradualAdaption = true;
	bool mUseMax = true;
	float mKey = 0.18f;
	float mAdaptionSpeed = 5.0f;

	/** One single queue to submit all the commands to: */
	avk::queue* mQueue;

	/** One descriptor cache to use for allocating all the descriptor sets from: */
	avk::descriptor_cache mDescriptorCache;

	/** A command pool for allocating (single-use) command buffers from: */
	avk::command_pool mCommandPool;

	int mPingPong{ 1 };

	/** Source/input image view in HDR: */
	avk::image_view mSourceHdr;
	/** Destination/output image view in LDR: */
	avk::image_view mDestinationLdr;

	// Internal image views:
	std::vector<avk::image_view> mAvgLogLumLevels;
	std::vector<avk::image_view> mMaxLogLumLevels;

	avk::buffer mLumBuffer;

	avk::compute_pipeline mToLogPipeline;
	avk::compute_pipeline mMaxPipeline;
	avk::compute_pipeline mUpdateLumBufferPipeline;
	avk::compute_pipeline mToneMappingPipeline;

	push_constants_data mPushConstants;
};

