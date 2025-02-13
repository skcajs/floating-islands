#include "imgui_utils.h"

// This class handles the anti-aliasing post-processing effect(s).
class anti_aliasing : public avk::invokee
{
	struct push_constants_for_taa {
		glm::vec4 mJitterAndAlpha;
	};

	struct matrices_for_taa {
		glm::mat4 mHistoryViewProjMatrix;
		glm::mat4 mInverseViewProjMatrix;
	};

public:
	anti_aliasing() : invokee("Anti-Aliasing Post Processing Effect", true)
	{ }

	// Execute after tone mapping and before transfer_to_swapchain:
	int execution_order() const override { return 80; }

	// Compute an offset for the projection matrix based on the given frame-id
	glm::vec2 get_jitter_offset_for_frame(int64_t aFrameId) const
	{
		using namespace avk;

		//
		// TODO Task 6: Compute and return an offset for the given frame-id!
		//              This offset is used in get_jittered_projection_matrix to offset the projection matrix (see below).
		//
		return {0.0f, 0.0f};
	}

	void save_view_matrix_and_modify_projection_matrix() {
		// we will reset the projection matrix at the end of render()

		auto camera = avk::current_composition()->element_by_type<avk::quake_camera>();

		mViewMatrixLast = mViewMatrixCurrent;
		mProjMatrixLast = mProjMatrixCurrent;
		mViewMatrixCurrent = camera->view_matrix();
		mProjMatrixCurrent = camera->projection_matrix();

		// now modify the projection matrix
		auto modifiedProjMat = get_jittered_projection_matrix(mProjMatrixCurrent, avk::context().main_window()->current_frame());
		mProjMatrixToRestore = mProjMatrixCurrent;
		camera->set_projection_matrix(modifiedProjMat, avk::projection_type::perspective);
	}

	// Applies a translation to the given matrix and returns the result
	glm::mat4 get_jittered_projection_matrix(glm::mat4 aProjMatrix, int64_t aFrameId) const
	{
		if (mTaaEnabled) {
			const auto xyOffset = get_jitter_offset_for_frame(aFrameId);
			return glm::translate(glm::vec3{xyOffset.x, xyOffset.y, 0.0f}) * aProjMatrix;
		}
		else {
			return aProjMatrix;
		}
	}

	/**	Method to configure this invokee, intended to be invoked BEFORE this invokee's invocation of initialize()
	 *	@param	aQueue					Stores an avk::queue* internally for future use, which has been created previously.
	 *	@param	aDescriptorCache		A descriptor cache that shall be used (possibly allowing descriptor re-use from other invokees)
	 *	@param	aUniformsBuffer			A buffer containing user input and current frame's data
	 *	@param	aSourceColorImageView	Input image in LDR format which contains the results to be anti-aliased.
	 *									The image's layout is expected to be GENERAL.
	 *	@param	aSourceDepthImageView	G-Buffer depth values associated to the color values in aSourceColor
	 *	@param	aDestinationImageView	Destination image which shall receive the anti-aliased results.
	 *									The image's layout is expected to be GENERAL.
	 */
	void config(avk::queue& aQueue, avk::descriptor_cache aDescriptorCache, avk::buffer aUniformsBuffer,
		avk::image_view aSourceColorImageView, avk::image_view aSourceDepthImageView, avk::image_view aDestinationImageView)
	{
		using namespace avk;

		mQueue = &aQueue;
		mDescriptorCache = std::move(aDescriptorCache);
		mUniformsBuffer = std::move(aUniformsBuffer);
		mSourceColorImageView = std::move(aSourceColorImageView);
		mSourceDepthImageView = std::move(aSourceDepthImageView);
		mDestinationImageView = std::move(aDestinationImageView);

		// Create some helper images:
		mHistoryColorImageView = context().create_image_view_from_template(mSourceColorImageView.as_reference());
		mHistoryDepthImageView = context().create_image_view_from_template(mSourceDepthImageView.as_reference());

		auto fen = context().record_and_submit_with_fence(command::gather(
			// Transition images into GENERAL layout
			sync::image_memory_barrier(mHistoryColorImageView->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
			sync::image_memory_barrier(mHistoryDepthImageView->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general)
		), *mQueue);
		fen->wait_until_signalled();
	}

	/**	Returns the result of the GPU timer query, which indicates how long TAA approximately took.
	 */
	float duration()
	{
		if (!mTaaEnabled) {
			return 0.0f;
		}
		return helpers::get_timing_interval_in_ms(std::format("TAA {}", mPingPong));
	}

	// Create all the compute pipelines used for the post processing effect(s),
	// prepare some command buffers with pipeline barriers to synchronize with subsequent commands,
	// create a new ImGui window that allows to enable/disable anti-aliasing, and to modify parameters:
	void initialize() override 
	{
		using namespace avk;
		
		// Create a command pool for allocating single-use (hence, transient) command buffers:
		mCommandPool = context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		mSampler = context().create_sampler(filter_mode::bilinear, border_handling_mode::clamp_to_border, 0);

		mMatricesBuffer = context().create_buffer(
			memory_usage::host_coherent, {},
			uniform_buffer_meta::create_from_size(sizeof(matrices_for_taa))
		);
		
		mTaaPipeline = context().create_compute_pipeline_for(
			"shaders/taa.comp",
			push_constant_binding_data { shader_type::compute, 0, sizeof(push_constants_for_taa) },
			descriptor_binding(0, 0, mSampler),
			descriptor_binding<image_view_as_sampled_image>(0, 1, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 2, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 3, 1u),
			descriptor_binding<image_view_as_sampled_image>(0, 4, 1u),
			descriptor_binding<image_view_as_storage_image>(0, 5, 1u),
			descriptor_binding(1, 0, mMatricesBuffer)
		);

		// Use this invokee's updater to enable shader hot reloading
		mUpdater.emplace();
		mUpdater->on(shader_files_changed_event(mTaaPipeline.as_reference()))
			.update(mTaaPipeline);

		// Create a new ImGui window:
		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if(nullptr != imguiManager) {
			imguiManager->add_callback([this](){
				ImGui::Begin("Anti-Aliasing Settings");
				ImGui::SetWindowPos(ImVec2(295.0f, 449.0f), ImGuiCond_FirstUseEver);
				ImGui::SetWindowSize(ImVec2(220.0f, 86.0f), ImGuiCond_FirstUseEver);
				ImGui::Checkbox("enabled", &mTaaEnabled);
				ImGui::SliderFloat("alpha", &mAlpha, 0.0f, 1.0f);
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
	}

	// Create a new command buffer every frame, record instructions into it, and submit it to the graphics queue:
	void render() override
	{
		using namespace avk;

		auto* mainWnd = context().main_window();
		const auto frameId = mainWnd->current_frame();
		const auto lastFrameId = frameId - 1;

		mPingPong = 1 - mPingPong; // current index

		bool historyIsValid = (mHistoryCreatedFromFrameId == lastFrameId);

		const auto jitter = get_jitter_offset_for_frame(frameId);
		mTaaPushConstants.mJitterAndAlpha = glm::vec4(jitter.x, jitter.y, 0.0f, mAlpha);

		// fill matrices UBO
		matrices_for_taa matrices;
		matrices.mInverseViewProjMatrix = glm::inverse(mProjMatrixCurrent * mViewMatrixCurrent);
		matrices.mHistoryViewProjMatrix = mProjMatrixLast * mViewMatrixLast;

		mMatricesBuffer->fill(&matrices, {}); // Host-coherent buffer => returned action_type_command will be empty.

		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		context().record({ // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):

			command::custom_commands([this, frameId, historyIsValid](avk::command_buffer_t& cb) {

				if (mTaaEnabled) {
					// ---------------------- If Anti-Aliasing is enabled perform the following actions --------------------------

					helpers::record_timing_interval_start(cb.handle(), std::format("TAA {}", mPingPong));

					if (historyIsValid) {

						const auto w = mDestinationImageView->get_image().width();
						const auto h = mDestinationImageView->get_image().height();

						cb.record(sync::global_memory_barrier(
							stage::transfer >> stage::compute_shader,
							access::transfer_write >> access::shader_read
						));

						// Apply temporal anti-aliasing:
						cb.record(avk::command::bind_pipeline(mTaaPipeline.as_reference()));
						cb.record(avk::command::bind_descriptors(mTaaPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mSampler),
							descriptor_binding(0, 1, mSourceColorImageView->as_sampled_image(layout::general)),
							descriptor_binding(0, 2, mSourceDepthImageView->as_sampled_image(layout::shader_read_only_optimal)),
							descriptor_binding(0, 3, mHistoryColorImageView->as_sampled_image(layout::general)),
							descriptor_binding(0, 4, mHistoryDepthImageView->as_sampled_image(layout::general)),
							descriptor_binding(0, 5, mDestinationImageView->as_storage_image(layout::general)),
							descriptor_binding(1, 0, mMatricesBuffer)
						})));
						cb.record(avk::command::push_constants(mTaaPipeline->layout(), mTaaPushConstants));
						cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
					}
					else {
						// History was not valid => Copy source color to destination:
						cb.record(copy_image_to_another(mSourceColorImageView->get_image(), layout::general, mDestinationImageView->get_image(), layout::general));
					}

					// Copy into history images, and record the appropriate instructions into cb:
					copy_depth_image_into_history_image(cb);
					copy_color_image_into_history_image(cb);

					mHistoryCreatedFromFrameId = frameId; // history is now valid for the next frame

					helpers::record_timing_interval_end(cb.handle(), std::format("TAA {}", mPingPong));
				}
				else {
					// Just copy-over unmodified:
					cb.record(copy_image_to_another(mSourceColorImageView->get_image(), layout::general, mDestinationImageView->get_image(), layout::general));
				}

				// In any case, sync with subsequent compute or transfer commands:
				cb.record(sync::global_memory_barrier(
					// Compute or transfer operations must complete before other compute or transfer operations may proceed:
					stage::compute_shader | stage::transfer               >> stage::compute_shader | stage::transfer,
					// Particularly write accesses must be made available, and we just assume here that next access for resources will be read:
					access::shader_storage_write | access::transfer_write >> access::shader_read | access::transfer_read
				));
			}),

		}) // End of command recording
		.into_command_buffer(cmdBfr)
			.then_submit_to(*mQueue)
			.submit();

		// Use a convenience function of avk::window to take care of the command buffer's lifetime:
		// It will get deleted in the future after #concurrent-frames have passed by.
		context().main_window()->handle_lifetime(std::move(cmdBfr));

		// restore the camera's projection matrix to the state it was before save_view_matrix_and_modify_projection_matrix() was called
		current_composition()->element_by_type<quake_camera>()->set_projection_matrix(mProjMatrixToRestore, avk::projection_type::perspective);
	}

private:
	/**	Helper function which copies the current mSourceDepthImageView into the mHistoryDepthImageView, 
	 *	so that we can access this frame's depth information in the next frame.
	 *	@param	cb		Reference to a command buffer where to record the appropriate instructions into.
	 */
	void copy_depth_image_into_history_image(avk::command_buffer_t& cb)
	{
		using namespace avk;
		cb.record(	sync::image_memory_barrier(mSourceDepthImageView->get_image(),
						stage::compute_shader >> stage::copy,           // Must wait for the compute shader to finish before starting to transition the layout.
						access::none          >> access::transfer_read
					).with_layout_transition(layout::shader_read_only_optimal >> layout::transfer_src));
		cb.record(	sync::image_memory_barrier(mHistoryDepthImageView->get_image(),
						stage::compute_shader >> stage::copy,           // Must wait for the compute shader to finish before starting to write into.
						access::none          >> access::transfer_write
					));

		cb.record(copy_image_to_another(mSourceDepthImageView->get_image(), layout::transfer_src, mHistoryDepthImageView->get_image(), layout::general, vk::ImageAspectFlagBits::eDepth));

		cb.record(	sync::image_memory_barrier(mSourceDepthImageView->get_image(), 
						stage::copy    >>  stage::early_fragment_tests | stage::late_fragment_tests,
						access::none   >>  access::depth_stencil_attachment_read
					).with_layout_transition(layout::transfer_src >> layout::shader_read_only_optimal));
	}

	/**	Helper function which copies the current mDestinationImageView into the mHistoryColorImageView,
	 *	so that we can access this frame's color information in the next frame.
	 *	@param	cb		Reference to a command buffer where to record the appropriate instructions into.
	 */
	void copy_color_image_into_history_image(avk::command_buffer_t& cb)
	{
		using namespace avk;

		// Ensure that writes to mDestinationImageView (be it through compute or copy) have finished and are visible to the subsequent copy operation:
		cb.record(sync::image_memory_barrier(mDestinationImageView->get_image(),
			stage::compute_shader        | stage::copy            >> stage::copy,
			access::shader_storage_write | access::transfer_write >> access::transfer_read
		));

		cb.record(	sync::image_memory_barrier(mHistoryColorImageView->get_image(),
						stage::compute_shader >> stage::copy,           // Must wait for the compute shader to finish before starting to write into.
						access::none          >> access::transfer_write
					));

		// Copy result to history color image:
		cb.record(copy_image_to_another(mDestinationImageView->get_image(), layout::general, mHistoryColorImageView->get_image(), layout::general));
	}

	// Settings, which can be modified via ImGui:
	bool mTaaEnabled = true;
	float mAlpha = 0.1f;

	/** One single queue to submit all the commands to: */
	avk::queue* mQueue;

	/** One descriptor cache to use for allocating all the descriptor sets from: */
	avk::descriptor_cache mDescriptorCache;

	/** A command pool for allocating (single-use) command buffers from: */
	avk::command_pool mCommandPool;

	int mPingPong{ 1 };

	/** Source/input image view in LDR: */
	avk::image_view mSourceColorImageView;
	/** Source/input depth image view: */
	avk::image_view mSourceDepthImageView;
	/** Destination/output image view in LDR: */
	avk::image_view mDestinationImageView;
	// Buffer containing the user input and matrices:
	avk::buffer mUniformsBuffer;

	avk::image_view mHistoryColorImageView;
	avk::image_view mHistoryDepthImageView;

	// For each history frame's image content, also store the associated projection matrix:
	// std::array<glm::mat4, 2> mHistoryProjMatrices;
	// std::array<glm::mat4, 2> mHistoryViewMatrices;
	glm::mat4 mProjMatrixLast, mProjMatrixCurrent;
	glm::mat4 mViewMatrixLast, mViewMatrixCurrent;
	glm::mat4 mProjMatrixToRestore;
	avk::buffer mMatricesBuffer;

	avk::window::frame_id_t mHistoryCreatedFromFrameId = 0;

	avk::sampler mSampler;
	avk::compute_pipeline mTaaPipeline;
	push_constants_for_taa mTaaPushConstants;
};

