#pragma once
#include "timer_interface.hpp"

class hole_checker : public avk::invokee
{
public:
	struct HoleInformation {
		float lastHoleFoundTime;
		glm::vec3 cameraLocation;
		glm::quat cameraRotation;
	};

	/** Can be invoked from within a render pass. It records a clear attachments
	 *	command into the given command buffer.
	 *	@param	aCommandBuffer	Command buffer to record into, must be in recording state.
	 */
	static void clear_to_red(avk::command_buffer_t& aCommandBuffer)
	{
		const std::array<float, 4> red = { 1.0f, 0.0f, 0.0f, 0.0f };
		vk::ClearAttachment clearAtt = vk::ClearAttachment(vk::ImageAspectFlagBits::eColor, 0u, vk::ClearValue(vk::ClearColorValue(red)));
		auto clearRect = vk::ClearRect(vk::Rect2D({}, avk::context().main_window()->swap_chain_extent()), 0u, 1u);
		aCommandBuffer.handle().clearAttachments(1, &clearAtt, 1u, &clearRect);
	}

	/** Constructor
	 *	@param	aQueue	Stores an avk::queue* internally for future use, which has been created previously.
	 */
	hole_checker(avk::queue& aQueue)
		: mQueue{ &aQueue }
	{
		avk::invokee::disable(); // Disable myself by default.
	}

	// invoke this after assignment2 (which has execution order 0)
	int execution_order() const override { return 10; }

	HoleInformation get_hole_information() { return mHoleInfo; }

	void initialize() override
	{
		using namespace avk;
		
		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = context().create_descriptor_cache();

		// Create a command pool for allocating single-use (hence, transient) command buffers:
		mCommandPool = context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);

		auto* window = context().main_window();
		const auto resolution = window->swap_chain_extent();

		auto numfif = window->number_of_frames_in_flight();
		for (decltype(numfif) fif = 0; fif < numfif; ++fif)
		{
			auto view = context().create_image_view(
				context().create_image(
					resolution.width, resolution.height, IMAGE_FORMAT, 1, 
					memory_usage::device, image_usage::shader_storage | image_usage::transfer_source | image_usage::transfer_destination
				)
			);
			mImageView.push_back(std::move(view));
			mImageViewValid.push_back(true);

			auto buf = context().create_buffer(memory_usage::host_visible, {}, storage_buffer_meta::create_from_size(sizeof(VkBool32)));
			mResultBuffer.push_back(std::move(buf));
		}

		mPipeline = context().create_compute_pipeline_for(
			compute_shader("shaders/utils/hole_checker.comp.spv"),
			descriptor_binding(0, 0, mImageView[0]->as_storage_image(layout::general)),
			descriptor_binding(0, 1, mResultBuffer[0]->as_storage_buffer())
		);

		mUpdater.emplace();
		mUpdater->on(swapchain_changed_event(window))
			.invoke([this]() {
				// mark all our image views invalid
				for (auto i = 0; i < mImageViewValid.size(); ++i) {
					mImageViewValid[i] = false;
				}
			});
			// no need to update the pipeline
	}

	void render() override
	{
		using namespace avk;

		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		
		auto* window = avk::context().main_window();
		const auto frameIndex = window->current_in_flight_index();

		// our work images may be invalid due to a swapchain resize - recreate them if necessary
		if (!mImageViewValid[frameIndex]) {
			LOG_DEBUG(std::format("Hole checker needs to recreate working image {}", frameIndex));
			auto resolution = window->swap_chain_extent();
			auto view = context().create_image_view(
				context().create_image(
					resolution.width, resolution.height, IMAGE_FORMAT, 1, 
					memory_usage::device, image_usage::general_color_attachment | image_usage::shader_storage
				)
			);
			mImageView[frameIndex] = std::move(view);
			mImageViewValid[frameIndex] = true;
		}

		auto& backbufferAttachment = window->current_image_reference();
		auto& auxiliaryImage = mImageView[frameIndex]->get_image();
		const auto w = auxiliaryImage.width();
		const auto h = auxiliaryImage.height();
		const VkBool32 falseBufferValue = VK_FALSE;

		auto fen = context().create_fence();

		context().record({
				// Barrier & layout transition for src image:
				sync::image_memory_barrier(backbufferAttachment, // <-- this is one of the window's backbuffer images
					stage::color_attachment_output >> stage::blit,
					access::color_attachment_write >> access::transfer_read
				).with_layout_transition(layout::color_attachment_optimal >> layout::transfer_src),

				// Barrier & layout transition for dst image:
				sync::image_memory_barrier(auxiliaryImage, // <-- auxiliary image, we don't care about its previous contents (hence, none/none/undefined)
					stage::none                    >> stage::blit,
					access::none                   >> access::transfer_write
				).with_layout_transition(layout::undefined >> layout::transfer_dst),

				// BLIT from backbufferAttachment -> auxiliaryImage
				blit_image(
					backbufferAttachment, layout::transfer_src,
					auxiliaryImage,       layout::transfer_dst
				),

				// Clear the result buffer:
				mResultBuffer[frameIndex]->fill(&falseBufferValue, 0),

				// Barrier & layout transition for our auxiliary (a.k.a. dst) image before used in compute shader:
				sync::image_memory_barrier(auxiliaryImage, 
					stage::blit            >> stage::compute_shader,
					access::transfer_write >> access::shader_storage_read | access::shader_storage_write
				).with_layout_transition(layout::transfer_dst >> layout::general),

				// Invoke the compute shader (currently no command:: functionality implemented... but... SOON)
				command::custom_commands([frameIndex, w, h, this](avk::command_buffer_t& cb) {
					cb.record(avk::command::bind_pipeline(mPipeline.as_reference()));
					cb.record(avk::command::bind_descriptors(mPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
						descriptor_binding(0, 0, mImageView[frameIndex]->as_storage_image(layout::general)),
						descriptor_binding(0, 1, mResultBuffer[frameIndex]->as_storage_buffer())
					})));
					cb.handle().dispatch((w + 15u) / 16u, (h + 15u) / 16u, 1);
				}),

				// Prepare layout for transitioning the results back into it:
				sync::image_memory_barrier(backbufferAttachment,
					stage::blit                   >> stage::blit,           // <-- previous operation on this image was a blit
					access::none                  >> access::transfer_write // <-- just read access before => no need to make anything available
				).with_layout_transition(layout::transfer_src >> layout::transfer_dst),

				// Prepare layout for transitioning the results back from it:
				sync::image_memory_barrier(auxiliaryImage,
					stage::compute_shader         >> stage::blit,          // <-- previous operation on this image was compute shader
					access::shader_storage_write  >> access::transfer_read // <-- as soon as storage writes are done, we can start to blit
				).with_layout_transition(layout::general >> layout::transfer_src),

				// BLIT from auxiliaryImage -> backbufferAttachment
				blit_image(
					auxiliaryImage,       layout::transfer_src,
					backbufferAttachment, layout::transfer_dst
				),

				// Turn backbuffer attachment image back into a color attachment image, because afterwards, we are going to continue render stuff into it (gizmos, ImGui)
				sync::image_memory_barrier(backbufferAttachment,
					stage::blit             >> stage::color_attachment_output,
					access::transfer_write  >> access::color_attachment_write
				).with_layout_transition(layout::transfer_dst >> layout::color_attachment_optimal),

			})
			.into_command_buffer(cmdBfr)
			.then_submit_to(*mQueue)
			.signaling_upon_completion(fen)
			.submit();

		fen->wait_until_signalled();
		// No need for the following when using a fence:
		//
		//// Use a convenience function of avk::window to take care of the command buffer's lifetime:
		//// It will get deleted in the future after #concurrent-frames have passed by.
		//context().main_window()->handle_lifetime(owned(cmdBfr));

		auto result = mResultBuffer[frameIndex]->read<VkBool32>(0);
		if (result) {
			mHoleInfo.lastHoleFoundTime = avk::time().absolute_time();
			auto cam = current_composition()->element_by_type<quake_camera>();
			if (cam != nullptr) {
				mHoleInfo.cameraLocation = cam->translation();
				mHoleInfo.cameraRotation = cam->rotation();
			}
		}
	}

private:
	/** One single queue to submit all the commands to: */
	avk::queue* mQueue;

	/** One descriptor cache to use for allocating all the descriptor sets from: */
	avk::descriptor_cache mDescriptorCache;

	/** A command pool for allocating (single-use) command buffers from: */
	avk::command_pool mCommandPool;

	static const vk::Format IMAGE_FORMAT = vk::Format::eR16G16B16A16Sfloat; // vk::Format::eR8G8B8A8Unorm;
	std::vector<avk::image_view> mImageView;
	std::vector<bool> mImageViewValid;
	avk::compute_pipeline mPipeline;
	std::vector<avk::buffer> mResultBuffer;
	HoleInformation mHoleInfo = {};
};

