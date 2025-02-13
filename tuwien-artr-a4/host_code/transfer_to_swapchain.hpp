#include "imgui_utils.h"

// This class copies or blits the given images to the swap chain images for further processing
class transfer_to_swapchain : public avk::invokee
{
public:
	enum struct transfer_type { copy, blit };

	/**	An invokee which just copies images into the swap chain images
	 */
	transfer_to_swapchain() : invokee("Transfer to swapchain", true)
	{}

	// Execution order of 99 => execute after all the post processing effects, but before gizmos and UI
	int execution_order() const override { return 99; }

	/**	Method to configure this invokee, intended to be invoked BEFORE this invokee's invocation of initialize()
	 *	@param	aQueue				Stores an avk::queue* internally for future use, which has been created previously.
	 *	@param	aSourceDepth		Depth image that shall be transferred to the swap chain's depth image.
	 *	@param	aDepthTransferType	Indicates by which operation (copy or blit) the transfer of the depth image shall happen.
	 *	@param	aDepthImageLayouts	Describes in which layout the depth image comes in, and which layout it shall be transitioned into after the transfer operation.
	 *	@param	aSourceColor		Color image that shall be transferred to the swap chain's color image.
	 *	@param	aColorTransferType	Indicates by which operation (copy or blit) the transfer of the color image shall happen.
	 *	@param	aColorImageLayouts	Describes in which layout the color image comes in, and which layout it shall be transitioned into after the transfer operation.
	 */
	void config(avk::queue& aQueue,
		avk::image_view aSourceDepth, transfer_type aDepthTransferType, avk::layout::image_layout_transition aDepthImageLayouts,
		avk::image_view aSourceColor, transfer_type aColorTransferType, avk::layout::image_layout_transition aColorImageLayouts,
		std::optional<std::tuple<avk::image_view, transfer_type, avk::layout::image_layout_transition>> aIntermediateColorImage = {}
	)
	{
		using namespace avk;

		mQueue = &aQueue;
		mSrcDepth = std::move(aSourceDepth);
		mDepthTransferType = aDepthTransferType;
		mDepthImageLayouts = aDepthImageLayouts;

		mSrcColor = std::move(aSourceColor);
		mColorTransferType = aColorTransferType;
		mColorImageLayouts = aColorImageLayouts;

		if (aIntermediateColorImage.has_value()) {
			auto& tpl = aIntermediateColorImage.value();
			mIntermediateColorImage = std::move(std::get<avk::image_view>(tpl));
			mIntermediateColorTransferType = std::get<transfer_type>(tpl);
			mIntermediateColorImageLayouts = std::get<avk::layout::image_layout_transition>(tpl);
		}
	}

	void initialize() override
	{
		generate_command_buffers();

		// create an updater, and recreate the command buffers on resize (needed for window resize)
		mUpdater.emplace();
		mUpdater->on(avk::swapchain_changed_event(avk::context().main_window())).invoke([this]() {
			generate_command_buffers();
		});
	}

	// Create a new command buffer every frame, record instructions into it, and submit it to the graphics queue:
	void render() override 
	{
		using namespace avk;
		mQueue->submit(mCommandBuffers[context().main_window()->current_image_index()].as_reference());
	}

private:

	void generate_command_buffers() 
	{
		using namespace avk;

		if (!mCommandPool.has_value())
		{
			// Create a command pool for allocating reusable command buffers
			mCommandPool = context().create_command_pool(mQueue->family_index());
		}

		const auto numSwapchainImages = static_cast<uint32_t>(context().main_window()->number_of_swapchain_images());
		
		// Create the command buffer:
		mCommandBuffers = mCommandPool->alloc_command_buffers(numSwapchainImages);

		for (uint32_t i = 0; i < numSwapchainImages; ++i) {
			context().record(command::gather(
				// Record a bunch of commands (which can be a mix of state-type commands and action-type commands):
				// Transition the layouts of
				//  - The G-Buffer image that contains the final rendered color result
				//  - The backbuffer's color attachment image
				//  - The G-Buffer image that contains the final depth buffer
				//  - The backbuffer's depth attachment image
				// in preparation for copying the data from the G-Buffer images into the backbuffer images:
				sync::image_memory_barrier(mSrcColor->get_image(),
					stage::color_attachment_output | stage::compute_shader   >>  stage::transfer,
					access::color_attachment_write | access::shader_write    >>  access::transfer_read
				).with_layout_transition(mColorImageLayouts.mOld >> layout::transfer_src),

				sync::image_memory_barrier(context().main_window()->backbuffer_at_index(i)->image_at(0), // Color attachment
					stage::color_attachment_output | stage::compute_shader >> stage::transfer,
					access::color_attachment_write | access::shader_write >>  access::transfer_write
				).with_layout_transition(layout::undefined >> layout::transfer_dst), // Don't care about the previous layout

				sync::image_memory_barrier(mSrcDepth->get_image(),
					stage::early_fragment_tests | stage::late_fragment_tests >>  (mDepthTransferType == transfer_type::blit ? stage::blit : stage::copy),
					access::depth_stencil_attachment_write                   >>  access::transfer_read
				).with_layout_transition(mDepthImageLayouts.mOld >> layout::transfer_src),

				sync::image_memory_barrier(context().main_window()->backbuffer_at_index(i)->image_at(1), // Depth attachment
					stage::color_attachment_output | stage::compute_shader >> (mDepthTransferType == transfer_type::blit ? stage::blit : stage::copy),
					access::color_attachment_write | access::shader_write  >>  access::transfer_write
				).with_layout_transition(layout::depth_stencil_attachment_optimal >> layout::transfer_dst),

				command::conditional([this]() { return mIntermediateColorImage.has_value(); },
					[this, i]() { return command::gather(
						sync::image_memory_barrier(mIntermediateColorImage->get_image(),
							stage::color_attachment_output | stage::compute_shader   >>  stage::transfer,
							access::color_attachment_write | access::shader_write    >>  access::transfer_write
						).with_layout_transition(mIntermediateColorImageLayouts.mOld >> layout::general),

						// Color -> intermediate
						command::conditional([this]() { return mColorTransferType == transfer_type::blit; },
							[this]() {
								return blit_image(
									mSrcColor->get_image(), layout::transfer_src,
									mIntermediateColorImage->get_image(), layout::general,
									vk::ImageAspectFlagBits::eColor
								);
							},
							[this]() {
								return copy_image_to_another(
									mSrcColor->get_image(), layout::transfer_src,
									mIntermediateColorImage->get_image(), layout::general,
									vk::ImageAspectFlagBits::eColor
								);
							}
						),

						sync::image_memory_barrier(mIntermediateColorImage->get_image(),
							stage::transfer         >>  stage::transfer,
							access::transfer_write  >>  access::transfer_read
						), // leave it in general layout

						// Transfer color from intermediate:
						command::conditional([this]() { return this->mIntermediateColorTransferType == transfer_type::blit; },
							[this, i]() {
								return blit_image(
									mIntermediateColorImage->get_image(), layout::general,
									context().main_window()->backbuffer_at_index(i)->image_at(0), layout::transfer_dst,
									vk::ImageAspectFlagBits::eColor
								);
							},
							[this, i]() {
								return copy_image_to_another(
									mIntermediateColorImage->get_image(), layout::general,
									context().main_window()->backbuffer_at_index(i)->image_at(0), layout::transfer_dst,
									vk::ImageAspectFlagBits::eColor
								);
							}
						)
					); },
					[this, i]() { return command::gather(
						// Transfer color:
						command::conditional([this]() { return this->mColorTransferType == transfer_type::blit; },
							[this, i]() {
								return blit_image(
									mSrcColor->get_image(), layout::transfer_src,
									context().main_window()->backbuffer_at_index(i)->image_at(0), layout::transfer_dst,
									vk::ImageAspectFlagBits::eColor
								);
							},
							[this, i]() {
								return copy_image_to_another(
									mSrcColor->get_image(), layout::transfer_src,
									context().main_window()->backbuffer_at_index(i)->image_at(0), layout::transfer_dst,
									vk::ImageAspectFlagBits::eColor
								);
							}
						)
					); }
				),

				// Transfer depth:
				command::conditional([this]() { return this->mDepthTransferType == transfer_type::blit; },
					[this, i]() {
						return blit_image(
							mSrcDepth->get_image(), layout::transfer_src,
							context().main_window()->backbuffer_at_index(i)->image_at(1), layout::transfer_dst,
							vk::ImageAspectFlagBits::eDepth
						);
					},
					[this, i]() {
						return copy_image_to_another(
							mSrcDepth->get_image(), layout::transfer_src,
							context().main_window()->backbuffer_at_index(i)->image_at(1), layout::transfer_dst,
							vk::ImageAspectFlagBits::eDepth
						);
					}
				),

				// Transition all the layouts again into whichever layouts we are going to need them next:
				sync::image_memory_barrier(mSrcColor->get_image(),
					stage::transfer                                         >>  stage::fragment_shader,
					access::transfer_write /* layout transition = write */  >> access::shader_sampled_read
				).with_layout_transition(layout::transfer_src >> mColorImageLayouts.mNew),

				sync::image_memory_barrier(context().main_window()->backbuffer_at_index(i)->image_at(0), // Color attachment
					stage::transfer         >>  stage::color_attachment_output,
					access::transfer_write  >>  access::color_attachment_write
				).with_layout_transition(layout::transfer_dst >> layout::color_attachment_optimal),

				sync::image_memory_barrier(mSrcDepth->get_image(),
					(mDepthTransferType == transfer_type::blit ? stage::blit : stage::copy)  >>  stage::fragment_shader,
					access::none                                                             >>  access::shader_sampled_read
				).with_layout_transition(layout::transfer_src >> mDepthImageLayouts.mNew),

				sync::image_memory_barrier(context().main_window()->backbuffer_at_index(i)->image_at(1), // Depth attachment
					(mDepthTransferType == transfer_type::blit ? stage::blit : stage::copy)  >>  stage::early_fragment_tests | stage::late_fragment_tests,
					access::transfer_write                                                   >>  access::depth_stencil_attachment_read | access::depth_stencil_attachment_write
				).with_layout_transition(layout::transfer_dst >> layout::depth_stencil_attachment_optimal),
	
				sync::global_memory_barrier(stage::transfer        >> stage::fragment_shader,
				                            access::transfer_write >> access::shader_sampled_read)

			)) // End of command recording
			.into_command_buffer(mCommandBuffers[i]);
		}
	}

private:
	avk::queue* mQueue;
	avk::command_pool mCommandPool;
	std::vector<avk::command_buffer> mCommandBuffers;

	avk::image_view mSrcDepth;
	transfer_type mDepthTransferType;
	avk::layout::image_layout_transition mDepthImageLayouts;

	avk::image_view mSrcColor;
	transfer_type mColorTransferType;
	avk::layout::image_layout_transition mColorImageLayouts;

	avk::image_view mIntermediateColorImage;
	transfer_type mIntermediateColorTransferType;
	avk::layout::image_layout_transition mIntermediateColorImageLayouts;
};

