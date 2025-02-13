#include "imgui_utils.h"
#include "ambient_occlusion.hpp"
#include "reflections.hpp"
#include "tone_mapping.hpp"
#include "anti_aliasing.hpp"
#include "transfer_to_swapchain.hpp"
#include <conversion_utils.hpp>

// TODO Bonus Tasks RTX ON Path: Uncomment the following line to turn RTX ON
//#define RTX_ON

class assignment4 : public avk::invokee
{
	// ------------------ Structs for transfering data from HOST -> DEVICE ------------------

	/** Struct definition for the vertex data, material index, and model matrix of draw calls.
	 *	In Assignment 4, we use the same approach as during Assignment 2, where
	 *	helpers::load_models_and_scenes_from_file does not return the already uploaded buffers,
	 *	but only the raw vertex data, and we must put them into buffers manually afterwards.
	 */
	struct draw_call
	{
		avk::buffer mIndexBuffer;
		avk::buffer mPositionsBuffer;
		avk::buffer mTexCoordsBuffer;
		avk::buffer mNormalsBuffer;
		avk::buffer mTangentsBuffer;
		avk::buffer mBitangentsBuffer;
		int mMaterialIndex;
		glm::mat4 mModelMatrix;
	};

#ifdef RTX_ON
	/**	A struct containing data that is used/need for the ray tracing pipelines
	 *	and ray tracing acceleration structures.
	 */
	struct rtx_data_per_draw_call
	{
		avk::buffer_view mIndexBufferView;
		avk::buffer_view mNormalsBufferView;
		avk::bottom_level_acceleration_structure mBottomLevelAS;
	};
#endif

	/** Struct definition for push constants used for the draw calls of the scene */
	struct push_constants_for_draw
	{
		glm::mat4 mModelMatrix;
		int mMaterialIndex;
	};

	/** Struct definition for data used as UBO across different pipelines, containing matrices and user input */
	struct matrices_and_user_input
	{
		// view matrix as returned from quake_camera
		glm::mat4 mViewMatrix;
		// projection matrix as returned from quake_camera
		glm::mat4 mProjMatrix;
		// Inverse of mProjMatrix:
		glm::mat4 mInverseProjMatrix;
		// transformation matrix which tranforms to camera's position
		glm::mat4 mCamPos;
		// x = tessellation factor, y = displacement strength, z = enable PN-triangles, w unused
		glm::vec4 mUserInput;
	};

	/** Struct definition for data used as UBO across different pipelines, containing lightsource data */
	struct lightsource_data
	{
		// x,y ... ambient light sources start and end indices; z,w ... directional light sources start and end indices
		glm::uvec4 mRangesAmbientDirectional;
		// x,y ... point light sources start and end indices; z,w ... spot light sources start and end indices
		glm::uvec4 mRangesPointSpot;
		// Contains all the data of all the active light sources
		std::array<avk::lightsource_gpu_data, MAX_NUMBER_OF_LIGHTSOURCES> mLightData;
	};

	// ----------------------------------------------------

public:
	/** Constructor
	 *	@param	aQueue	Stores an avk::queue* internally for future use, which has been created previously.
	 */
	assignment4(avk::queue& aQueue)
		: mQueue{ &aQueue }
		, mSkyboxSphere{ &aQueue }
	{
	}

	// ----------------------- vvv   INITIALIZATION   vvv -----------------------

	/**	Initialize callback is invoked by the framework at initialization time.
	 *	Here, all resources are created, such as pipelines, and buffers containing the
	 *	3D geometry---which is loaded from file and then into device buffers.
	 */
	void initialize() override
	{
		using namespace avk;

		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = context().create_descriptor_cache();

		// Create a command pool for allocating single-use (hence, transient) command buffers:
		mCommandPool = context().create_command_pool(mQueue->family_index(), vk::CommandPoolCreateFlagBits::eTransient);
		
		// Load 3D scenes/models from files:
		std::vector<helpers::data_for_draw_call> dataForDrawCalls;
		std::tie(mMaterials, mImageSamplers, dataForDrawCalls) = helpers::load_models_and_scenes_from_file({
			// Load a scene from file (path according to the Visual Studio filters!), and apply a transformation matrix (identity, here):
			  { "assets/sponza_and_terrain.fscene",                                 glm::mat4{1.0f} }
		}, mQueue);

#ifdef RTX_ON
		std::vector<avk::geometry_instance> geometryInstancesForTopLevelAS;
#endif

		std::vector<recorded_commands_t> commandsToBeExcecuted;

		// helpers::load_models_and_scenes_from_file returned only the raw vertex data.
		//  => Put them all into buffers which we can use during rendering:
		for (auto& data : dataForDrawCalls) {
			// Excluding one blue curtain (of a total of three) by modifying the loaded indices before uploading them to a GPU buffer:
			if (data.mModelName.find("sponza_fabric") != std::string::npos && data.mMeshName == "sponza_326") {
				data.mIndices.erase(std::begin(data.mIndices), std::begin(data.mIndices) + 3 * 4864);
			}

#ifdef RTX_ON
			auto [bufferPositions , commandsPositions ] = avk::create_buffer<decltype(data.mPositions) , vertex_buffer_meta, uniform_texel_buffer_meta, read_only_input_to_acceleration_structure_builds_buffer_meta>(data.mPositions , content_description::position);
			auto [bufferTexCoords , commandsTexCoords ] = avk::create_buffer<decltype(data.mTexCoords) , vertex_buffer_meta, uniform_texel_buffer_meta>(data.mTexCoords , content_description::texture_coordinate);
			auto [bufferNormals   , commandsNormals   ] = avk::create_buffer<decltype(data.mNormals)   , vertex_buffer_meta, uniform_texel_buffer_meta>(data.mNormals   , content_description::normal);
			auto [bufferTangents  , commandsTangents  ] = avk::create_buffer<decltype(data.mTangents)  , vertex_buffer_meta, uniform_texel_buffer_meta>(data.mTangents  , content_description::tangent);
			auto [bufferBitangents, commandsBitangents] = avk::create_buffer<decltype(data.mBitangents), vertex_buffer_meta, uniform_texel_buffer_meta>(data.mBitangents, content_description::bitangent);

			auto& dc = mDrawCalls.emplace_back(
				// Create the index buffer manually because we need to set some special configuration in preparation for its usage in the ray tracing shaders:
				context().create_buffer(
					memory_usage::device, {},
					index_buffer_meta::create_from_data(data.mIndices),                                  // This is the special configuration mentioned above:
					uniform_texel_buffer_meta::create_from_data(data.mIndices).set_format<glm::uvec3>(), // <-- Set a different format: Combine 3 consecutive elements to one unit, when used as uniform texel buffer
					read_only_input_to_acceleration_structure_builds_buffer_meta::create_from_data(data.mIndices)
				),
				// For all the other buffers, use a convenience function:
				//   Note that the positions buffer is created with an additional meta data, indicating that this buffer will be used for BLAS builds!
				std::move(bufferPositions),
				std::move(bufferTexCoords),
				std::move(bufferNormals),
				std::move(bufferTangents),
				std::move(bufferBitangents),
				data.mMaterialIndex,
				data.mModelMatrix
			);

			// Since we didn't use the convenience function for the indices, we still have to transfer the data into the buffer:
			commandsToBeExcecuted.push_back(dc.mIndexBuffer->fill(data.mIndices.data(), 0));

			// Push back the commands that need to be executed for this task, and execute them right after the loop:
			commandsToBeExcecuted.push_back(std::move(commandsPositions));
			commandsToBeExcecuted.push_back(std::move(commandsTexCoords));
			commandsToBeExcecuted.push_back(std::move(commandsNormals));
			commandsToBeExcecuted.push_back(std::move(commandsTangents));
			commandsToBeExcecuted.push_back(std::move(commandsBitangents));

			// Keep track of the current index, we'd like to store it down below, when creating a new geometry instance:
			const auto dataIndex = static_cast<uint32_t>(mRtxData.size());

			auto& rc = mRtxData.emplace_back(
				// After we have used positions and indices for building the BLAS, still need to create buffer views which allow us to access
				// the per vertex data in ray tracing shaders, where they will be accessible via samplerBuffer- or usamplerBuffer-type uniforms.
				context().create_buffer_view(dc.mIndexBuffer),
				context().create_buffer_view(dc.mNormalsBuffer)
			);

			mIndexBufferUniformTexelBufferViews.push_back(rc.mIndexBufferView->as_uniform_texel_buffer_view());
			mNormalBufferUniformTexelBufferViews.push_back(rc.mNormalsBufferView->as_uniform_texel_buffer_view());

			// Create a Bottom Level Acceleration Structure per geometry entry in mDrawCalls:
			rc.mBottomLevelAS = context().create_bottom_level_acceleration_structure(
				// We just use the very same geometry (vertices and indices) for building the bottom level acceleration structure (BLAS) as we use
				// for rendering with the graphics pipeline:
				{ avk::acceleration_structure_size_requirements::from_buffers(vertex_index_buffer_pair{ dc.mPositionsBuffer.as_reference(), dc.mIndexBuffer.as_reference() }) }, // This is only temporary here, no commands are stored => as_reference() is fine
				false // no need to allow updates for static geometry
			);

			// Create a bottom level acceleration structure instance with this geometry, i.e., transfer the geometry into the BLAS and build it:
			// We must ensure, however, that the buffer copies have finished before:
			commandsToBeExcecuted.push_back(sync::buffer_memory_barrier(dc.mPositionsBuffer.as_reference(), stage::auto_stage + access::auto_access >> stage::auto_stage + access::auto_access));
			commandsToBeExcecuted.push_back(sync::buffer_memory_barrier(dc.mIndexBuffer.as_reference(),     stage::auto_stage + access::auto_access >> stage::auto_stage + access::auto_access));
			commandsToBeExcecuted.push_back(rc.mBottomLevelAS->build({ vertex_index_buffer_pair{ dc.mPositionsBuffer.as_reference(), dc.mIndexBuffer.as_reference() } })); // Passing them as_reference() is good enough, since mDrawCalls outlives any BLAS build in our application for sure.
			// Note: The BLAS is build with the positions in the space that we got them from helpers::load_models_and_scenes_from_file.
			//       Since we haven't transformed the geometry in the meantime, this means that we are passing object space coordinates.

			// Now create an instance of this BLAS that we have built one step earlier.
			// Such a geometry instance is basically a reference from the top level acceleration structure (TLAS) to the geometry (represented by a BLAS.
			// And such a geometry is positioned somewhere in the world using a matrix.
			// We use the same model matrix that we have also stored in the associated mDrawCalls instance, to position this geometry instance in the world.
			geometryInstancesForTopLevelAS.push_back(
				context().create_geometry_instance(rc.mBottomLevelAS.as_reference()) // Refer to the concrete BLAS
					// Handle triangle meshes with an instance offset of 0:
					.set_instance_offset(0)
					// Set this instance's transformation matrix to position it in the world:
					.set_transform_column_major(avk::to_array(dc.mModelMatrix))
					// Set this instance's custom index, which is especially important since we'll use it in shaders
					// to refer to the right material and also vertex data (these two are aligned index-wise):
					.set_custom_index(dataIndex)
			);
#else
			auto [bufferIndices   , commandsIndices   ] = avk::create_buffer<decltype(data.mIndices)   , index_buffer_meta >(data.mIndices   , content_description::index);
			auto [bufferPositions , commandsPositions ] = avk::create_buffer<decltype(data.mPositions) , vertex_buffer_meta>(data.mPositions , content_description::position);
			auto [bufferTexCoords , commandsTexCoords ] = avk::create_buffer<decltype(data.mTexCoords) , vertex_buffer_meta>(data.mTexCoords , content_description::texture_coordinate);
			auto [bufferNormals   , commandsNormals   ] = avk::create_buffer<decltype(data.mNormals)   , vertex_buffer_meta>(data.mNormals   , content_description::normal);
			auto [bufferTangents  , commandsTangents  ] = avk::create_buffer<decltype(data.mTangents)  , vertex_buffer_meta>(data.mTangents  , content_description::tangent);
			auto [bufferBitangents, commandsBitangents] = avk::create_buffer<decltype(data.mBitangents), vertex_buffer_meta>(data.mBitangents, content_description::bitangent);

			mDrawCalls.emplace_back(
				std::move(bufferIndices),
				std::move(bufferPositions),
				std::move(bufferTexCoords),
				std::move(bufferNormals),
				std::move(bufferTangents),
				std::move(bufferBitangents),
				data.mMaterialIndex,
				data.mModelMatrix
			);

			// Push back the commands that need to be executed for this task, and execute them right after the loop:
			commandsToBeExcecuted.push_back(std::move(commandsIndices));
			commandsToBeExcecuted.push_back(std::move(commandsPositions));
			commandsToBeExcecuted.push_back(std::move(commandsTexCoords));
			commandsToBeExcecuted.push_back(std::move(commandsNormals));
			commandsToBeExcecuted.push_back(std::move(commandsTangents));
			commandsToBeExcecuted.push_back(std::move(commandsBitangents));
#endif
		}

#ifdef RTX_ON
		mTopLevelAS = avk::context().create_top_level_acceleration_structure(
			static_cast<uint32_t>(geometryInstancesForTopLevelAS.size()), // <-- Specify how many geometry instances there are expected to be at most
			false  // <-- No updates required, only static geometry
		);

		// Build it, sync before:
		//                                                  Note: This ||| is okay since the previous command in commandsToBeExecuted is a bottom-level acceleration structure build. 
		//                                                             |||      If it wouldn't be, we'd have to use something like: stage::auto_stages(5) + access::auto_accesses(5) >> stage::auto_stage + access::auto_access
		//                                                             vvv      in order to take 5 steps into previously recorded commands direction, in order to accumulate all their data.
		commandsToBeExcecuted.push_back(sync::global_memory_barrier(stage::auto_stage + access::auto_access >> stage::auto_stage + access::auto_access));
		commandsToBeExcecuted.push_back(mTopLevelAS->build(geometryInstancesForTopLevelAS));
#endif

		auto fen = context().record_and_submit_with_fence(std::move(commandsToBeExcecuted), *mQueue);
		fen->wait_until_signalled();

		// Create helper geometry for the skybox:
		mSkyboxSphere.create_sphere();
		
		mUniformsBuffer = context().create_buffer(
			memory_usage::host_coherent, {}, // Create its backing memory in a host coherent memory region (writable from the host-side)
			uniform_buffer_meta::create_from_size(sizeof(matrices_and_user_input)) // Meta data tells the type of this buffer => A uniform buffer
		);
		mLightsBuffer = context().create_buffer(
			memory_usage::device, {}, // Create its backing memory in a device-only memory region (takes an additional intermediate step
									  // to be filled (internally handled) through a host visible buffer, but faster access during rendering.)
			uniform_buffer_meta::create_from_size(sizeof(lightsource_data)) // Meta data tells the type of this buffer => A uniform buffer
		);

		// Initialize the cameras, and then add them to our composition (they are `avk::invokee`s too):
		mOrbitCam.set_translation({ -6.81f, 1.71f, -0.72f });
		mOrbitCam.look_along({ 1.0f, 0.0f, 0.0f });
		mOrbitCam.set_perspective_projection(glm::radians(60.0f), context().main_window()->aspect_ratio(), 0.1f, 1000.0f);
		current_composition()->add_element(mOrbitCam);

		mQuakeCam.copy_parameters_from(mOrbitCam);
		current_composition()->add_element(mQuakeCam);
		mQuakeCam.disable();
		mOriginalProjectionMatrix = mQuakeCam.projection_matrix();

		// Create the graphics pipelines for drawing the scene:
		init_pipelines();
		// Initialize the GUI, which is drawn through ImGui:
		init_gui(false);
		// Enable swapchain recreation and shader hot reloading:
		enable_the_updater();

		mAmbientOcclusion.config(*mQueue, mDescriptorCache,
			mUniformsBuffer,
			mFramebuffer->image_views()[0],
			mFramebuffer->image_views()[1],
			mFramebuffer->image_views()[2],
			mStorageImageViewsHdr[0] // <-- Destination
		);
		current_composition()->add_element(mAmbientOcclusion);

		mReflections.config(*mQueue, mDescriptorCache,
			mUniformsBuffer,
			mStorageImageViewsHdr[0], // <-- Source colors for reflections shall have ambient occlusion already applied
			mFramebuffer->image_views()[1],
			mFramebuffer->image_views()[2],
			mFramebuffer->image_views()[3],
			mStorageImageViewsHdr[1], // <-- Destination
			mMaterials, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)
		);
#ifdef RTX_ON
		mReflections.config_rtx_on(
			mLightsBuffer,
			mIndexBufferUniformTexelBufferViews, mNormalBufferUniformTexelBufferViews,
			mTopLevelAS
		);
#endif
		current_composition()->add_element(mReflections);

		mToneMapping.config(*mQueue, mDescriptorCache,
			mStorageImageViewsHdr[1],  // <-- HDR input
			mStorageImageViewsLdr[0]   // <-- Destination
		);
		current_composition()->add_element(mToneMapping);

		mAntiAliasing.config(*mQueue, mDescriptorCache,
			mUniformsBuffer,
			mStorageImageViewsLdr[0],		// <-- Source
			mFramebuffer->image_views()[1],	// <-- Depth
			mStorageImageViewsLdr[1]		// <-- Destination
		);
		current_composition()->add_element(mAntiAliasing);

		// Transfer the latest destination image into the swapchain image:
		mTransferToSwapchain.config(*mQueue,
			mFramebuffer->image_views()[1], transfer_to_swapchain::transfer_type::copy, layout::shader_read_only_optimal >> layout::shader_read_only_optimal,
			mStorageImageViewsLdr[1]      , transfer_to_swapchain::transfer_type::copy, layout::general >> layout::general,
			// By passing the (optional) intermediate image, instead of copying/blitting directly into the swap chain images, we perform:
			//   1) Copy of the LdrUnorm image -> sRGB image (because the LdrUnorm already contains gamma corrected values)
			//   2) Blit the sRGB image -> sRGB swap chain image (blit, s.t. the color channels are transferred in correct order)
			std::make_tuple(
				mImageViewSrgb         , transfer_to_swapchain::transfer_type::blit, layout::general >> layout::general
			)
		);
		current_composition()->add_element(mTransferToSwapchain);
	}

	/** TODO:	Helper function, which creates a renderpass with three sub passes, and
	 *	three graphics pipelines---one for each of the renderpass' sub passes.
	 *
	 */
	void init_pipelines()
	{
		using namespace avk;

		// Create some G-Buffer attachments here, let us first gather some data like formats:
		const auto resolution = context().main_window()->resolution();
		// Define the formats of our image-attachments:
		constexpr auto attachmentFormats = make_array<vk::Format>(
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eD32Sfloat,
			vk::Format::eR32G32B32A32Sfloat,
			vk::Format::eR16G16B16A16Uint
			);
		constexpr auto storageFormat = attachmentFormats[0];

		auto colorAttachment = context().create_image(resolution.x, resolution.y, attachmentFormats[0], 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled | image_usage::tiling_optimal | image_usage::transfer_source);
		auto depthAttachment = context().create_image(resolution.x, resolution.y, attachmentFormats[1], 1, memory_usage::device, image_usage::depth_stencil_attachment | image_usage::input_attachment | image_usage::sampled | image_usage::tiling_optimal | image_usage::transfer_source);
		auto uvNrmAttachment = context().create_image(resolution.x, resolution.y, attachmentFormats[2], 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled | image_usage::tiling_optimal);
		auto matIdAttachment = context().create_image(resolution.x, resolution.y, attachmentFormats[3], 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled | image_usage::tiling_optimal);

		// Note: sRGB formats are, unfortunately, not supported for storage images on many GPUs.
		auto storageImagesHdr = avk::make_array<avk::image>(
			context().create_image(resolution.x, resolution.y, storageFormat, 1, memory_usage::device, image_usage::shader_storage | image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled | image_usage::tiling_optimal),
			context().create_image(resolution.x, resolution.y, storageFormat, 1, memory_usage::device, image_usage::shader_storage | image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled | image_usage::tiling_optimal)
		);
		auto storageImagesLdr = avk::make_array<avk::image>(
			context().create_image(resolution.x, resolution.y, vk::Format::eR8G8B8A8Unorm, 1, memory_usage::device, image_usage::shader_storage | image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled | image_usage::tiling_optimal),
			context().create_image(resolution.x, resolution.y, vk::Format::eR8G8B8A8Unorm, 1, memory_usage::device, image_usage::shader_storage | image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled | image_usage::tiling_optimal)
		);
		auto imageSrgb = context().create_image(resolution.x, resolution.y, vk::Format::eR8G8B8A8Srgb, 1, memory_usage::device, image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled | image_usage::tiling_optimal);

		// Before using all these G-Buffer attachments => Let's transition their layouts into something useful:
		auto fen = context().record_and_submit_with_fence(command::gather(
				sync::image_memory_barrier(colorAttachment.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(depthAttachment.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(uvNrmAttachment.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(matIdAttachment.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				// Transition the storage image into GENERAL layout and keep it in that layout forever:
				sync::image_memory_barrier(storageImagesHdr[0].as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				sync::image_memory_barrier(storageImagesHdr[1].as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				sync::image_memory_barrier(storageImagesLdr[0].as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				sync::image_memory_barrier(storageImagesLdr[1].as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				sync::image_memory_barrier(imageSrgb.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				// Also transition all the backbuffer images into a useful initial layout:
				context().main_window()->layout_transitions_for_all_backbuffer_images()
			), *mQueue);
		fen->wait_until_signalled();
		
		// Before we can attach the G-Buffer images to a framebuffer, we have to wrap each of them with an image view:
		auto colorAttachmentView = context().create_image_view(std::move(colorAttachment));
		auto depthAttachmentView = context().create_image_view(std::move(depthAttachment));
		auto uvNrmAttachmentView = context().create_image_view(std::move(uvNrmAttachment));
		auto matIdAttachmentView = context().create_image_view(std::move(matIdAttachment));

		assert(mStorageImageViewsHdr.size() == storageImagesHdr.size());
		for (int i = 0; i < mStorageImageViewsHdr.size(); ++i) {
			mStorageImageViewsHdr[i] = context().create_image_view(std::move(storageImagesHdr[i]));
		}
		assert(mStorageImageViewsLdr.size() == storageImagesLdr.size());
		for (int i = 0; i < mStorageImageViewsLdr.size(); ++i) {
			mStorageImageViewsLdr[i] = context().create_image_view(std::move(storageImagesLdr[i]));
		}
		mImageViewSrgb = context().create_image_view(std::move(imageSrgb));

		// A renderpass is used to describe some configuration parts of a graphics pipeline, namely:
		//  - Which kinds of attachments are used and for how many sub passes
		//  - What dependencies are necessary between sub passes, to achieve correct rendering results
		auto renderpass = context().create_renderpass(
			{ // We have THREE sub passes here!   vvv    To properly set this up, we need to define for every attachment, how it is used in each single one of these THREE sub passes    vvv
				attachment::declare(attachmentFormats[0], on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::unused        >> usage::color(0) >> usage::color(0)      , on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(attachmentFormats[1], on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::depth_stencil >> usage::input(0) >> usage::depth_stencil , on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(attachmentFormats[2], on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(0)      >> usage::input(1) >> usage::preserve      , on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(attachmentFormats[3], on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(1)      >> usage::input(2) >> usage::preserve      , on_store::store.in_layout(layout::shader_read_only_optimal)),
			},
			{ // Describe the dependencies between external commands and the FIRST sub pass:
                subpass_dependency( subpass::external                >>   subpass::index(0),
					    			stage::color_attachment_output   >>  stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
									access::none                     >>  access::depth_stencil_attachment_read | access::depth_stencil_attachment_write | access::color_attachment_write
								  ),
				// Describe the dependencies between the FIRST and the SECOND sub pass:
				subpass_dependency( subpass::index(0)                                                                          >>   subpass::index(1),
					    			stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output  >>  stage::fragment_shader,
									access::depth_stencil_attachment_write | access::color_attachment_write                    >>  access::input_attachment_read
								  ),
				// Describe the dependencies between the SECOND and the THIRD sub pass:
				subpass_dependency( subpass::index(1)               >>  subpass::index(2),
									stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output  >>  stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
									access::depth_stencil_attachment_write | access::color_attachment_write                    >>  access::depth_stencil_attachment_read | access::depth_stencil_attachment_write | access::color_attachment_write
									// Note: Although this might seem unintuitive, we have to synchronize with read AND write access to the depth/stencil attachment here  ^^^  This is due to the image layout transition (input attachment optimal >> depth/stencil attachment optimal)
								  ),
				// Describe the dependencies between the THIRD sub pass and external commands:
				subpass_dependency( subpass::index(2)               >>  subpass::external,
									// Commands after this renderpass will be either compute shaders or copy/blit commands => sync with them here:
									stage::color_attachment_output | stage::late_fragment_tests              >>  stage::compute_shader | stage::transfer,
									access::color_attachment_write | access::depth_stencil_attachment_write  >>  access::shader_read   | access::transfer_read
									// Note: Gotta include depth writes ^ in the first synchronization scope due to a layout transition happening on the depth attachment after the last subpass.
				                  )
			}
		);
		
		// With images, image views, and renderpass described, let us create a separate framebuffer
		// which we will render into during all the passes described above:
		mFramebuffer = context().create_framebuffer(
			// It is composed of the renderpass:
			renderpass,
			avk::make_vector( // ...and of all the image views:
				  colorAttachmentView // Let us just pass all of them in a "shared" manner, which 
				, depthAttachmentView // means that they can be used at other places, too, and 
				, uvNrmAttachmentView // will lead them being stored in shared_ptrs internally.
				, matIdAttachmentView
			)
		);
		
		// Create a graphics pipeline consisting of a vertex shader and a fragment shader, plus additional config:
		mGBufferPassPipeline = context().create_graphics_pipeline_for(
			vertex_shader("shaders/transform_and_pass_on.vert"),
			tessellation_control_shader("shaders/tess_pn_controlpoints.tesc"),
			tessellation_evaluation_shader("shaders/tess_pn_interp_and_displacement.tese"),
			fragment_shader("shaders/blinnphong_and_normal_mapping.frag"),

			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // Stream positions from the vertex buffer bound at index #0
			from_buffer_binding(1)->stream_per_vertex<glm::vec2>()->to_location(1), // Stream texture coordinates from the vertex buffer bound at index #1
			from_buffer_binding(2)->stream_per_vertex<glm::vec3>()->to_location(2), // Stream normals from the vertex buffer bound at index #2
			from_buffer_binding(3)->stream_per_vertex<glm::vec3>()->to_location(3), // Stream tangents from the vertex buffer bound at index #3
			from_buffer_binding(4)->stream_per_vertex<glm::vec3>()->to_location(4), // Stream bitangents from the vertex buffer bound at index #4

			// Use the renderpass created above, and specify that we're intending to use this pipeline for its first subpass:
			renderpass, cfg::subpass_index{ 0u },

			// Configuration parameters for this graphics pipeline:
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),
	
			cfg::primitive_topology::patches,
			cfg::tessellation_patch_control_points{ 3u },

			// Define push constants and resource descriptors which are to be used with this draw call:
			push_constant_binding_data{ shader_type::vertex | shader_type::fragment | shader_type::tessellation_control | shader_type::tessellation_evaluation, 0, sizeof(push_constants_for_draw) },
			descriptor_binding(0, 0, mMaterials),
			descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
			descriptor_binding(1, 0, mUniformsBuffer),
			descriptor_binding(1, 1, mLightsBuffer)
		);

		// Create an (almost identical) pipeline to render the scene in wireframe mode
		mGBufferPassWireframePipeline = context().create_graphics_pipeline_from_template(mGBufferPassPipeline.as_reference(), [](graphics_pipeline_t& p) {
			p.rasterization_state_create_info().setPolygonMode(vk::PolygonMode::eLine);
		});
		
		// Create the graphics pipeline to be used for drawing the lit scene:
		mLightingPassGraphicsPipeline = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/lighting_pass.vert"),
			fragment_shader("shaders/lighting_pass.frag"),

			// Use the renderpass created above, and specify that we're intending to use this pipeline for its SECOND subpass:
			renderpass, cfg::subpass_index{ 1u },

			// Configuration parameters for this graphics pipeline:
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),
			cfg::depth_test::disabled(),

			// Define push constants and resource descriptors which are to be used with this draw call:
			push_constant_binding_data{ shader_type::vertex | shader_type::fragment | shader_type::tessellation_control | shader_type::tessellation_evaluation, 0, sizeof(push_constants_for_draw) },
			descriptor_binding(0, 0, mMaterials),
			descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
			descriptor_binding(1, 0, mUniformsBuffer),
			descriptor_binding(1, 1, mLightsBuffer),
			descriptor_binding(2, 0, mFramebuffer->image_view_at(1)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment),
			descriptor_binding(2, 1, mFramebuffer->image_view_at(2)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment),
			descriptor_binding(2, 2, mFramebuffer->image_view_at(3)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment)
#ifdef RTX_ON
			, descriptor_binding(3, 0, mTopLevelAS)
#endif
		);

		// Create the graphics pipeline to be used for drawing the skybox:
		mSkyboxPipeline = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/sky_gradient.vert"),
			fragment_shader("shaders/sky_gradient.frag"),
			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // Stream positions from the vertex buffer bound at index #0

			// Use the renderpass created above, and specify that we're intending to use this pipeline for its THIRD subpass:
			renderpass, cfg::subpass_index{ 2u },

			// Configuration parameters for this graphics pipeline:
			cfg::culling_mode::disabled,	// No backface culling required
			cfg::depth_test::enabled().set_compare_operation(cfg::compare_operation::less_or_equal), // as the depth buffer is cleared to 1.0 and the skybox clamps to 1.0, we need a <= comparison
			cfg::depth_write::disabled(),	// Don't write depth values
			cfg::depth_bounds::enable(1.0f, 1.0f),
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),

			descriptor_binding(0, 0, mUniformsBuffer)
		);
	}

	/**	Helper function, which sets up drawing of the GUI at initialization time.
	 *	For that purpose, it gets a handle to the imgui_manager component and installs a callback.
	 *	The GUI is drawn using the library Dear ImGui: https://github.com/ocornut/imgui
	 */
	void init_gui(bool recreate)
	{
		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (nullptr == imguiManager) {
			LOG_ERROR("Failed to init GUI, because composition does not contain an element of type avk::imgui_manager.");
			return;
		}

		// empty the array of textures shown in the UI
		texturesShownInTheUI.clear();

		auto sampler = avk::context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_border, 0.0f);
		int attachmentId = 0;
		for (auto& attachment : mFramebuffer->image_views()) {
			auto& tpl = texturesShownInTheUI.emplace_back();
			if (attachment->get_image().create_info().samples != vk::SampleCountFlagBits::e1) {
				std::get<std::string>(tpl) = std::format("Not rendering attachment #{} due to its sample count of {}", attachmentId++, vk::to_string(attachment->get_image().create_info().samples));
			}
			else if (avk::is_int_format(attachment->get_image().create_info().format) || avk::is_uint_format(attachment->get_image().create_info().format)) {
				std::get<std::string>(tpl) = std::format("Not rendering attachment #{} due to its (u)int format: {}", attachmentId++, vk::to_string(attachment->get_image().create_info().format));
			}
			else {
				std::get<std::string>(tpl) = std::format("Attachment {}:", attachmentId++);
				std::get<avk::image_sampler>(tpl) = avk::context().create_image_sampler(attachment, sampler);
				std::get< avk::layout::image_layout>(tpl) = avk::layout::shader_read_only_optimal;
			}
		}
		for (int i = 0; i < mStorageImageViewsHdr.size(); ++i) {
			texturesShownInTheUI.push_back(std::make_tuple(std::format("HDR Storage Image [{}]:", i), avk::context().create_image_sampler(mStorageImageViewsHdr[i], sampler), avk::layout::general));
		}
		for (int i = 0; i < mStorageImageViewsLdr.size(); ++i) {
			texturesShownInTheUI.push_back(std::make_tuple(std::format("LDR Storage Image [{}]:", i), avk::context().create_image_sampler(mStorageImageViewsLdr[i], sampler), avk::layout::general));
		}
		texturesShownInTheUI.push_back(std::make_tuple("sRGB Image:", avk::context().create_image_sampler(mImageViewSrgb, sampler), avk::layout::general));

		// Do not add the callback twice
		if (recreate == true) return;

		// Install a callback which will be invoked each time imguiManager's render() is invoked by the framework:
		imguiManager->add_callback([this, &lGBufferTextures = texturesShownInTheUI, imguiManager]() {

			ImGui::Begin("Settings");
			ImGui::SetWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
			ImGui::SetWindowSize(ImVec2(275.0f, 990.0f), ImGuiCond_FirstUseEver);
			ImGui::Text("%.3f ms/frame (%.1f fps)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::Text("%.3f ms/Ambient Occlusion", mAmbientOcclusion.duration());
			ImGui::Text("%.3f ms/Reflections", mReflections.duration());
			ImGui::Text("%.3f ms/Tone Mapping", mToneMapping.duration());
			ImGui::Text("%.3f ms/Anti Aliasing", mAntiAliasing.duration());
			
			static std::vector<float> accum; // accumulate (then average) 10 frames
			accum.push_back(ImGui::GetIO().Framerate);
			static std::vector<float> values;
			if (accum.size() == 10) {
				values.push_back(std::accumulate(std::begin(accum), std::end(accum), 0.0f) / 10.0f);
				accum.clear();
			}
			if (values.size() > 90) { // Display up to 90(*10) history frames
				values.erase(values.begin());
			}
			ImGui::PlotLines("FPS", values.data(), static_cast<int>(values.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 50.0f));

			ImGui::Separator();
			bool quakeCamEnabled = mQuakeCam.is_enabled();
			if (ImGui::Checkbox("Enable Quake Camera", &quakeCamEnabled)) {
				if (quakeCamEnabled) { // => should be enabled
					mQuakeCam.enable();
					mOrbitCam.disable();
				}
			}
			if (quakeCamEnabled) {
			    ImGui::TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[Esc] to exit Quake Camera navigation");
				if (avk::input().key_pressed(avk::key_code::escape)) {
					mOrbitCam.enable();
					mQuakeCam.disable();
				}
			}
			else {
				ImGui::TextColored(ImVec4(.8f, .4f, .4f, 1.f), "[Esc] to exit application");
			}
			if (imguiManager->begin_wanting_to_occupy_mouse() && mOrbitCam.is_enabled()) {
				mOrbitCam.disable();
			}
			if (imguiManager->end_wanting_to_occupy_mouse() && !mQuakeCam.is_enabled()) {
				mOrbitCam.enable();
			}
			ImGui::Separator();

			ImGui::SetNextItemWidth(100);
			ImGui::InputInt("Max point lights", &mLimitNumPointlights, 0, 0);

			// GUI elements for controlling renderin parameters, passed on to mGBufferPassPipeline and mSkyboxPipeline:
			ImGui::PushItemWidth(100);
			ImGui::SliderFloat("Tessellation Level", &mTessellationLevel, 1.0f, 32.0f, "%.0f");
			ImGui::SliderFloat("Displacement Strength", &mDisplacementStrength, 0.0f, 1.0f);
			ImGui::PopItemWidth();

			ImGui::Checkbox("Wireframe", &mWireframeMode);
			ImGui::Checkbox("PN on/off", &mPnEnabled);
			
			ImGui::Separator();
			// GUI elements for the light sources, enables showing/hiding light gizmos, and the light source editor:
			bool enableGizmos = helpers::are_lightsource_gizmos_enabled();
			if (ImGui::Checkbox("Light gizmos", &enableGizmos)) {
				helpers::set_lightsource_gizmos_enabled(enableGizmos);
			}
			bool showLightsEd = helpers::is_lightsource_editor_visible();
			if (ImGui::Checkbox("Light editor", &showLightsEd)) {
				helpers::set_lightsource_editor_visible(showLightsEd);
			}

			// GUI elements for showing camera data, camera presets (interesting perspectives) and the camera presets editor:
			auto camPresets = avk::current_composition()->element_by_type<camera_presets>();
			bool showCamPresets = helpers::is_camera_presets_editor_visible();
			if (ImGui::Checkbox("Camera presets", &showCamPresets)) {
				helpers::set_camera_presets_editor_visible(showCamPresets);
			}
			
			ImGui::Text(std::format("Cam pos: {}", avk::to_string(mQuakeCam.translation())).c_str());

			ImGui::Separator();

			auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
			assert(nullptr != imguiManager);
			const auto resolution = avk::context().main_window()->resolution();
			ImGui::Text("G-BUFFER ATTACHMENTS");
			static constexpr float scale = 1.0f / 8.0f;
			for (const auto& tpl : lGBufferTextures) {
				ImGui::Text(std::get<std::string>(tpl).c_str());
				if (std::get<avk::image_sampler>(tpl).has_value()) {
					ImTextureID texId = imguiManager->get_or_create_texture_descriptor(std::get<avk::image_sampler>(tpl).get(), std::get<avk::layout::image_layout>(tpl));
					ImGui::ImageWithBg(texId, ImVec2(resolution.x * scale, resolution.y * scale), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f), ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
				}
			}

			ImGui::End();
		});
	}

	/**	The updater takes care of performing the necessary updates after
	 *	the swapchain has been changed (e.g., through a window resize),
	 *	and it also enables shader hot reloading.
	 *
	 *	Shader Hot Reloading: If you leave the post build helper running in the background,
	 *	                      it will monitor your shader files for changes (i.e. just edit
	 *	                      and save). On each save event, the shader will be compiled to
	 *						  SPIR-V automatically and (if successful) hot reloaded on the fly.
	 */
	void enable_the_updater()
	{
		using namespace avk;

		// The updater takes care of making the necessary updates after window resizes:
		mUpdater.emplace();
		mUpdater->on(swapchain_changed_event(context().main_window()))
			.invoke([this]() { // Fix camera's aspect ratios
				mOrbitCam.set_aspect_ratio(context().main_window()->aspect_ratio());
				mQuakeCam.set_aspect_ratio(context().main_window()->aspect_ratio());
				auto newRes = context().main_window()->resolution();
				auto newFramebuffer = context().create_framebuffer_from_template(*mFramebuffer, 
					[w = newRes[0], h = newRes[1]](avk::image_t& aImage)            { aImage.create_info().extent.setWidth(w).setHeight(h); },
					[w = newRes[0], h = newRes[1]](avk::image_view_t& aImageView)   { },
					[w = newRes[0], h = newRes[1]](avk::framebuffer_t& aFramebuffer) {aFramebuffer.create_info().setWidth(w).setHeight(h); }
				);

				// recreate HDR images
				for (int i = 0; i < mStorageImageViewsHdr.size(); i++) {
					auto image = context().create_image_from_template(mStorageImageViewsHdr[i]->get_image(), [w = newRes[0], h = newRes[1]](avk::image_t& aImage) { aImage.create_info().extent.setWidth(w).setHeight(h); });
					mStorageImageViewsHdr[i] = context().create_image_view(std::move(image));
				}

				// recreate LDR images
				for (int i = 0; i < mStorageImageViewsLdr.size(); i++) {
					auto image = context().create_image_from_template(mStorageImageViewsLdr[i]->get_image(), [w = newRes[0], h = newRes[1]](avk::image_t& aImage) { aImage.create_info().extent.setWidth(w).setHeight(h); });
					mStorageImageViewsLdr[i] = context().create_image_view(std::move(image));
				}

				// recreate image SRGB
				{
					auto imageSrgb = context().create_image_from_template(mImageViewSrgb->get_image(), [w = newRes[0], h = newRes[1]](avk::image_t& aImage) { aImage.create_info().extent.setWidth(w).setHeight(h); });
					mImageViewSrgb = context().create_image_view(std::move(imageSrgb));
				}

				// transition newly created images from undefined to shader read only optimal
				std::vector<recorded_commands_t> commandsToBeExcecuted;
				for (int i = 0; i < newFramebuffer->image_views().size(); i++) {
					commandsToBeExcecuted.push_back(sync::image_memory_barrier(newFramebuffer->image_at(i), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal));
				}

				// mStorageImageViewsHdr, mStorageImageViewsLdr and mImageViewSrgb layout transitions
				for (int i = 0; i < mStorageImageViewsHdr.size(); i++) {
					commandsToBeExcecuted.push_back(sync::image_memory_barrier(mStorageImageViewsHdr[i]->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general));
				}
				for (int i = 0; i < mStorageImageViewsLdr.size(); i++) {
					commandsToBeExcecuted.push_back(sync::image_memory_barrier(mStorageImageViewsLdr[i]->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general));
				}
				commandsToBeExcecuted.push_back(sync::image_memory_barrier(mImageViewSrgb->get_image(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general));

				// Also transition all the backbuffer images into a useful initial layout:
				auto backbufferTransitions = context().main_window()->layout_transitions_for_all_backbuffer_images();
				commandsToBeExcecuted.insert(commandsToBeExcecuted.end(), backbufferTransitions.begin(), backbufferTransitions.end());
				auto fen = context().record_and_submit_with_fence(std::move(commandsToBeExcecuted), *mQueue);
				fen->wait_until_signalled();

				// swap out:
				std::swap(*newFramebuffer, *mFramebuffer);

				// configer all post processing effects with the updated images

				mAmbientOcclusion.config(*mQueue, mDescriptorCache,
					mUniformsBuffer,
					mFramebuffer->image_views()[0],
					mFramebuffer->image_views()[1],
					mFramebuffer->image_views()[2],
					mStorageImageViewsHdr[0] // <-- Destination
				);

				mReflections.config(*mQueue, mDescriptorCache,
					mUniformsBuffer,
					mStorageImageViewsHdr[0], // <-- Source colors for reflections shall have ambient occlusion already applied
					mFramebuffer->image_views()[1],
					mFramebuffer->image_views()[2],
					mFramebuffer->image_views()[3],
					mStorageImageViewsHdr[1], // <-- Destination
					mMaterials, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)
				);

				mToneMapping.config(*mQueue, mDescriptorCache,
					mStorageImageViewsHdr[1],  // <-- HDR input
					mStorageImageViewsLdr[0]   // <-- Destination
				);

				mAntiAliasing.config(*mQueue, mDescriptorCache,
					mUniformsBuffer,
					mStorageImageViewsLdr[0],		// <-- Source
					mFramebuffer->image_views()[1],	// <-- Depth
					mStorageImageViewsLdr[1]		// <-- Destination
				);

				mTransferToSwapchain.config(*mQueue,
					mFramebuffer->image_views()[1], transfer_to_swapchain::transfer_type::copy, layout::shader_read_only_optimal >> layout::shader_read_only_optimal,
					mStorageImageViewsLdr[1], transfer_to_swapchain::transfer_type::copy, layout::general >> layout::general,
					// By passing the (optional) intermediate image, instead of copying/blitting directly into the swap chain images, we perform:
					//   1) Copy of the LdrUnorm image -> sRGB image (because the LdrUnorm already contains gamma corrected values)
					//   2) Blit the sRGB image -> sRGB swap chain image (blit, s.t. the color channels are transferred in correct order)
					std::make_tuple(
						mImageViewSrgb, transfer_to_swapchain::transfer_type::blit, layout::general >> layout::general
					)
				);

				init_gui(true);
				// new == now the old one => destroy in 1 frame:
				context().main_window()->handle_lifetime(std::move(newFramebuffer));
			}) 
			.update(mGBufferPassPipeline) // Update some of the pipelines after the swap chain has changed
			.update(mGBufferPassWireframePipeline)
			.update(mLightingPassGraphicsPipeline)
			.update(mSkyboxPipeline);
		
		// Also enable shader hot reloading via the updater:
		mUpdater->on(shader_files_changed_event(mGBufferPassPipeline.as_reference()))
			.update(mGBufferPassPipeline);
		mUpdater->on(shader_files_changed_event(mGBufferPassWireframePipeline.as_reference()))
			.update(mGBufferPassWireframePipeline);
		mUpdater->on(shader_files_changed_event(mLightingPassGraphicsPipeline.as_reference()))
			.update(mLightingPassGraphicsPipeline);
		mUpdater->on(shader_files_changed_event(mSkyboxPipeline.as_reference()))
			.update(mSkyboxPipeline);
	}

	// ----------------------- ^^^   INITIALIZATION   ^^^ -----------------------

	// ----------------------- vvv  PER FRAME ACTION  vvv -----------------------

	/**	Update callback which is invoked by the framework every frame before every render() callback is invoked.
	 *	Here, we handle things like user input and animation.
	 */
	void update() override
	{
		using namespace avk;

		// Keep the cameras sync to make life easier:
		if (mQuakeCam.is_enabled()) {
			mOrbitCam.set_matrix(mQuakeCam.matrix());
		}
		if (mOrbitCam.is_enabled()) {
			mQuakeCam.set_matrix(mOrbitCam.matrix());
		}

		// Escape tears everything down (if quake camera is not active):
		if (!mQuakeCam.is_enabled() && avk::input().key_pressed(avk::key_code::escape) || avk::context().main_window()->should_be_closed()) {
			// Stop the current composition:
			avk::current_composition()->stop();
		}

		// SPACE toggles between light sources animating and holding positions
		if (input().key_pressed(key_code::space)) {
			mLightsAnimating = !mLightsAnimating;
			if (!mLightsAnimating) {
				mLightAniPauseTime = time().time_since_start();
			}
			else {
				float offset = time().time_since_start() - mLightAniPauseTime;
				mLightAniTimeSub += offset;
			}
		}
	}

	/**	TODO: Render callback which is invoked by the framework every frame after every update() callback has been invoked.
	 *	Here, we handle everything drawing-related, which includes updating/uploading all buffers, and issuing all draw calls.
	 *
	 *	Important: We must establish a dependency to the "swapchain image available" condition, i.e., we must wait for the
	 *	           next swap chain image to become available before we may start to render into it.
	 *			   This dependency is expressed through a semaphore, and the framework demands us to use it via the function:
	 *			   context().main_window()->consume_current_image_available_semaphore() for the main_window (our only window).
	 *
	 *			   More background information: At one point, we also must tell the presentation engine when we are done
	 *			   with rendering by the means of a semaphore. Actually, we would have to use the framework function:
	 *			   mainWnd->add_present_dependency_for_current_frame() for that purpose, but we don't have to do it in our case
	 *			   since we are rendering a GUI. imgui_manager will add a semaphore as dependency for the presentation engine.
	 */
	void render() override
	{
		using namespace avk;

		// We get a semaphore from the framework which will get signaled as soon as the next swap chain image becomes available.
		// Only after it has become available, we may start rendering the current frame into it.
		// We get the semaphore here, and use it further down to describe a dependency of our recorded commands:
		auto imageAvailableSemaphore = context().main_window()->consume_current_image_available_semaphore();

		// Let Temporal Anti-Aliasing modify the camera's projection matrix (it will restore it after it has processed the current frame):
		mAntiAliasing.save_view_matrix_and_modify_projection_matrix();

		// Update the data in our uniform buffers:
		matrices_and_user_input uni;
		uni.mViewMatrix        = mQuakeCam.view_matrix();
		uni.mProjMatrix        = mQuakeCam.projection_matrix();
		uni.mInverseProjMatrix = glm::inverse(uni.mProjMatrix);
		uni.mCamPos            = glm::translate(mQuakeCam.translation());
		uni.mUserInput         = glm::vec4{ mTessellationLevel, mDisplacementStrength, mPnEnabled ? 1.0f : 0.0f, 0.0f };
		uni.mUserInput[3]      = 1.0f; // Always reconstruct position from depth

		// Since this buffer has its backing memory in a "host visible" memory region, we just need to write the new data to it.
		// No need to submit the (empty, in this case!) action_type_command that is returned by buffer_t::fill() to a queue.
		// If its backing memory was in a "device" memory region, we would have to, though (see lights buffer below for the difference!).
		mUniformsBuffer->fill(&uni, 0);

		// Animate lights:
		if (mLightsAnimating) {
			helpers::animate_lights(helpers::get_lights(), time().time_since_start() - mLightAniTimeSub);
		}

		// Update the data in our light sources buffer:
		auto activeLights = helpers::get_active_lightsources(mLimitNumPointlights);
		lightsource_data lightsData{
			glm::uvec4{
				helpers::get_lightsource_type_begin_index(activeLights, lightsource_type::ambient),
				helpers::get_lightsource_type_end_index(activeLights, lightsource_type::ambient),
				helpers::get_lightsource_type_begin_index(activeLights, lightsource_type::directional),
				helpers::get_lightsource_type_end_index(activeLights, lightsource_type::directional)
			},
			glm::uvec4{
				helpers::get_lightsource_type_begin_index(activeLights, lightsource_type::point),
				helpers::get_lightsource_type_end_index(activeLights, lightsource_type::point),
				helpers::get_lightsource_type_begin_index(activeLights, lightsource_type::spot),
				helpers::get_lightsource_type_end_index(activeLights, lightsource_type::spot)
			},
			convert_for_gpu_usage<std::array<lightsource_gpu_data, MAX_NUMBER_OF_LIGHTSOURCES>>(activeLights, mQuakeCam.view_matrix())
		};
		auto lightsSemaphore = context().record_and_submit_with_semaphore(
			// The buffer's backing memory is in a "device" memory region. Therefore, the data must first be copied into 
			// a host visible buffer (done internally) and then transferred onto the device, into that device memory.
			// This process must be synchronized => we need to submit the action_type_command to a queue:
			{ mLightsBuffer->fill(&lightsData, 0) },
			*mQueue, 
			stage::copy
		);
		// Upon completion of this ^ memory transfer into device memory, a semaphore is signaled.
		// We can use this semaphore so that other work must wait on it.

		// Alloc a new command buffer for the current frame, which we are going to record commands into, and then submit to the queue:
		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		// Determine which pipelines to use:
		auto& scenePipeline = mWireframeMode ? mGBufferPassWireframePipeline.as_reference() : mGBufferPassPipeline.as_reference();

		context().record({ // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):

			command::custom_commands([&,this](avk::command_buffer_t& cb) {
					// Note 1: The Vulkan SDK's command buffer class (from Vulkan-Hpp in this case) provides 
					//         ALL the commands there are. Use it to record anything into the command buffer:
					const vk::CommandBuffer& vkHppCommandBuffer = cb.handle();

					// Note 2: For some commands, the framework's avk::command_buffer_t class provides methods,
					//         which allow more convenient usage/recording of functionality into the command buffer.
					//         The following code uses mostly these avk::command_buffer_t methods:
					cb.record(command::begin_render_pass_for_framebuffer(scenePipeline.renderpass_reference(), mFramebuffer.as_reference()));
					
					// Bind the pipeline for subsequent draw calls:
					cb.record(command::bind_pipeline(scenePipeline));
					// Bind all resources we need in shaders:
					cb.record(avk::command::bind_descriptors(scenePipeline.layout(), mDescriptorCache->get_or_create_descriptor_sets({
						descriptor_binding(0, 0, mMaterials),
						descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
						descriptor_binding(1, 0, mUniformsBuffer),
						descriptor_binding(1, 1, mLightsBuffer)
					})));

					for (const auto& drawCall : mDrawCalls) {
						cb.record(avk::command::push_constants(scenePipeline.layout(), push_constants_for_draw{ drawCall.mModelMatrix, drawCall.mMaterialIndex }));
						cb.record(avk::command::draw_indexed(
							drawCall.mIndexBuffer.as_reference(),     // Index buffer
							drawCall.mPositionsBuffer.as_reference(), // Vertex buffer at index #0
							drawCall.mTexCoordsBuffer.as_reference(), // Vertex buffer at index #1
							drawCall.mNormalsBuffer.as_reference(),   // Vertex buffer at index #2
							drawCall.mTangentsBuffer.as_reference(),  // Vertex buffer at index #3
							drawCall.mBitangentsBuffer.as_reference() // Vertex buffer at index #4
						));
					}

					cb.record(avk::command::next_subpass());

					cb.record(avk::command::bind_pipeline(mLightingPassGraphicsPipeline.as_reference()));
					// Bind all resources we need in shaders:
					cb.record(avk::command::bind_descriptors(mLightingPassGraphicsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
						descriptor_binding(0, 0, mMaterials),
						descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
						descriptor_binding(1, 0, mUniformsBuffer),
						descriptor_binding(1, 1, mLightsBuffer),
						descriptor_binding(2, 0, mFramebuffer->image_view_at(1)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment),
						descriptor_binding(2, 1, mFramebuffer->image_view_at(2)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment),
						descriptor_binding(2, 2, mFramebuffer->image_view_at(3)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment)
#ifdef RTX_ON
						, descriptor_binding(3, 0, mTopLevelAS)
#endif
					})));
					// Let's just use the raw Vulkan-Hpp class for the draw call, because we don't need any convenience functionality here:
					vkHppCommandBuffer.draw(6u, 1u, 0u, 1u);

					cb.record(avk::command::next_subpass());

					if (!mWireframeMode) {
						cb.record(avk::command::bind_pipeline(mSkyboxPipeline.as_reference()));
						cb.record(avk::command::bind_descriptors(mSkyboxPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mUniformsBuffer)
						})));
						cb.record(avk::command::draw_indexed(mSkyboxSphere.mIndexBuffer.as_reference(), mSkyboxSphere.mPositionsBuffer.as_reference()));
					}

					cb.record(avk::command::end_render_pass());

				}),
			}) // End of command recording
			.into_command_buffer(cmdBfr)
			.then_submit_to(*mQueue)
			// The work package we are submitting to the queue must wait in the EARLY FRAGMENT TESTS for the 
		    // imageAvailableSemaphore being signaled, because in that stage, the depth buffer is accessed:
			.waiting_for(imageAvailableSemaphore >> stage::early_fragment_tests)
			// Hint: We could add further semaphore dependencies here, if we needed to wait on other work, too.
			.waiting_for(lightsSemaphore >> stage::fragment_shader) // The lights are accessed in the fragment shader => that stage must wait!
			.submit();

		cmdBfr->handle_lifetime_of(std::move(lightsSemaphore));
		
		// Use a convenience function of avk::window to take care of the command buffer's lifetime:
		// It will get deleted in the future after #concurrent-frames have passed by.
		context().main_window()->handle_lifetime(std::move(cmdBfr));
	}

	// ----------------------- ^^^  PER FRAME ACTION  ^^^ -----------------------
	
	void finalize() override
	{
		helpers::clean_up_timing_resources();
	}
	
	// ----------------------- vvv  MEMBER VARIABLES  vvv -----------------------
private:
	/** One single queue to submit all the commands to: */
	avk::queue* mQueue;

	/** One descriptor cache to use for allocating all the descriptor sets from: */
	avk::descriptor_cache mDescriptorCache;

	/** A command pool for allocating (single-use) command buffers from: */
	avk::command_pool mCommandPool;

	/** Buffer containing all the different materials as loaded from 3D models/ORCA scenes: */
	avk::buffer mMaterials;
	/** Set of image samplers which are referenced by the materials in mMaterials: */
	std::vector<avk::image_sampler> mImageSamplers;
	/** Draw calls which are for all the geometry, references materials mMaterials by index: */
	std::vector<draw_call> mDrawCalls;
#ifdef RTX_ON
	/** Draw calls which are for all the geometry, references materials mMaterials by index: */
	std::vector<rtx_data_per_draw_call> mRtxData;

	std::vector<avk::buffer_view_descriptor_info> mIndexBufferUniformTexelBufferViews;
	std::vector<avk::buffer_view_descriptor_info> mNormalBufferUniformTexelBufferViews;

	/** Acceleration structure which will contain the entire scene. */
	avk::top_level_acceleration_structure mTopLevelAS;
#endif

	// Good old camera duo:
	avk::orbit_camera mOrbitCam;
	avk::quake_camera mQuakeCam;
	/** A framebuffer for the G-Buffer pass: */
	avk::framebuffer mFramebuffer;

	// Helper images:
	std::array<avk::image_view, 2> mStorageImageViewsHdr;
	std::array<avk::image_view, 2> mStorageImageViewsLdr;
	avk::image_view mImageViewSrgb;

	avk::graphics_pipeline mGBufferPassPipeline, mGBufferPassWireframePipeline;
	avk::graphics_pipeline mLightingPassGraphicsPipeline;

	avk::buffer mUniformsBuffer;
	avk::buffer mLightsBuffer;
	
	// ------------------ UI Parameters -------------------
	/** Factor that determines to which amount normals shall be distorted through normal mapping: */
	float mDisplacementStrength = 0.5f;

	/** Tessellation factor, controlled through the UI: */
	float mTessellationLevel = 8.0f;

	/** Flag controlled through the UI, indicating whether the scene shall be rendered with filled polygons or in wireframe mode: */
	bool mWireframeMode = false;

	/** Flag controlled through the UI, indicating whether PN triangles is currently active or not: */
	bool mPnEnabled = true;
	
	int mLimitNumPointlights = 98 + EXTRA_POINTLIGHTS;

	// Helper variables to allow the pausing of light source animations:
	bool mLightsAnimating = true;
	float mLightAniPauseTime = 0.0f;
	float mLightAniTimeSub = 0.0f;

	// --------------------- Skybox -----------------------
	simple_geometry mSkyboxSphere;
	avk::graphics_pipeline mSkyboxPipeline;

	// --------------------- Other -----------------------
	// Stores the original projection matrix of the quake camera, because it gets modified by temporal anti-aliasing
	glm::mat4 mOriginalProjectionMatrix;

	// ----------------------- ^^^  MEMBER VARIABLES  ^^^ -----------------------

	// The elements to handle the post processing effects:
	ambient_occlusion mAmbientOcclusion;
	reflections mReflections;
	tone_mapping mToneMapping;
	anti_aliasing mAntiAliasing;
	transfer_to_swapchain mTransferToSwapchain;

	// The list of textures that are shown by the UI
	std::vector<std::tuple<std::string, avk::image_sampler, avk::layout::image_layout>> texturesShownInTheUI;
};

//  Main:
//
// +---------------------------------------+
// |                                       |
// |        ARTR 2024 Assignment 4         |
// |                                       |
// +---------------------------------------+
//
//  All good things come in... fours!
// 
int main() 
{
	using namespace avk;

	int result = EXIT_FAILURE;

	try {
		// Create a window, set some configuration parameters (also relevant for its swap chain), and open it:
		auto mainWnd = context().create_window("ARTR 2024 Assignment 4");
		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->set_additional_back_buffer_attachments({ attachment::declare(vk::Format::eD32Sfloat, on_load::clear, usage::depth_stencil, on_store::store) });
		mainWnd->enable_resizing(true);
		mainWnd->request_srgb_framebuffer(true);
		mainWnd->set_presentaton_mode(presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(1u); // For simplicity, we are using only one concurrent frame in Assignment 4, otherwise we'd have to duplicate many resources as in Assignment 1
		mainWnd->open();

		// Create one single queue which we will submit all command buffers to:
		// (We pass the mainWnd because also presentation shall be submitted to this queue)
		auto& singleQueue = context().create_queue({}, queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->set_queue_family_ownership(singleQueue.family_index());
		mainWnd->set_present_queue(singleQueue);

		// Create an instance of our main class which contains the relevant host code for Assignment 1:
		auto app = assignment4(singleQueue);

		// Create another element for drawing the GUI via the library Dear ImGui:
		auto ui = imgui_manager(singleQueue);
        ui.set_custom_font("assets/JetBrainsMono-Regular.ttf");

		// Two more utility elements:
		auto lightsEditor = helpers::create_lightsource_editor(singleQueue, false);
		auto camPresets = helpers::create_camera_presets(singleQueue, false);

		// Pass everything to avk::start and off we go:
		auto composition = configure_and_compose(
			application_name("ARTR 2024 Framework"),
			[](vk::PhysicalDeviceFeatures& features) {
				features.fillModeNonSolid = VK_TRUE; // this device feature is required for wireframe rendering
				features.depthBounds = VK_TRUE;
			},
			[](vk::DebugUtilsMessageTypeFlagsEXT& messageTypes) {
				// Exclude the ePerformance flag to make validation output less verbose:
				messageTypes = messageTypes & ~vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
				// Otherwise we would get performance warnings whenever we used blit/copy with images in general layout.
			},
#ifdef RTX_ON
			avk::required_device_extensions()
				// We need several extensions for ray tracing:
				.add_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
				.add_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
				.add_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
				.add_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME),
			[](vk::PhysicalDeviceVulkan12Features& aVulkan12Featues) {
				// Also this Vulkan 1.2 feature is required for ray tracing:
				aVulkan12Featues.setBufferDeviceAddress(VK_TRUE);
			},
			[](vk::PhysicalDeviceAccelerationStructureFeaturesKHR& aAccelerationStructureFeatures) {
				// Enabling the extensions is not enough, we need to activate ray tracing features explicitly.
				// Here for usage of acceleration structures:
				aAccelerationStructureFeatures.setAccelerationStructure(VK_TRUE);
			},
			[](vk::PhysicalDeviceRayQueryFeaturesKHR& aRayQueryFeatures) {
				// Here for ray query:
				aRayQueryFeatures.setRayQuery(VK_TRUE);
			},
			[](vk::PhysicalDeviceRayTracingPipelineFeaturesKHR& aRayTracingFeatures) {
				// And here for ray tracing pipelines:
				aRayTracingFeatures.setRayTracingPipeline(VK_TRUE);
			},
#endif
			mainWnd,
			// Pass the so-called "invokees" which will get their callback methods (such as update() or render()) invoked:
			app, ui, lightsEditor, camPresets
		);

		// Create an invoker object, which defines the way how invokees/elements are invoked
		// (In this case, just sequentially in their execution order):
		sequential_invoker invoker;

		// Off we go:
		composition.start_render_loop(
			// Callback in the case of update:
			[&invoker](const std::vector<invokee*>& aToBeInvoked) {
				// Call all the update() callbacks:
				invoker.invoke_updates(aToBeInvoked);
			},
			// Callback in the case of render:
			[&invoker](const std::vector<invokee*>& aToBeInvoked) {
				// Sync (wait for fences and so) per window BEFORE executing render callbacks
				avk::context().execute_for_each_window([](window* wnd) {
					wnd->sync_before_render();
				});

				// Call all the render() callbacks:
				invoker.invoke_renders(aToBeInvoked);

				// Render per window:
				avk::context().execute_for_each_window([](window* wnd) {
					wnd->render_frame();
				});
			}
		);

		result = EXIT_SUCCESS;
	}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}

	return result;
}

