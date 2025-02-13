#include "imgui_utils.h"
#include "utils/image_based_lighting_helper.hpp"

/**	Main class for the host code of ARTR 2024 Assignment 3.
 *
 *	It is derived from avk::invokee, s.t. it can be handed-over to avk::start, which
 *	adds it to a avk::composition internally => its callbacks (such as initialize(),
 *	update(), or render() will be invoked).
 *
 *	Hint: Look out for "TODO Task X" comments!
 */
class assignment3 : public avk::invokee
{
	// ------------------ Structs for transfering data from HOST -> DEVICE ------------------

	/** Struct definition for push constants used for the draw calls of the scene */
	struct push_constants_for_draw
	{
		glm::mat4 mModelMatrix;
		glm::vec4 mPbsOverride; // override values for physically based shading: x = metallic, y = roughness, z = use override
		int mMaterialIndex;
	};

	/** Struct definition for push constants used for the lighting pass dispatch call */
	struct push_constants_for_dispatch
	{
		int mSampleCount;
	};
	
	/** Struct definition for data used as UBO across different pipelines, containing matrices and user input */
	struct matrices_and_user_input
	{
		// view matrix as returned from the camera
		glm::mat4 mViewMatrix;
		// projection matrix as returned from the camera
		glm::mat4 mProjMatrix;
		// Inverse of mProjMatrix:
		glm::mat4 mInverseProjMatrix;
		// transformation matrix which tranforms to camera's position
		glm::mat4 mCamPos;
		// x = tessellation factor, y = displacement strength, z = enable PN-triangles, w unused
		glm::vec4 mUserInput;
		// master switch to enable physically based shading
		VkBool32 mPbsEnabled;
		// user-parameter to scale roughness for physically based shading
		float mUserDefinedRoughnessStrength;
		// light intensity multiplier for physically based shading
		float mPbsLightBoost;
		// master switch to enable image based lighting
		VkBool32 mIblEnabled;
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
	assignment3(avk::queue& aQueue)
		: mQueue{ &aQueue }
		, mIblHelper{ aQueue }
		, mSkyboxSphere{ &aQueue }
		, mSkyboxIblCube{ &aQueue }
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
		std::tie(mMaterials, mImageSamplers, mDrawCalls, mMaterialInfo) = helpers::load_models_and_scenes_from_file({
			// Load a scene from file (path according to the Visual Studio filters!), and apply a transformation matrix (identity, here):
			  { "assets/sponza_and_terrain.fscene",                                 glm::mat4{1.0f} }
		}
		, mQueue
		);

		// Initialize helper for Imaged Based Lighting (modifies mDrawCalls)
		mIblHelper.initialize(mDrawCalls, mMaterialInfo
			, mQueue
		);

		// Create helper geometry for the skybox:
		mSkyboxSphere.create_sphere();
		
		mUniformsBuffer = context().create_buffer(
			memory_usage::host_visible, {}, // Create its backing memory in a host visible memory region (writable from the host-side)
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

		// Create the graphics pipelines for drawing the scene:
		init_pipelines();
		// Initialize the GUI, which is drawn through ImGui:
		init_gui();
		// Enable swapchain recreation and shader hot reloading:
		enable_the_updater();
	}

	/**	Helper function, which creates a renderpass with three sub passes, and
	 *	three graphics pipelines---one for each of the renderpass' sub passes.
	 *
	 *	Also a compute shader-based lighting pass pipeline is created (which
	 *	is not compatible with renderpasses at all; renderpasses are only a
	 *	concept that is to be used with graphics pipelines.)
	 */
	void init_pipelines()
	{
		using namespace avk;

		// Print info for Task 3:
		LOG_INFO(std::format("Maximum supported framebuffer color sample count on this GPU: {}", vk::to_string(context().physical_device().getProperties().limits.framebufferColorSampleCounts)));
		LOG_INFO(std::format("Maximum supported framebuffer depth sample count on this GPU: {}", vk::to_string(context().physical_device().getProperties().limits.framebufferDepthSampleCounts)));

		//
		// TODO Task 3: Decide upon a sample count and create the attachments with sample counts > 1
		//              Furthermore, create one depth and one color attachment with sample count = 1
		//              At a suitable location, you've got to resolve multisampled buffers (one of
		//              each) into those attachments with sample count = 1. Handle everything via
		//              renderpass sub pass declarations!
		//
		// Hint: You'll find all kinds of subpass usages in the namespace avk::usage::
		//       While operator>> switches to the next subpass usage, so to say,
		//       there's also operator+, which can ADD another usage to the current
		//       subpass usage.
		//       Example: An attachment shall be used as color attachment at location=0 AND
		//                it shall be resolved into the attachment with attachment-index 5:
		//                      usage::color(0) + usage::resolve_to(5)
		//

		// Create some G-Buffer attachments here, let us first gather some data like formats:
		const auto resolution = context().main_window()->resolution();
		const auto format0 = context().main_window()->swap_chain_image_format();
		const auto format1 = vk::Format::eD32Sfloat;
		const auto format2 = format0;
		const auto format3 = vk::Format::eR16G16B16A16Sfloat; // normals
		const auto format4 = vk::Format::eR8G8B8A8Unorm; // ambient
		const auto format5 = vk::Format::eR8G8B8A8Unorm; // emissive
		const auto format6 = vk::Format::eR8G8B8A8Unorm; // diffuse
		const auto format7 = vk::Format::eR8G8B8A8Unorm; // spec
		const auto format8 = vk::Format::eR8G8B8A8Unorm; // shininess
		const auto storageFormat = vk::Format::eR8G8B8A8Unorm;

		const auto sampleCount = vk::SampleCountFlagBits::e4;

		// Create one G-Buffer attachment,   same resolution as backbuffer   vvv,   same format as backbuffer   vvv,   in device memory   vvv,   and with appropriate usage flags   vvv
		auto attachment0 = context().create_image(resolution.x, resolution.y, std::make_tuple(format0, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Colour
		auto attachment1 = context().create_image(resolution.x, resolution.y, std::make_tuple(format1, sampleCount), 1, memory_usage::device, image_usage::depth_stencil_attachment | image_usage::input_attachment | image_usage::transfer_source | image_usage::sampled); // Depth
		auto attachment2 = context().create_image(resolution.x, resolution.y, std::make_tuple(format2, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled);  // Output Colour
		auto attachment3 = context().create_image(resolution.x, resolution.y, std::make_tuple(format3, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Normal
		auto attachment4 = context().create_image(resolution.x, resolution.y, std::make_tuple(format4, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Ambient
		auto attachment5 = context().create_image(resolution.x, resolution.y, std::make_tuple(format5, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Emmisive
		auto attachment6 = context().create_image(resolution.x, resolution.y, std::make_tuple(format6, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Diffuse
		auto attachment7 = context().create_image(resolution.x, resolution.y, std::make_tuple(format7, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Spec
		auto attachment8 = context().create_image(resolution.x, resolution.y, std::make_tuple(format8, sampleCount), 1, memory_usage::device, image_usage::color_attachment | image_usage::input_attachment | image_usage::sampled); // Shininess

		auto attachment9  = context().create_image(resolution.x, resolution.y, std::make_tuple(format1, vk::SampleCountFlagBits::e1), 1, memory_usage::device, image_usage::depth_stencil_attachment | image_usage::input_attachment | image_usage::transfer_source | image_usage::sampled); // Flat Depth
		auto attachment10 = context().create_image(resolution.x, resolution.y, std::make_tuple(format2, vk::SampleCountFlagBits::e1), 1, memory_usage::device, image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled); // Flat colour

		// We will use the storage image starting with Task 4 to store results into from a compute shader.
		//
		// Note: sRGB formats are, unfortunately, not supported for storage images on many GPUs.
		//
		auto storageImage = context().create_image(resolution.x, resolution.y, storageFormat, 1, memory_usage::device, image_usage::shader_storage | image_usage::color_attachment | image_usage::transfer_source | image_usage::sampled);

		//
		// TODO Task 1 and possibly also any further task (according to your chosen G-Buffer layout):
		//             Transition every additional attachment once from undefined >> shader_read_only_optimal layout!
		//
		// layout::shader_read_only_optimal will serve as our go-to layout for many cases. One of the reasons for that is
		// that we'd like to render the G-Buffer attachments with ImGui, and that layout is optimal for that usage.
		//
		// Hint: If you get strange (any oftentimes annoying) image layout transition validaiton errors, try the following:
		//       - Add .from_previous_layout(layout::shader_read_only_optimal) to your on_load attachment declarations!
		//           e.g.: on_load::clear.from_previous_layout(layout::shader_read_only_optimal)
		//       - Add .in_layout(layout::shader_read_only_optimal) to your on_store attachment declarations!
		//           e.g.: on_store::store.in_layout(layout::shader_read_only_optimal)
		//
		// FYI: Image layout transition validation errors will NOT lead to point deductions!
		//      Would be nice if you could try to fix them, regardless. Please use the TUWEL forum for questions!
		//

		// Before using all these G-Buffer attachments => Let's transition their layouts into something useful:
		auto fen = context().record_and_submit_with_fence(command::gather(
				sync::image_memory_barrier(attachment0.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment1.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment2.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment3.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment4.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment5.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment6.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment7.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment8.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment9.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				sync::image_memory_barrier(attachment10.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::shader_read_only_optimal),
				// Transition the storage image into GENERAL layout and keep it in that layout forever:
				sync::image_memory_barrier(storageImage.as_reference(), stage::none >> stage::none).with_layout_transition(layout::undefined >> layout::general),
				// Also transition all the backbuffer images into a useful initial layout:
				context().main_window()->layout_transitions_for_all_backbuffer_images()
			), *mQueue);
		fen->wait_until_signalled();
		
		// Before we can attach the G-Buffer images to a framebuffer, we have to wrap them with an image view:
		auto view0 = context().create_image_view(std::move(attachment0));
		auto view1 = context().create_image_view(std::move(attachment1));
		auto view2 = context().create_image_view(std::move(attachment2));
		auto view3 = context().create_image_view(std::move(attachment3));
		auto view4 = context().create_image_view(std::move(attachment4));
		auto view5 = context().create_image_view(std::move(attachment5));
		auto view6 = context().create_image_view(std::move(attachment6));
		auto view7 = context().create_image_view(std::move(attachment7));
		auto view8 = context().create_image_view(std::move(attachment8));
		auto view9 = context().create_image_view(std::move(attachment9));
		auto view10 = context().create_image_view(std::move(attachment10));

		mStorageImageView = context().create_image_view(std::move(storageImage));

		// A renderpass is used to describe some configuration parts of a graphics pipeline, namely:
		//  - Which kinds of attachments are used and for how many sub passes
		//  - What dependencies are necessary between sub passes, to achieve correct rendering results
		auto renderpass = context().create_renderpass(
			//
			// TODO Task 1: Declare for each and every attachment how it is going to be used in both subpasses!
			// 
			// TODO Task 3: Since all the G-Buffer attachments MUST be multisampled formats, add resolve operations at some point 
			//              for both, the multisampled depth buffer and the multisampled color buffer!
			//              The buffers with sample counts of 1 must be used in render() for the blit/copy operations!
			// 
			// Hint 1: Different places are possible for the resolve operations! Think about where it would make most sense to put them!
			// 
			// Hint 2: Should a resolve operation result in a black image, please try to declare the affected attachment's load operation
			//         as on_load::load (instead of on_load::clear). Some GPUs might decide to clear an attachment when it is FIRST
			//         used as, e.g., a color attachment. If that first color attachment usage happens AFTER the resolve operation, it
			//         can be that the attachment is cleared after the resolve. on_load::load should fix this.
			//
			{ // We have THREE sub passes here!   vvv    To properly set this up, we need to define for every attachment, how it is used in each single one of these THREE sub passes    vvv
				attachment::declare(std::make_tuple(format0, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(0)	    >> usage::input(0) >> usage::preserve,                             on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format1, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::depth_stencil >> usage::input(1) >> usage::depth_stencil + usage::resolve_to(9), on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format2, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::unused        >> usage::color(0) >> usage::color(0) + usage::resolve_to(10),	   on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format3, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(1)	    >> usage::input(2) >> usage::preserve,                             on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format4, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(2)	    >> usage::input(3) >> usage::preserve,							   on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format5, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(3)	    >> usage::input(4) >> usage::preserve,							   on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format6, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(4)	    >> usage::input(5) >> usage::preserve,							   on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format7, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(5)	    >> usage::input(6) >> usage::preserve,							   on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format8, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(6)	    >> usage::input(7) >> usage::preserve,							   on_store::store.in_layout(layout::shader_read_only_optimal)),

				attachment::declare(std::make_tuple(format1, vk::SampleCountFlagBits::e1), on_load::load.from_previous_layout(layout::shader_read_only_optimal), usage::unused >> usage::unused >> usage::unused, on_store::store.in_layout(layout::shader_read_only_optimal)), // Depth resolve target
				attachment::declare(std::make_tuple(format2, vk::SampleCountFlagBits::e1), on_load::load.from_previous_layout(layout::shader_read_only_optimal), usage::unused >> usage::unused >> usage::unused, on_store::store.in_layout(layout::shader_read_only_optimal)), // Color resolve target
			},
			{ // Describe the dependencies between external commands and the FIRST sub pass:
                subpass_dependency( subpass::external   >>   subpass::index(0),
									// vvv Actually we would not have to wait for anything before... but for the layout transition we do have to (it's complicated)   >>   depth reads/writes or color writes
					    			stage::color_attachment_output   >>  stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
									access::none                     >>  access::depth_stencil_attachment_read | access::depth_stencil_attachment_write | access::color_attachment_write
								  ),
				// Describe the dependencies between the FIRST and the SECOND sub pass:
				subpass_dependency( subpass::index(0)   >>   subpass::index(1),
					    			stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output   >>   stage::fragment_shader,
									access::depth_stencil_attachment_write | access::color_attachment_write                     >>   access::input_attachment_read
								  ),
				// Describe the dependencies between the SECOND and the THIRD sub pass:
				subpass_dependency( subpass::index(1)   >>  subpass::index(2),
									stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output   >>   stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
									access::depth_stencil_attachment_write | access::color_attachment_write                     >>   access::depth_stencil_attachment_read | access::depth_stencil_attachment_write | access::color_attachment_write
									// Note: Although this might seem unintuitive, we have to synchronize with read AND write access to the depth/stencil attachment here  ^^^  This is due to the image layout transition (input attachment optimal >> depth/stencil attachment optimal)
								  ),
				// Describe the dependencies between the THIRD sub pass and external commands:
				subpass_dependency( subpass::index(2)  >>  subpass::external,
									stage::color_attachment_output   >>   stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
									access::color_attachment_write   >>   access::depth_stencil_attachment_write | access::color_attachment_write
				                  )
			}
		);

		// 
		// TODO Task 4: Create a new renderpass which contains only ONE sub pass, namely the G-Buffer pass!
		//              Lighting and multisample resolve is to be performed in a compute shader for Task 4.
		//              Pay attention to establish proper sub pass dependencies to the subsequent compute shader!
		//
		// Hint: The external >> index(0) sub pass dependencies must, naturally, be set up in the same way as above.
		//       But for the sub pass dependencies index(0) >> external, think about which WRITE operations performed
		//       inside this single sub pass at index(0) need to be made available to which READ operations of the
		//       subsequently executing compute shader!
		//
		//       These are the possible stages: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits2KHR.html
		//       These are the possible access types: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkAccessFlagBits2.html
		//       All stages have corresponding values in stage::, and all access types have corresponding values in access::
		//
		// Note: What to do with the skybox?
		//       How and where will you draw it?
		//

		auto renderpass2 = context().create_renderpass(
			{ // We have ONE sub pass here!   vvv    To properly set this up, we need to define for every attachment, how it is used in the sub pass    vvv
				attachment::declare(std::make_tuple(format0, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(0),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format1, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::depth_stencil + usage::resolve_to(9),	on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format2, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::unused + usage::resolve_to(10),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format3, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(1),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format4, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(2),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format5, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(3),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format6, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(4),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format7, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(5),			on_store::store.in_layout(layout::shader_read_only_optimal)),
				attachment::declare(std::make_tuple(format8, sampleCount),  on_load::clear.from_previous_layout(layout::shader_read_only_optimal), usage::color(6),			on_store::store.in_layout(layout::shader_read_only_optimal)),

				attachment::declare(std::make_tuple(format1, vk::SampleCountFlagBits::e1), on_load::load.from_previous_layout(layout::shader_read_only_optimal), usage::unused, on_store::store.in_layout(layout::shader_read_only_optimal)), // Depth resolve target
				attachment::declare(std::make_tuple(format2, vk::SampleCountFlagBits::e1), on_load::load.from_previous_layout(layout::shader_read_only_optimal), usage::unused, on_store::store.in_layout(layout::shader_read_only_optimal)), // Color resolve target
			},
			{ 
				// Describe the dependencies between external commands and the sub pass:
				subpass_dependency(subpass::external >> subpass::index(0),
					// vvv Actually we would not have to wait for anything before... but for the layout transition we do have to (it's complicated)   >>   depth reads/writes or color writes
					stage::color_attachment_output >> stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
					access::none >> access::depth_stencil_attachment_read | access::depth_stencil_attachment_write | access::color_attachment_write
				),
				// Subpass 0 to External dependency (considering compute shader usage)
				subpass_dependency(subpass::index(0) >> subpass::external,
					stage::color_attachment_output >> stage::compute_shader,
					access::color_attachment_write >> access::shader_read
				)
			}
		);

		// renderpassSkybox -> blitted output + mltspampled depth buffer (no mltsampled colour buffer- previously resolved)

		auto skyboxRenderpass = context().create_renderpass(
			{
				attachment::declare(std::make_tuple(format2, vk::SampleCountFlagBits::e1), on_load::load.from_previous_layout(layout::shader_read_only_optimal), usage::color(0), on_store::store.in_layout(layout::shader_read_only_optimal)), // Color resolve target
				attachment::declare(std::make_tuple(format1, vk::SampleCountFlagBits::e1), on_load::load.from_previous_layout(layout::shader_read_only_optimal), usage::depth_stencil, on_store::store.in_layout(layout::shader_read_only_optimal)), // Depth resolve target
			},
			{
				// Describe the dependencies between external commands and the sub pass:
				subpass_dependency(subpass::external >> subpass::index(0),
				// vvv Actually we would not have to wait for anything before... but for the layout transition we do have to (it's complicated)   >>   depth reads/writes or color writes
				stage::color_attachment_output >> stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
				access::none >> access::depth_stencil_attachment_read | access::depth_stencil_attachment_write | access::color_attachment_write
				),
				// Subpass 0 to External dependency (considering compute shader usage)
				subpass_dependency(subpass::index(0) >> subpass::external,
					stage::color_attachment_output >> stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
					access::color_attachment_write >> access::depth_stencil_attachment_write | access::color_attachment_write
				)
			}
		);
		
		// With images, image views, and renderpass described, let us create a separate framebuffer
		// which we will render into during all the passes described above:
		mFramebuffer = context().create_framebuffer(
			// It is composed of the renderpass:
			renderpass,
			avk::make_vector( // ...and of all the image views:
				  view0 // Let us just pass all of them in a "shared" manner, which 
				, view1 // means that they can be used at other places, too, and 
				, view2 // will lead them being stored in shared_ptrs internally.
				, view3
				, view4
				, view5
				, view6
				, view7
				, view8
				, view9
				, view10
			)
		);

		mFramebuffer2 = context().create_framebuffer(
			// It is composed of the renderpass:
			renderpass2,
			avk::make_vector( // ...and of all the image views:
				  view0 // Let us just pass all of them in a "shared" manner, which 
				, view1 // means that they can be used at other places, too, and 
				, view2 // will lead them being stored in shared_ptrs internally.
				, view3
				, view4
				, view5
				, view6
				, view7
				, view8
				, view9
				, view10
			)
		);

		mSkyboxFramebuffer = context().create_framebuffer(
			// It is composed of the renderpass:
			skyboxRenderpass,
			avk::make_vector( // ...and of all the image views:
				  view10
				, view9
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

		mGBufferPassPipeline2 = context().create_graphics_pipeline_for(
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
			renderpass2, cfg::subpass_index{ 0u },

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

		// Create an (almost identical) pipeline to render the scene in wireframe mode
		mGBufferPassWireframePipeline2 = context().create_graphics_pipeline_from_template(mGBufferPassPipeline2.as_reference(), [](graphics_pipeline_t& p) {
			p.rasterization_state_create_info().setPolygonMode(vk::PolygonMode::eLine);
			});


		// 
		// TODO Task 3: Declare that the mLightingPassGraphicsPipeline shall be executed PER SAMPLE and not per fragment!
		//              This can be declared by passing cfg::shade_per_sample() to create_graphics_pipeline_for
		//

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
			cfg::shade_per_sample(),
			cfg::depth_test::disabled(),

			// Define push constants and resource descriptors which are to be used with this draw call:
			push_constant_binding_data{ shader_type::vertex | shader_type::fragment | shader_type::tessellation_control | shader_type::tessellation_evaluation, 0, sizeof(push_constants_for_draw) },
			descriptor_binding(0, 0, mMaterials),
			descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
			descriptor_binding(1, 0, mUniformsBuffer),
			descriptor_binding(1, 1, mLightsBuffer),

			descriptor_binding(2, 0, mFramebuffer->image_view_at(0)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Color buffer
			descriptor_binding(2, 1, mFramebuffer->image_view_at(1)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Depth buffer
			descriptor_binding(2, 2, mFramebuffer->image_view_at(3)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Position buffer -> Normal
			descriptor_binding(2, 3, mFramebuffer->image_view_at(4)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Normal buffer -> Ambient
			descriptor_binding(2, 4, mFramebuffer->image_view_at(5)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Ambient buffer -> So on..
			descriptor_binding(2, 5, mFramebuffer->image_view_at(6)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Emissive buffer
			descriptor_binding(2, 6, mFramebuffer->image_view_at(7)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Diffuse buffer
			descriptor_binding(2, 7, mFramebuffer->image_view_at(8)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Specular buffer

			descriptor_binding(3, 0, mIblHelper.get_irradiance_map()             ->as_combined_image_sampler(layout::shader_read_only_optimal), shader_type::fragment),
			descriptor_binding(3, 1, mIblHelper.get_prefiltered_environment_map()->as_combined_image_sampler(layout::shader_read_only_optimal), shader_type::fragment),
			descriptor_binding(3, 2, mIblHelper.get_brdf_lookup_table()          ->as_combined_image_sampler(layout::shader_read_only_optimal), shader_type::fragment)
		);

		//
		// TODO Task 4: Use the following compute pipeline for the lighting pass instead of the graphics pipeline-based lighting pass!
		//
		mLightingPassComputePipeline = context().create_compute_pipeline_for(
			compute_shader("shaders/lighting_pass.comp"),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_dispatch) },
			descriptor_binding(0, 0, mUniformsBuffer),
			descriptor_binding(0, 1, mLightsBuffer),

			descriptor_binding(1, 0, mFramebuffer2->image_view_at(0)->as_sampled_image(layout::shader_read_only_optimal)), // Gotta specify the layouts of the images here! This one is read-only.
			descriptor_binding(1, 1, mFramebuffer2->image_view_at(2)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 2, mFramebuffer2->image_view_at(3)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 3, mFramebuffer2->image_view_at(4)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 4, mFramebuffer2->image_view_at(5)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 5, mFramebuffer2->image_view_at(6)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 6, mFramebuffer2->image_view_at(7)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 7, mFramebuffer2->image_view_at(8)->as_sampled_image(layout::shader_read_only_optimal)),
			descriptor_binding(1, 8, mStorageImageView->as_storage_image(layout::general))       // Since this image will be written to, we need to use a different layout.
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

			descriptor_binding(0, 0, mUniformsBuffer) // Doesn't have to be the exact buffer, but one that describes the correct layout for the pipeline.
		);

		mSkyboxPipeline2 = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/sky_gradient.vert"),
			fragment_shader("shaders/sky_gradient.frag"),
			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // Stream positions from the vertex buffer bound at index #0

			// Use the renderpass created above, and specify that we're intending to use this pipeline for its First subpass:
			skyboxRenderpass, cfg::subpass_index{ 0u },

			// Configuration parameters for this graphics pipeline:
			cfg::culling_mode::disabled,	// No backface culling required
			cfg::depth_test::enabled().set_compare_operation(cfg::compare_operation::less_or_equal), // as the depth buffer is cleared to 1.0 and the skybox clamps to 1.0, we need a <= comparison
			cfg::depth_write::disabled(),	// Don't write depth values
			cfg::depth_bounds::enable(1.0f, 1.0f),
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),

			descriptor_binding(0, 0, mUniformsBuffer) // Doesn't have to be the exact buffer, but one that describes the correct layout for the pipeline.
		);

		// Create the graphics pipeline used for drawing the skybox with image-based lighting in the Bonus Task 2:
		// (This is exactly the same as mSkyboxPipeline, except for the shaders and one additional descriptor binding)
		mSkyboxPipelineIbl = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/skybox_for_ibl.vert"),
			fragment_shader("shaders/skybox_for_ibl.frag"),
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

			descriptor_binding(0, 0, mUniformsBuffer), // Doesn't have to be the exact buffer, but one that describes the correct layout for the pipeline.
			descriptor_binding(0, 1, mIblHelper.get_background_image_sampler()->as_combined_image_sampler(layout::shader_read_only_optimal))
		);

		mSkyboxPipelineIbl2 = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/skybox_for_ibl.vert"),
			fragment_shader("shaders/skybox_for_ibl.frag"),
			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // Stream positions from the vertex buffer bound at index #0

			// Use the renderpass created above, and specify that we're intending to use this pipeline for its THIRD subpass:
			skyboxRenderpass, cfg::subpass_index{ 0u },

			// Configuration parameters for this graphics pipeline:
			cfg::culling_mode::disabled,	// No backface culling required
			cfg::depth_test::enabled().set_compare_operation(cfg::compare_operation::less_or_equal), // as the depth buffer is cleared to 1.0 and the skybox clamps to 1.0, we need a <= comparison
			cfg::depth_write::disabled(),	// Don't write depth values
			cfg::depth_bounds::enable(1.0f, 1.0f),
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),

			descriptor_binding(0, 0, mUniformsBuffer), // Doesn't have to be the exact buffer, but one that describes the correct layout for the pipeline.
			descriptor_binding(0, 1, mIblHelper.get_background_image_sampler()->as_combined_image_sampler(layout::shader_read_only_optimal))
		);


	}

	/**	Helper function, which sets up drawing of the GUI at initialization time.
	 *	For that purpose, it gets a handle to the imgui_manager component and installs a callback.
	 *	The GUI is drawn using the library Dear ImGui: https://github.com/ocornut/imgui
	 */
	void init_gui()
	{
		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (nullptr == imguiManager) {
			LOG_ERROR("Failed to init GUI, because composition does not contain an element of type avk::imgui_manager.");
			return;
		}

		std::vector<std::tuple<std::string, avk::image_sampler>> gBufferTextures;
		auto sampler = avk::context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_border, 0.0f);
		int attachmentId = 0;
		for (auto& attachment : mFramebuffer->image_views()) {
			auto& tpl = gBufferTextures.emplace_back();
			if (attachment->get_image().create_info().samples != vk::SampleCountFlagBits::e1) {
				std::get<std::string>(tpl) = std::format("Not rendering attachment #{} due to its sample count of {}", attachmentId++, vk::to_string(attachment->get_image().create_info().samples));
			}
			else {
				std::get<std::string>(tpl) = std::format("Attachment {}:", attachmentId++);
				std::get<avk::image_sampler>(tpl) = avk::context().create_image_sampler(attachment, sampler);
			}
		}

		// Install a callback which will be invoked each time imguiManager's render() is invoked by the framework:
		imguiManager->add_callback([this, lGBufferTextures = std::move(gBufferTextures), imguiManager]() {
			ImGui::Begin("Settings");
			ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
			ImGui::SetWindowSize(ImVec2(280.0f, 1000.0f), ImGuiCond_FirstUseEver);
			ImGui::Text("%.3f ms (%.1f fps)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

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
			ImGui::Checkbox("Compute Shader", &mComputeShaderEnabled);
			ImGui::Separator();
			if (ImGui::CollapsingHeader("Physically Based Shading")) {
				ImGui::PushID("PBS");
				ImGui::Checkbox("Enable PBS", &mPbsEnable);
				ImGui::SliderFloat("Roughn.-Scaling", &mUserDefinedRoughnessStrength, 0.f, 1.f);
				ImGui::SliderFloat("Light boost", &mPbsLightBoost, 1.f, 4.f);
				ImGui::Checkbox("Override values from textures:", &mPbsOverride.enable);
				ImGui::Indent(4.f);
				ImGui::SliderFloat("Roughness", &mPbsOverride.roughness, 0.f, 1.f);
				ImGui::SliderFloat("Metallic",  &mPbsOverride.metallic,  0.f, 1.f);
				ImGui::Unindent(4.f);
				ImGui::PopID();
			}
			if (ImGui::CollapsingHeader("Image Based Lighting")) {
				ImGui::PushID("IBL");
				ImGui::Checkbox("Enable IBL", &mIblEnable);
				int geo = mIblHelper.get_geometry_to_render();
				int oldgeo = geo;
				ImGui::RadioButton("Sponza Vase", &geo, 0); ImGui::SameLine();
				ImGui::RadioButton("Sphere Grid", &geo, 1);
				if (geo != oldgeo) mIblHelper.set_geometry_to_render(geo);
				if (geo == 1) {
					int mat = mIblHelper.get_material_index_to_use();
					if (ImGui::SliderInt("Spheres material", &mat, 0, int(mMaterialInfo.mNumMaterialsInGpuBuffer) - 1)) {
						if (mat < 0) mat = 0;
						if (mat >= mMaterialInfo.mNumMaterialsInGpuBuffer) mat = int(mMaterialInfo.mNumMaterialsInGpuBuffer) - 1;
						mIblHelper.set_material_index_to_use(mat);
					}
					ImGui::Text("> "); ImGui::SameLine();
					ImGui::Text(mMaterialInfo.mMaterialNames[mat].c_str());
					bool texdata = mIblHelper.get_use_texture_pbr_data();
					if (ImGui::Checkbox("Use rough./metallic from textures", &texdata)) mIblHelper.set_use_texture_pbr_data(texdata);
				}
				bool rot = mIblHelper.get_rotate();
				if (ImGui::Checkbox("Rotate", &rot)) mIblHelper.set_rotate(rot);
				if (ImGui::Button("Rebuild maps")) mIblHelper.invalidate_maps();
				ImGui::PopID();
			}

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
					ImTextureID texId = imguiManager->get_or_create_texture_descriptor(std::get<avk::image_sampler>(tpl).get(), avk::layout::shader_read_only_optimal);
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
			.invoke([this] { // Fix camera aspect ratios:
				mOrbitCam.set_aspect_ratio(context().main_window()->aspect_ratio());
				mQuakeCam.set_aspect_ratio(context().main_window()->aspect_ratio());
			})
			.update(mGBufferPassPipeline) // Update the pipeline after the swap chain has changed
			.update(mGBufferPassWireframePipeline) // this one too
			.update(mLightingPassGraphicsPipeline)
			.update(mSkyboxPipeline) // and the pipeline for drawing the skybox as well
			.update(mSkyboxPipelineIbl);


		// Also enable shader hot reloading via the updater:
		mUpdater->on(shader_files_changed_event(mGBufferPassPipeline.as_reference()))
			.update(mGBufferPassPipeline);
		mUpdater->on(shader_files_changed_event(mGBufferPassWireframePipeline.as_reference()))
			.update(mGBufferPassWireframePipeline);
		mUpdater->on(shader_files_changed_event(mLightingPassGraphicsPipeline.as_reference()))
			.update(mLightingPassGraphicsPipeline);
		mUpdater->on(shader_files_changed_event(mLightingPassComputePipeline.as_reference()))
			.update(mLightingPassComputePipeline);
		mUpdater->on(shader_files_changed_event(mSkyboxPipeline.as_reference()))
			.update(mSkyboxPipeline);
		mUpdater->on(shader_files_changed_event(mSkyboxPipelineIbl.as_reference()))
			.update(mSkyboxPipelineIbl);

		// Also enable shader hot reloading for the IBL map-building shader
		mIblHelper.make_shaders_hot_reloadable(mUpdater);
	}

	// ----------------------- ^^^   INITIALIZATION   ^^^ -----------------------
	//
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
	}

	/**	Render callback which is invoked by the framework every frame after every update() callback has been invoked.
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

		// If we use Image Based Lighting, compute the corresponding maps (this only needs to be done once)
		if (mIblEnable && !mIblHelper.are_maps_initialized()) {
			mIblHelper.build_maps(mQueue, mCommandPool, mDescriptorCache);
		}

		// As described above, we get a semaphore from the framework which will get signaled as soon as
		// the next swap chain image becomes available. Only after it has become available, we may start
		// rendering the current frame into it.
		// We get the semaphore here, and use it further down to describe a dependency of our recorded commands:
		auto imageAvailableSemaphore = context().main_window()->consume_current_image_available_semaphore();

		//auto frameIndex = context().main_window()->in_flight_index_for_frame();
		//buffer& currentUniformsBuffer = mUniformsBuffer[frameIndex];
		//buffer& currentLightsBuffer = mLightsBuffer[frameIndex];
		
		// Update the data in our uniform buffers:
		matrices_and_user_input uni;
		uni.mViewMatrix        = mQuakeCam.view_matrix();
		uni.mProjMatrix        = mQuakeCam.projection_matrix();
		uni.mInverseProjMatrix = glm::inverse(uni.mProjMatrix);
		uni.mCamPos            = glm::translate(mQuakeCam.translation());
		uni.mUserInput         = glm::vec4{ mTessellationLevel, mDisplacementStrength, mPnEnabled ? 1.0f : 0.0f, 0.0f };
		uni.mPbsEnabled = mPbsEnable;
		uni.mUserDefinedRoughnessStrength = mUserDefinedRoughnessStrength;
		uni.mPbsLightBoost = mPbsLightBoost;
		uni.mIblEnabled = mIblEnable;

		// Since this buffer has its backing memory in a "host visible" memory region, we just need to write the new data to it.
		// No need to submit the (empty, in this case!) action_type_command that is returned by buffer_t::fill() to a queue.
		// If its backing memory was in a "device" memory region, we would have to, though (see lights buffer below for the difference!).
		mUniformsBuffer->fill(&uni, 0);

		// Animate lights:
		static auto startTime = static_cast<float>(context().get_time());
		helpers::animate_lights(helpers::get_lights(), static_cast<float>(context().get_time()) - startTime);

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
		auto* scenePipeline = mWireframeMode ? &mGBufferPassWireframePipeline.as_reference() : &mGBufferPassPipeline.as_reference();
		auto* skyboxPipeline = mIblEnable ? &mSkyboxPipelineIbl.as_reference() : &mSkyboxPipeline.as_reference(); // (for Bonus Task 2, we use a different skybox)
		auto* framebuffer = &mFramebuffer.as_reference();

		auto* scenePipeline2 = mWireframeMode ? &mGBufferPassWireframePipeline2.as_reference() : &mGBufferPassPipeline2.as_reference();
		auto* skyboxPipeline2 = mIblEnable ? &mSkyboxPipelineIbl2.as_reference() : &mSkyboxPipeline2.as_reference(); // (for Bonus Task 2, we use a different skybox)
		auto* framebuffer2 = &mFramebuffer2.as_reference();

		auto* skyboxFramebuffer = &mSkyboxFramebuffer.as_reference();

		// We'll blit/copy from these two indices into the backbuffer images:
		//
		// TODO Task 1 and possibly also any further task (according to your chosen G-Buffer layout):
		//             Change the following indices to copy/blit the appropriate attachments
		//             from mFramebuffer into the current backbuffer attachments!
		//
		// FYI: Copy and blit are performed further down, in the middle of those many image layout transitions.
		//
		const auto* depthSrc = &mFramebuffer->image_at(9);
		const auto* colorSrc = &mFramebuffer->image_at(10);

		auto finalDepthSrcLayout = layout::shader_read_only_optimal;
		auto finalColorSrcLayout = layout::shader_read_only_optimal;

		constexpr auto WORKGROUP_SIZE = uint32_t{ 16u };
		const auto resolution = context().main_window()->resolution();

		context().record(command::gather( // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):
			command::conditional([this] {return !mComputeShaderEnabled;  }, 
				[&] 
				{
					return command::gather(
						command::begin_render_pass_for_framebuffer(
							scenePipeline->renderpass_reference(), // <-- Use the renderpass of mGBufferPassPipeline (or mGBufferPassWireframePipeline),
							*framebuffer
						),

						// Bind the pipeline for subsequent draw calls:
						command::bind_pipeline(*scenePipeline),
						// Bind all resources we need in shaders:
						command::bind_descriptors(scenePipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mMaterials),
							descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
							descriptor_binding(1, 0, mUniformsBuffer),
							descriptor_binding(1, 1, mLightsBuffer)
							})),

						// If IBL is not active, render the normal scene geometry...
						command::conditional([this] { return !mIblEnable; },
							[this, &scenePipeline] {
								return command::many_for_each(mDrawCalls, [this, &scenePipeline](const auto& drawCall) {
									return command::gather(
										command::push_constants(scenePipeline->layout(), push_constants_for_draw{ drawCall.mModelMatrix, mPbsOverride.to_vec4(), drawCall.mMaterialIndex }),
										command::draw_indexed(
											drawCall.mIndexBuffer.as_reference(),     // Index buffer
											drawCall.mPositionsBuffer.as_reference(), // Vertex buffer at index #0
											drawCall.mTexCoordsBuffer.as_reference(), // Vertex buffer at index #1
											drawCall.mNormalsBuffer.as_reference(),   // Vertex buffer at index #2
											drawCall.mTangentsBuffer.as_reference(),  // Vertex buffer at index #3
											drawCall.mBitangentsBuffer.as_reference() // Vertex buffer at index #4
										)
									);
									});
							},
							[this, &scenePipeline] {
								return command::gather(command::custom_commands([this, &scenePipeline](avk::command_buffer_t& cb) {
									mIblHelper.render_geometry(cb, mPbsOverride.to_vec4(), [&](const glm::mat4& aModelMatrix, const glm::vec4& aPbsOverride, int aMaterialIndex) {
										cb.record(avk::command::push_constants(scenePipeline->layout(), push_constants_for_draw{ aModelMatrix, aPbsOverride, aMaterialIndex }));
										});
									}));
							}
						),
						command::next_subpass(),

						command::bind_pipeline(*mLightingPassGraphicsPipeline),
						//push_constant_binding_data{ shader_type::vertex | shader_type::fragment | shader_type::tessellation_control | shader_type::tessellation_evaluation, 0, sizeof(push_constants_for_draw) },
						// Bind all resources we need in shaders:
						command::bind_descriptors(mLightingPassGraphicsPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mMaterials),
							descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
							descriptor_binding(1, 0, mUniformsBuffer),
							descriptor_binding(1, 1, mLightsBuffer),
							descriptor_binding(2, 0, framebuffer->image_view_at(0)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Color buffer
							descriptor_binding(2, 1, framebuffer->image_view_at(1)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Depth buffer
							descriptor_binding(2, 2, framebuffer->image_view_at(3)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Position buffer
							descriptor_binding(2, 3, framebuffer->image_view_at(4)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Normal buffer
							descriptor_binding(2, 4, framebuffer->image_view_at(5)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Ambient buffer
							descriptor_binding(2, 5, framebuffer->image_view_at(6)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Emissive buffer
							descriptor_binding(2, 6, framebuffer->image_view_at(7)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Diffuse buffer
							descriptor_binding(2, 7, framebuffer->image_view_at(8)->as_input_attachment(layout::shader_read_only_optimal), shader_type::fragment), // Specular buffer
							descriptor_binding(3, 0, mIblHelper.get_irradiance_map()->as_combined_image_sampler(layout::shader_read_only_optimal), shader_type::fragment),
							descriptor_binding(3, 1, mIblHelper.get_prefiltered_environment_map()->as_combined_image_sampler(layout::shader_read_only_optimal), shader_type::fragment),
							descriptor_binding(3, 2, mIblHelper.get_brdf_lookup_table()->as_combined_image_sampler(layout::shader_read_only_optimal), shader_type::fragment)
							})),
						// Draw a a full-screen quad:
						command::draw(6, 1, 0, 1),

						command::next_subpass()
						,
						command::conditional([this] { return !mWireframeMode; },
							[this, skyboxPipeline] {
								return command::gather(
									command::bind_pipeline(*skyboxPipeline),
									command::conditional([this] { return mIblEnable; },
										[this, skyboxPipeline] {
											return command::bind_descriptors(skyboxPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
												descriptor_binding(0, 0, mUniformsBuffer),
												descriptor_binding(0, 1, mIblHelper.get_background_image_sampler()->as_combined_image_sampler(layout::shader_read_only_optimal))
												}));
										},
										[this, skyboxPipeline] {
											return command::bind_descriptors(skyboxPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
												descriptor_binding(0, 0, mUniformsBuffer)
												}));
										}
									),
									command::draw_indexed(mSkyboxSphere.mIndexBuffer.as_reference(), mSkyboxSphere.mPositionsBuffer.as_reference())
								);
							}
						),
						command::end_render_pass()
					);
				},
				[&] 
				{
					return command::gather(
						command::begin_render_pass_for_framebuffer(
							scenePipeline2->renderpass_reference(), // <-- Use the renderpass of mGBufferPassPipeline (or mGBufferPassWireframePipeline),
							*framebuffer2
						),

						// Bind the pipeline for subsequent draw calls:
						command::bind_pipeline(*scenePipeline2),
						// Bind all resources we need in shaders:
						command::bind_descriptors(scenePipeline2->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mMaterials),
							descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
							descriptor_binding(1, 0, mUniformsBuffer),
							descriptor_binding(1, 1, mLightsBuffer)
							})),

						// If IBL is not active, render the normal scene geometry...
						command::conditional([this] { return !mIblEnable; },
							[this, &scenePipeline2] {
								return command::many_for_each(mDrawCalls, [this, &scenePipeline2](const auto& drawCall) {
									return command::gather(
										command::push_constants(scenePipeline2->layout(), push_constants_for_draw{ drawCall.mModelMatrix, mPbsOverride.to_vec4(), drawCall.mMaterialIndex }),
										command::draw_indexed(
											drawCall.mIndexBuffer.as_reference(),     // Index buffer
											drawCall.mPositionsBuffer.as_reference(), // Vertex buffer at index #0
											drawCall.mTexCoordsBuffer.as_reference(), // Vertex buffer at index #1
											drawCall.mNormalsBuffer.as_reference(),   // Vertex buffer at index #2
											drawCall.mTangentsBuffer.as_reference(),  // Vertex buffer at index #3
											drawCall.mBitangentsBuffer.as_reference() // Vertex buffer at index #4
										)
									);
									});
							},
							[this, &scenePipeline2] {
								return command::gather(command::custom_commands([this, &scenePipeline2](avk::command_buffer_t& cb) {
									mIblHelper.render_geometry(cb, mPbsOverride.to_vec4(), [&](const glm::mat4& aModelMatrix, const glm::vec4& aPbsOverride, int aMaterialIndex) {
										cb.record(avk::command::push_constants(scenePipeline2->layout(), push_constants_for_draw{ aModelMatrix, aPbsOverride, aMaterialIndex }));
										});
									}));
							}
						),
						command::end_render_pass(),

						command::bind_pipeline(*mLightingPassComputePipeline),
						command::bind_descriptors(mLightingPassComputePipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
							descriptor_binding(0, 0, mUniformsBuffer),
							descriptor_binding(0, 1, mLightsBuffer),
							descriptor_binding(1, 0, framebuffer2->image_view_at(0)->as_sampled_image(layout::shader_read_only_optimal)), // Color buffer
							descriptor_binding(1, 1, framebuffer2->image_view_at(1)->as_sampled_image(layout::shader_read_only_optimal)), // Depth buffer
							descriptor_binding(1, 2, framebuffer2->image_view_at(3)->as_sampled_image(layout::shader_read_only_optimal)), // Normal buffer
							descriptor_binding(1, 3, framebuffer2->image_view_at(4)->as_sampled_image(layout::shader_read_only_optimal)), // Ambient buffer
							descriptor_binding(1, 4, framebuffer2->image_view_at(5)->as_sampled_image(layout::shader_read_only_optimal)), // Emissive buffer
							descriptor_binding(1, 5, framebuffer2->image_view_at(6)->as_sampled_image(layout::shader_read_only_optimal)), // Diffuse buffer
							descriptor_binding(1, 6, framebuffer2->image_view_at(7)->as_sampled_image(layout::shader_read_only_optimal)), // Specular buffer
							descriptor_binding(1, 7, framebuffer2->image_view_at(8)->as_sampled_image(layout::shader_read_only_optimal)), // Shininess buffer
							descriptor_binding(1, 8, mStorageImageView->as_storage_image(layout::general))
							})),
						command::dispatch(
							(resolution.x + 15u) / WORKGROUP_SIZE,
							(resolution.y + 15u) / WORKGROUP_SIZE,
							1
						),
						sync::image_memory_barrier(mStorageImageView->get_image(),
							stage::compute_shader >> stage::blit,
							access::shader_storage_write >> access::transfer_read
						).with_layout_transition(layout::general >> layout::transfer_src),

						sync::image_memory_barrier(skyboxFramebuffer->image_at(0), // Color attachment
							// Gotta use a stage  vvv  here to establish a dependency chain for the image layout transition.
							stage::color_attachment_output >> stage::blit,
							access::color_attachment_write >> access::transfer_write
						).with_layout_transition(layout::shader_read_only_optimal >> layout::transfer_dst), // Don't care about the previous layout

						// Blit both, color and depth from the framebuffer -> the window's backbuffer for further usage (in lights editor and ImGui):
						copy_image_to_another(
							mStorageImageView->get_image(), layout::transfer_src,
							skyboxFramebuffer->image_at(0), layout::transfer_dst,
							vk::ImageAspectFlagBits::eColor
						),

						// Transition all the layouts again into whichever layouts we are going to need them next:
						sync::image_memory_barrier(mStorageImageView->get_image(),
							stage::blit >> stage::compute_shader,
							access::none >> access::shader_storage_write
						).with_layout_transition(layout::transfer_src >> layout::general),

						sync::image_memory_barrier(skyboxFramebuffer->image_at(0), // Color attachment
							stage::blit >> stage::fragment_shader,
							access::transfer_write >> access::shader_sampled_read
						).with_layout_transition(layout::transfer_dst >> finalColorSrcLayout),


						command::begin_render_pass_for_framebuffer(
							skyboxPipeline2->renderpass_reference(),
							*skyboxFramebuffer
						),
						command::conditional([this] { return !mWireframeMode; },
							[this, skyboxPipeline2] {
								return command::gather(
									command::bind_pipeline(*skyboxPipeline2),
									command::conditional([this] { return mIblEnable; },
										[this, skyboxPipeline2] {
											return command::bind_descriptors(skyboxPipeline2->layout(), mDescriptorCache->get_or_create_descriptor_sets({
												descriptor_binding(0, 0, mUniformsBuffer),
												descriptor_binding(0, 1, mIblHelper.get_background_image_sampler()->as_combined_image_sampler(layout::shader_read_only_optimal))
												}));
										},
										[this, skyboxPipeline2] {
											return command::bind_descriptors(skyboxPipeline2->layout(), mDescriptorCache->get_or_create_descriptor_sets({
												descriptor_binding(0, 0, mUniformsBuffer)
												}));
										}
									),
									command::draw_indexed(mSkyboxSphere.mIndexBuffer.as_reference(), mSkyboxSphere.mPositionsBuffer.as_reference())
								);
							}
						),
						command::end_render_pass()
					);
				}
				),

				// So far, so good. And now:
				//
				// Copy the results into the backbuffer images, which are the swapchain's images => s.t. we can hand them over to the presentation engine
				//
				//    vvv

				// Transition the layouts of
				//  - The G-Buffer image that contains the final rendered color result
				//  - The backbuffer's color attachment image
				//  - The G-Buffer image that contains the final depth buffer
				//  - The backbuffer's depth attachment image
				// in preparation for copying the data from the G-Buffer images into the backbuffer images:
				sync::image_memory_barrier(*colorSrc,
					stage::color_attachment_output   >>    stage::blit,
					access::color_attachment_write   >>    access::transfer_read
				).with_layout_transition(layout::shader_read_only_optimal >> layout::transfer_src),

				sync::image_memory_barrier(context().main_window()->current_backbuffer()->image_at(0), // Color attachment
					// Gotta use a stage  vvv  here to establish a dependency chain for the image layout transition.
					stage::color_attachment_output   >>    stage::blit,
					access::none                     >>    access::transfer_write
				).with_layout_transition(layout::undefined >> layout::transfer_dst), // Don't care about the previous layout

				sync::image_memory_barrier(*depthSrc,
					stage::early_fragment_tests | stage::late_fragment_tests >> stage::copy,
					access::depth_stencil_attachment_write                   >> access::transfer_read
				).with_layout_transition(layout::shader_read_only_optimal >> layout::transfer_src),

				sync::image_memory_barrier(context().main_window()->current_backbuffer()->image_at(1), // Depth attachment
					stage::none     >>    stage::copy,
					access::none    >>    access::transfer_write
				).with_layout_transition(layout::depth_stencil_attachment_optimal >> layout::transfer_dst),

				// Blit both, color and depth from the framebuffer -> the window's backbuffer for further usage (in lights editor and ImGui):
				blit_image(
					*colorSrc, layout::transfer_src,
					context().main_window()->current_backbuffer()->image_at(0), layout::transfer_dst,
					vk::ImageAspectFlagBits::eColor
				),
				copy_image_to_another(
					*depthSrc, layout::transfer_src,
					context().main_window()->current_backbuffer()->image_at(1), layout::transfer_dst,
					vk::ImageAspectFlagBits::eDepth
				),

				// Transition all the layouts again into whichever layouts we are going to need them next:
				sync::image_memory_barrier(*colorSrc,
					stage::blit     >>     stage::fragment_shader,
					access::none    >>     access::shader_sampled_read
				).with_layout_transition(layout::transfer_src >> finalColorSrcLayout),

				sync::image_memory_barrier(context().main_window()->current_backbuffer()->image_at(0), // Color attachment
					stage::blit              >>    stage::color_attachment_output,
					access::transfer_write   >>    access::color_attachment_write
				).with_layout_transition(layout::transfer_dst >> layout::color_attachment_optimal),

				sync::image_memory_barrier(*depthSrc,
					stage::copy     >>     stage::fragment_shader,
					access::none    >>     access::shader_sampled_read
				).with_layout_transition(layout::transfer_src >> finalDepthSrcLayout),

				sync::image_memory_barrier(context().main_window()->current_backbuffer()->image_at(1), // Depth attachment
					stage::copy              >>    stage::early_fragment_tests | stage::late_fragment_tests,
					access::transfer_write   >>    access::depth_stencil_attachment_read | access::depth_stencil_attachment_write
					// Note: Gotta synchronize with both read and write access to the depth/stencil attachment here  ^^^  due to the image layout transition.
				).with_layout_transition(layout::transfer_dst >> layout::depth_stencil_attachment_optimal),

				// For proper synchronization with G-Buffer attachment display in the UI, we've gotta sync the last access to 
				// each single image with the shader read access (which is how the UI renders them) at some point:
				sync::image_memory_barrier(mFramebuffer->image_at(0),
					stage::color_attachment_output    >>    stage::fragment_shader,
					access::color_attachment_write    >>    access::shader_read
				)

			)) // End of command recording
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
	std::vector<helpers::data_for_draw_call> mDrawCalls;
	/** Info about the loaded materials */
	helpers::LoadedMaterialsInfo mMaterialInfo;

	/** Helper object for Imaged Based Lighting */
	IblHelper mIblHelper;

	// A bunch of cameras:
	avk::orbit_camera mOrbitCam;
	avk::quake_camera mQuakeCam;
	/** A framebuffer for the G-Buffer pass: */
	avk::framebuffer mFramebuffer;
	avk::framebuffer mFramebuffer2;
	avk::framebuffer mSkyboxFramebuffer;

	/** An image that is used in a compute shader to store color values into:  */
	avk::image_view mStorageImageView;

	avk::graphics_pipeline mGBufferPassPipeline, mGBufferPassWireframePipeline;
	avk::graphics_pipeline mGBufferPassPipeline2, mGBufferPassWireframePipeline2;
	avk::graphics_pipeline mLightingPassGraphicsPipeline;
	avk::compute_pipeline mLightingPassComputePipeline;

	avk::buffer mUniformsBuffer;
	avk::buffer mLightsBuffer;
	
	// ------------------ UI Parameters -------------------
	/** Factor that determines to which amount normals shall be distorted through normal mapping: */
	float mDisplacementStrength = 0.5f;

	float mTessellationLevel = 8.0f;

	/** Flag controlled through the UI, indicating whether the scene shall be rendered with filled polygons or in wireframe mode: */
	bool mWireframeMode = false;
	/** Flag controlled through the UI, indicating whether PN triangles is currently active or not: */
	bool mPnEnabled = true;
	/** Flag controlled through the UI, indicating whether the scene shall be rendered with compute shader based lighting */
	bool mComputeShaderEnabled = false;
	/** master switch to enable physically based shading */
	bool mPbsEnable = false;

	/** user-parameter to scale roughness for physically based shading */
	float mUserDefinedRoughnessStrength = 1.0;

	/** light intensity multiplier for physically based shading */
	float mPbsLightBoost = 2.4f;

	/** override metallic and roughness */
	struct {
		bool  enable = false;
		float metallic = 0.0f;
		float roughness = 0.5f;
		glm::vec4 to_vec4() { return glm::vec4(metallic, roughness, enable ? 1.0f : 0.0f, 0.0f); }
	} mPbsOverride;

	/** master switch to enable image based lighting */
	bool mIblEnable = false;

	int mLimitNumPointlights = 98 + EXTRA_POINTLIGHTS;

	// --------------------- Skybox -----------------------
	simple_geometry mSkyboxSphere;
	avk::graphics_pipeline mSkyboxPipeline;
	avk::graphics_pipeline mSkyboxPipeline2;
	simple_geometry mSkyboxIblCube;
	avk::graphics_pipeline mSkyboxPipelineIbl;
	avk::graphics_pipeline mSkyboxPipelineIbl2;
	avk::command_buffer mSkyboxCommandBuffer;

	// ----------------------- ^^^  MEMBER VARIABLES  ^^^ -----------------------
};

//  Main:
//
// +---------------------------------------+
// |                                       |
// |        ARTR 2024 Assignment 3         |
// |                                       |
// +---------------------------------------+
//
//  For the third time, already.
// 
int main() 
{
	using namespace avk;

	int result = EXIT_FAILURE;

	try {
		// Create a window, set some configuration parameters (also relevant for its swap chain), and open it:
		auto mainWnd = context().create_window("ARTR 2024 Assignment 3");
		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->set_additional_back_buffer_attachments({ attachment::declare(vk::Format::eD32Sfloat, on_load::clear, usage::depth_stencil, on_store::store) });
		mainWnd->enable_resizing(false);
		mainWnd->request_srgb_framebuffer(true);
		mainWnd->set_presentaton_mode(presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(1u); // For simplicity, we are using only one concurrent frame in Assignment 3
		mainWnd->open();

		// Create one single queue which we will submit all command buffers to:
		// (We pass the mainWnd because also presentation shall be submitted to this queue)
		auto& singleQueue = context().create_queue({}, queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->set_queue_family_ownership(singleQueue.family_index());
		mainWnd->set_present_queue(singleQueue);

		// Create an instance of our main class which contains the relevant host code for Assignment 1:
		auto app = assignment3(singleQueue);

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

