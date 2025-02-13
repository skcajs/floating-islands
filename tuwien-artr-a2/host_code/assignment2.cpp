#include "utils/debug_camera.hpp"

/**	Main class for the host code part of ARTR 2024 Assignment 2.
 *
 *	It is derived from avk::invokee, s.t. it can be handed-over to avk::start, which
 *	adds it to a avk::composition internally => its callbacks (such as initialize(),
 *	update(), or render() will be invoked).
 *
 *	Hint: Look out for "TODO Task X" comments!
 */
class assignment2 : public avk::invokee
{
	// ------------------ Structs for transfering data from HOST -> DEVICE ------------------

	/** Struct definition for the vertex data, material index, and model matrix of draw calls.
	 *	In Assignment 2, helpers::load_models_and_scenes_from_file does not return the buffers,
	 *	but only the raw vertex data, and we must put them into buffers manually afterwards.
	 *
	 *	The reason for this ^ behavior is Bonus Task 1, because in order to solve it, we must
	 *	modify the indices for some meshes before loading them into buffers.
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
		bool mEnforceTessellation; // By default terrain and debris are always tessellated, but no other meshes. Set mEnforceTessellation=true to tessellate this mesh as well.
		bool mUsePnAenTessellation;
	};

	/** Struct definition for push constants used for the draw calls of the scene */
	struct push_constants
	{
		glm::mat4 mModelMatrix;
		int mMaterialIndex;
		VkBool32 mEnforceTessellation;
	};
	
	/** Struct definition for data used as UBO across different pipelines, containing matrices and user input */
	struct matrices_and_user_input
	{
		// view matrix as returned from quake_camera
		glm::mat4 mViewMatrix;
		// projection matrix as returned from quake_camera
		glm::mat4 mProjMatrix;
		// transformation matrix which tranforms to camera's position
		glm::mat4 mCamPos;
		// x = tessellation factor, y = displacement strength, z and w unused
		glm::vec4 mUserInput;
		// transformation from the current view space to the debug view space
		glm::mat4 mViewToDebugviewMatrix;
		// adaptive tessellation mode (Task 5)
		int mAdaptiveTessellationMode;
		// frustum culling in tessellation control shader (Task 3)
		VkBool32 mFrustumCullingBeforeTess;
		// backface culling in tessellation control shader (Task 4)
		VkBool32 mBackfaceCullingBeforeTess;
		// displacement antialiasing (Bonus Task 2)
		VkBool32 mDisplacementAntiAliasing;
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
	assignment2(avk::queue& aQueue)
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

		// helpers::load_models_and_scenes_from_file returned only the raw vertex data.
		//  => Put them all into buffers which we can use during rendering:
		std::vector<recorded_commands_t> commandsToBeExcecuted;
		for (auto& data : dataForDrawCalls) {
			// Excluding one blue curtain (of a total of three) by modifying the loaded indices before uploading them to a GPU buffer:
			if (data.mModelName.find("sponza_fabric") != std::string::npos && data.mMeshName == "sponza_326") {
				data.mIndices.erase(std::begin(data.mIndices), std::begin(data.mIndices) + 3*4864);
			}

			bool enforceTessellation = false; // <-- Force tessellation on for this mesh?
			bool usePnAenTessellation = false;  // <-- Store for later that we'd like to render this mesh with the PN-AEN technique.
			if (data.mModelName == "sponza_structure" && data.mMeshName.find("pillar_floor") != std::string::npos) {
				//
				// TODO Bonus Task 1: Uncomment the following two lines of code to 1) enable tessellation for this 
				//                    submesh, and 2) also save the information that we should render it with a 
				//                    (separate) pipeline which implements the PN-AEN technique.
				// 
				// enforceTessellation = true;
				// usePnAenTessellation = true;

				//
				// TODO Bonus Task 1: Adapt the mesh's indices according to the PN-AEN-Triangles technique
				//      Don't limit yourself to the pillars, feel free to select different/all submeshes of sponza!
				// 
			}

			auto [bufferIndices   , commandsIndices   ] = avk::create_buffer<decltype(data.mIndices)   , index_buffer_meta >(data.mIndices   , content_description::index);
			auto [bufferPositions , commandsPositions ] = avk::create_buffer<decltype(data.mPositions) , vertex_buffer_meta>(data.mPositions , content_description::position);
			auto [bufferTexCoords , commandsTexCoords ] = avk::create_buffer<decltype(data.mTexCoords) , vertex_buffer_meta>(data.mTexCoords , content_description::texture_coordinate);
			auto [bufferNormals   , commandsNormals   ] = avk::create_buffer<decltype(data.mNormals)   , vertex_buffer_meta>(data.mNormals   , content_description::normal);
			auto [bufferTangents  , commandsTangents  ] = avk::create_buffer<decltype(data.mTangents)  , vertex_buffer_meta>(data.mTangents  , content_description::tangent);
			auto [bufferBitangents, commandsBitangents] = avk::create_buffer<decltype(data.mBitangents), vertex_buffer_meta>(data.mBitangents, content_description::bitangent);

			mDrawCalls.emplace_back(
				std::move(bufferIndices   ),
				std::move(bufferPositions ),
				std::move(bufferTexCoords ),
				std::move(bufferNormals   ),
				std::move(bufferTangents  ),
				std::move(bufferBitangents),
				data.mMaterialIndex,
				data.mModelMatrix,
				enforceTessellation,
				usePnAenTessellation
			);

			commandsToBeExcecuted.push_back(std::move(commandsIndices));
			commandsToBeExcecuted.push_back(std::move(commandsPositions));
			commandsToBeExcecuted.push_back(std::move(commandsTexCoords));
			commandsToBeExcecuted.push_back(std::move(commandsNormals));
			commandsToBeExcecuted.push_back(std::move(commandsTangents));
			commandsToBeExcecuted.push_back(std::move(commandsBitangents));
		}

		auto fen = context().record_and_submit_with_fence(std::move(commandsToBeExcecuted), *mQueue);
		fen->wait_until_signalled();

		// Create sphere geometry for the skybox
		mSkyboxSphere.create_sphere();

		// Create GPU buffers which will be populated with frame-specific user data (matrices, settings), and lightsource data:
		for (auto i = 0; i < context().main_window()->number_of_frames_in_flight(); ++i) {
			mUniformsBuffer.push_back(context().create_buffer(avk::memory_usage::host_visible, {}, avk::uniform_buffer_meta::create_from_size(sizeof(matrices_and_user_input))));
			mLightsBuffer.push_back(context().create_buffer(avk::memory_usage::device, {}, avk::uniform_buffer_meta::create_from_size(sizeof(lightsource_data))));
		}

		// Initialize the quake_camera, and then add it to our composition (it is a avk::invokee, too):
		mOrbitCam.set_translation({ -6.81f, 1.71f, -0.72f });
		mOrbitCam.look_along({ 1.0f, 0.0f, 0.0f });
		mOrbitCam.set_perspective_projection(glm::radians(60.0f), context().main_window()->aspect_ratio(), 0.1f, 1000.0f);
		current_composition()->add_element(mOrbitCam);

		mQuakeCam.copy_parameters_from(mOrbitCam);
		current_composition()->add_element(mQuakeCam);
		mQuakeCam.disable();

		mDebugCam.copy_parameters_from(mOrbitCam);
		current_composition()->add_element(mDebugCam);

		// Create the graphics pipelines for drawing the scene:
		init_pipelines();
		// Initialize the GUI, which is drawn through ImGui:
		init_gui();
		// Enable swapchain recreation and shader hot reloading:
		enable_the_updater();
	}

	/**	Helper function, which creates the graphics pipelines at initialization time:
	 *	 - mPipeline is relevant for all tasks, renders the whole scene
	 *	 - mSkyboxPipeline is relevant for Bonus Task 2, renders the skybox
	 */
	void init_pipelines()
	{
		using namespace avk;

		// Before defining image usages through a renderpass, let us transition the backbuffer images into useful initial layouts:
		auto fen = context().record_and_submit_with_fence(command::gather(
			context().main_window()->layout_transitions_for_all_backbuffer_images()
		), *mQueue);
		fen->wait_until_signalled();

		// A renderpass is used to describe some configuration parts of a graphics pipeline.
		// More precisely:
		//  1) It describes which kinds of attachments are used and what they are used for.
		//        (In our case, we have two attachments: a color attachment and a depth attachment, both
		//         used for ONE SINGLE SUBPASS, i.e., external commands -> SUBPASS #0 -> external commands)
		//  2) It describes the synchronization for accessing the attachments
		//        (I.e., which stages must wait on previous external commands on the same queue before they can
		//         be executed, and which stages of subsequent commands must wait on what within the renderpass.)
		auto renderpass = context().create_renderpass(
			{ // ad 1) Describe the attachments: One color attachment, and one depth attachment:
				//                                                                       Don't care about previous layout => hence, undefined  vvv
				attachment::declare(format_from_window_color_buffer(context().main_window()),   on_load::clear.from_previous_layout(layout::undefined),  usage::color(0),       on_store::store),
				attachment::declare(format_from_window_depth_buffer(context().main_window()),   on_load::clear,                                          usage::depth_stencil,  on_store::store),
			}, 
			{ // ad 2) Describe the dependency between previous external commands and the first (and only) subpass:
                subpass_dependency( subpass::external   >>  subpass::index(0),
				    // vvv Actually we would not have to wait for anything before... but for the layout transition we do have to (it's complicated)   >>   depth reads/writes or color writes
					stage::color_attachment_output    >>   stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
					access::none                      >>   access::color_attachment_write | access::depth_stencil_attachment_read | access::depth_stencil_attachment_write
				),
				// ad 2) Describe the dependency between (and only) subpass and external subsequent commands:
				subpass_dependency( subpass::index(0)  >>  subpass::external,
				    // vvv Color and depth writes must be finished, before...                                  >>  ...subsequent depth tests, depth writes, or color writes may continue
					stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output  >>  stage::early_fragment_tests | stage::late_fragment_tests | stage::color_attachment_output,
					access::depth_stencil_attachment_write | access::color_attachment_write                    >>  access::color_attachment_write | access::depth_stencil_attachment_read | access::depth_stencil_attachment_write
				)
			}
		);

		// Create a graphics pipeline consisting of a vertex shader and a fragment shader, plus additional config:
		mPipeline = context().create_graphics_pipeline_for(
			vertex_shader("shaders/transform_and_pass_on.vert"),
			//
			// TODO Task 1: Add tessellation control and tessellation evaluation shaders to this pipeline!
			//            
			// Note: Thanks to context().create_graphics_pipeline_from_template which is used below, the
			//       mPipelineWireframe will get these additional shaders and parameters added, too. 
			// 
			tessellation_control_shader("shaders/tess_pn_controlpoints.tesc"),
			tessellation_evaluation_shader("shaders/tess_pn_interp_and_displacement.tese"),
			fragment_shader("shaders/blinnphong_and_normal_mapping.frag"),

			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // Stream positions from the vertex buffer bound at index #0
			from_buffer_binding(1)->stream_per_vertex<glm::vec2>()->to_location(1), // Stream texture coordinates from the vertex buffer bound at index #1
			from_buffer_binding(2)->stream_per_vertex<glm::vec3>()->to_location(2), // Stream normals from the vertex buffer bound at index #2
			from_buffer_binding(3)->stream_per_vertex<glm::vec3>()->to_location(3), // Stream tangents from the vertex buffer bound at index #3
			from_buffer_binding(4)->stream_per_vertex<glm::vec3>()->to_location(4), // Stream bitangents from the vertex buffer bound at index #4

			// Use the renderpass created above:
			renderpass,

			// Configuration parameters for this graphics pipeline:
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),
			//
			// TODO Task 1: Set/change/add pipeline configuration parameters for enabling tessellation!
			// 
			cfg::primitive_topology::patches, // <-- Currently, the pipeline is configured to a primitive topology of triangles
			cfg::tessellation_patch_control_points(3),

			// Define push constants and resource descriptors which are to be used with this draw call:
			push_constant_binding_data{ shader_type::vertex | shader_type::fragment | shader_type::tessellation_control | shader_type::tessellation_evaluation, 0, sizeof(push_constants) },
			descriptor_binding(0, 0, mMaterials),
			descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
			descriptor_binding(1, 0, mUniformsBuffer[0]),
			descriptor_binding(1, 1, mLightsBuffer[0])
		);

		// Create an (almost identical) pipeline to render the scene in wireframe mode
		mPipelineWireframe = context().create_graphics_pipeline_from_template(mPipeline.as_reference(), [](graphics_pipeline_t& p) {
			p.rasterization_state_create_info().setPolygonMode(vk::PolygonMode::eLine);
		});

		// Create the graphics pipeline to be used for drawing the skybox:
		mSkyboxPipeline = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/sky_gradient.vert"),
			fragment_shader("shaders/sky_gradient.frag"),
			from_buffer_binding(0)->stream_per_vertex<glm::vec3>()->to_location(0), // Stream positions from the vertex buffer bound at index #0

			// Use the renderpass created above:

			context().create_renderpass(
				{ // ad 1) Describe the attachments: One color attachment, and one depth attachment:
					//                    vvv Copy the format from the window                       vvv clear it    vvv used as       vvv after renderpass finished, store
					attachment::declare(format_from_window_color_buffer(context().main_window()),   on_load::load, usage::color(0),      on_store::store),
					attachment::declare(format_from_window_depth_buffer(context().main_window()),   on_load::load, usage::depth_stencil, on_store::store),
				},
				renderpass->subpass_dependencies() // copy from the renderpass above
			),

			// Configuration parameters for this graphics pipeline:
			cfg::culling_mode::disabled,	// No backface culling required
			cfg::depth_test::enabled()
			,
			cfg::depth_write::disabled(),	// Don't write depth values
			cfg::viewport_depth_scissors_config::from_framebuffer(
				context().main_window()->backbuffer_reference_at_index(0) // Just use any compatible framebuffer here
			),

			descriptor_binding(0, 0, mUniformsBuffer[0]) // Doesn't have to be the exact buffer, but one that describes the correct layout for the pipeline.
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

		// Install a callback which will be invoked each time imguiManager's render() is invoked by the framework:
		imguiManager->add_callback([this, imguiManager] {
			ImGui::Begin("Settings");
			ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
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

			ImGui::SetNextItemWidth(100);
			ImGui::InputInt("Max point lights", &mLimitNumPointlights, 0, 0);

			// GUI elements for controlling renderin parameters, passed on to mPipeline and mSkyboxPipeline:
			ImGui::Checkbox("Backface Culling", &mBackfaceCullingBeforeTess);
			ImGui::Checkbox("View Frustum Culling", &mFrustumCullingBeforeTess);
			//
			// TODO Task 5: Rename/add tessellation LOD modes here:
			//
			const char* adaptiveTessModes[] = { "Off", "distance", "angle"};
			ImGui::PushItemWidth(100);
			ImGui::Combo("Adaptive Tessellation Mode", &mAdaptiveTessellationMode, adaptiveTessModes, IM_ARRAYSIZE(adaptiveTessModes));
			ImGui::SliderFloat("Tessellation Level", &mTessellationLevel, 1.0f, 32.0f, "%.0f");
			ImGui::SliderFloat("Displacement Strength", &mDisplacementStrength, 0.0f, 1.0f);
			ImGui::PopItemWidth();

			ImGui::Checkbox("Wireframe", &mWireframeMode);


			if (ImGui::Checkbox("Debug camera", &mCullingDebugMode)) {
				if (mCullingDebugMode) {
					mDebugCam.set_translation(mQuakeCam.translation());
					mDebugCam.set_rotation(mQuakeCam.rotation());
				}
			}

			auto* holeChecker = avk::current_composition()->element_by_type<hole_checker>();
			assert(nullptr != holeChecker);
			bool enableHoleCheck = holeChecker->is_enabled();
			if (ImGui::Checkbox("Hole check", &enableHoleCheck)) {
				if (enableHoleCheck) {
					holeChecker->enable();
				}
				else {
					holeChecker->disable();
				}
			}
			if (mLastHoleButtonVisible) {
				ImGui::SameLine();
				if (ImGui::Button("Show##Show hole")) {
					avk::current_composition()->element_by_type<camera_presets>()->invoke_preset("Last Hole Found");
				}
				ImGui::SameLine();
				if (ImGui::Button("x##hide hole button")) {
					mLastHoleButtonVisible = false;
				};
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

			ImGui::Separator();
			// GUI elements for showing camera data, camera presets (interesting perspectives) and the camera presets editor:
			auto camPresets = avk::current_composition()->element_by_type<camera_presets>();
			bool showCamPresets = helpers::is_camera_presets_editor_visible();
			if (ImGui::Checkbox("Camera presets", &showCamPresets)) {
				helpers::set_camera_presets_editor_visible(showCamPresets);
			}

			ImGui::Text(std::format("Cam pos: {}", avk::to_string(mQuakeCam.translation())).c_str());

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
			}) // Fix camera's aspect ratio
			.update(mPipeline) // Update the pipeline after the swap chain has changed
			.update(mPipelineWireframe) // this one too
			.update(mSkyboxPipeline); // and the pipeline for drawing the skybox as well

		// Also enable shader hot reloading via the updater:
		mUpdater->on(shader_files_changed_event(mPipeline.as_reference()))
			.update(mPipeline);
		mUpdater->on(shader_files_changed_event(mPipelineWireframe.as_reference()))
			.update(mPipelineWireframe);
		mUpdater->on(shader_files_changed_event(mSkyboxPipeline.as_reference()))
			.update(mSkyboxPipeline);
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

		// Check if the hole checker found a (new) hole, and if so: update the camera preset
		auto holeInfo = avk::current_composition()->element_by_type<hole_checker>()->get_hole_information();
		if (holeInfo.lastHoleFoundTime != mLastHoleFoundTime) {
			mLastHoleFoundTime = holeInfo.lastHoleFoundTime;
			avk::current_composition()->element_by_type<camera_presets>()->change_location("Last Hole Found", holeInfo.cameraLocation, holeInfo.cameraRotation);
			mLastHoleButtonVisible = true;
		}

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

		// As described above, we get a semaphore from the framework which will get signaled as soon as
		// the next swap chain image becomes available. Only after it has become available, we may start
		// rendering the current frame into it.
		// We get the semaphore here, and use it further down to describe a dependency of our recorded commands:
		auto imageAvailableSemaphore = context().main_window()->consume_current_image_available_semaphore();

		auto frameIndex = context().main_window()->in_flight_index_for_frame();
		buffer& currentUniformsBuffer = mUniformsBuffer[frameIndex];
		buffer& currentLightsBuffer = mLightsBuffer[frameIndex];
		
		// Update the data in our uniform buffers:
		matrices_and_user_input uni;
		uni.mViewMatrix = mQuakeCam.view_matrix();
		uni.mProjMatrix = mQuakeCam.projection_matrix();
		uni.mCamPos     = glm::translate(mQuakeCam.translation());
		uni.mUserInput  = glm::vec4{ mTessellationLevel, mDisplacementStrength, 0.0f, 0.0f };
		if (mCullingDebugMode) {
			uni.mViewToDebugviewMatrix = mDebugCam.view_matrix() * glm::inverse(mQuakeCam.view_matrix());
		}
		else {
			uni.mViewToDebugviewMatrix = glm::mat4(1);
		}
		uni.mAdaptiveTessellationMode = mAdaptiveTessellationMode;
		uni.mFrustumCullingBeforeTess = mFrustumCullingBeforeTess;
		uni.mBackfaceCullingBeforeTess = mBackfaceCullingBeforeTess;
		uni.mDisplacementAntiAliasing = mDisplacementAntiAliasing;

		// Since this buffer has its backing memory in a "host visible" memory region, we just need to write the new data to it.
		// No need to submit the (empty, in this case!) action_type_command that is returned by buffer_t::fill() to a queue.
		// If its backing memory was in a "device" memory region, we would have to, though (see lights buffer below for the difference!).
		currentUniformsBuffer->fill(&uni, 0);

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
			{ currentLightsBuffer->fill(&lightsData, 0) },
			*mQueue, 
			stage::copy
		);
		// Upon completion of this ^ memory transfer into device memory, a semaphore is signaled.
		// We can use this semaphore so that other work must wait on it.

		// Alloc a new command buffer for the current frame, which we are going to record commands into, and then submit to the queue:
		auto cmdBfr = mCommandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		// not all UI selections are mutually compatible...
		bool doWireframe = mWireframeMode;
		bool doHoleCheck = avk::current_composition()->element_by_type<hole_checker>()->is_enabled() && (!doWireframe); // not hole check in wireframe mode
		bool doSkybox    = (!doWireframe) && (!doHoleCheck); // no skybox in wireframe or hole check mode

		// Use either the solid or the wireframe pipeline for drawing the scene
		auto& scenePipeline = doWireframe ? mPipelineWireframe : mPipeline;

		context().record({ // Record a bunch of commands (which can be a mix of state-type commands and action-type commands):

			// Note: For many commands, the framework's provides commands in the command:: namespace,
			//       which allow more convenient usage/recording of functionality into the command buffer.
			//       command::render_pass would be an example and is used here:
			command::render_pass(
				scenePipeline->renderpass_reference(),                   // <-- Use the renderpass of mPipeline (or mPipelineWireframe),
				context().main_window()->current_backbuffer_reference(), // <-- render into the window's backbuffer,
				command::gather(                                         // <-- record nested commands
					command::conditional([doHoleCheck]() { return doHoleCheck; },
						[]() {
							// Note: The Vulkan SDK's command buffer class (from Vulkan-Hpp in this case) provides 
							//       ALL the commands there are. It is accessible through command::custom_commands.
							//       This way, anything can be recorded into the command buffer:
							return command::custom_commands([](avk::command_buffer_t& cb) {
								const vk::CommandBuffer& vkHppCommandBuffer = cb.handle();
								hole_checker::clear_to_red(cb);
							});
						}
					),

					// Bind the pipeline for subsequent draw calls:
					command::bind_pipeline(scenePipeline.as_reference()),
					// Bind all resources we need in shaders:
					command::bind_descriptors(scenePipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
						descriptor_binding(0, 0, mMaterials),
						descriptor_binding(0, 1, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
						descriptor_binding(1, 0, currentUniformsBuffer),
						descriptor_binding(1, 1, currentLightsBuffer)
					})),

					command::many_for_each(mDrawCalls, [&scenePipeline](const draw_call& drawCall) {
						return command::gather(
							command::push_constants(scenePipeline->layout(), push_constants{ drawCall.mModelMatrix, drawCall.mMaterialIndex, drawCall.mEnforceTessellation }),
							command::draw_indexed(
								drawCall.mIndexBuffer.as_reference(),     // Index buffer
								drawCall.mPositionsBuffer.as_reference(), // Vertex buffer at index #0
								drawCall.mTexCoordsBuffer.as_reference(), // Vertex buffer at index #1
								drawCall.mNormalsBuffer.as_reference(),   // Vertex buffer at index #2
								drawCall.mTangentsBuffer.as_reference(),  // Vertex buffer at index #3
								drawCall.mBitangentsBuffer.as_reference() // Vertex buffer at index #4
							)
						);
					})
				)), // <-- end of command::render_pass

				command::conditional([doSkybox] { return doSkybox; },
					[this, &currentUniformsBuffer] { return command::render_pass(mSkyboxPipeline->renderpass_reference(), context().main_window()->current_backbuffer_reference(), {
							command::bind_pipeline(mSkyboxPipeline.as_reference()),
							command::bind_descriptors(mSkyboxPipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
								descriptor_binding(0, 0, currentUniformsBuffer)
							})),
							command::draw_indexed(mSkyboxSphere.mIndexBuffer.as_reference(), mSkyboxSphere.mPositionsBuffer.as_reference())
						});
					}
				)
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

	// A bunch of cameras:
	avk::orbit_camera mOrbitCam;
	avk::quake_camera mQuakeCam;
	debug_camera mDebugCam;

	// A bunch of graphics pipelines:
	avk::graphics_pipeline mPipeline, mPipelineWireframe;

	std::vector<avk::buffer> mUniformsBuffer;
	std::vector<avk::buffer> mLightsBuffer;
	
	float mLastHoleFoundTime = 0.0f;
	bool  mLastHoleButtonVisible = false;

	// ------------------ UI Parameters -------------------
	/** Factor that determines to which amount normals shall be distorted through normal mapping: */
	float mDisplacementStrength = 0.5f;

	float mTessellationLevel = 8.0f;

	bool mWireframeMode = false;

	int mAdaptiveTessellationMode = 0;
	bool mFrustumCullingBeforeTess = false;
	bool mBackfaceCullingBeforeTess = false;
	bool mDisplacementAntiAliasing = false;

	bool mCullingDebugMode = false;

	int mLimitNumPointlights = 98 + EXTRA_POINTLIGHTS;

	// --------------------- Skybox -----------------------
	simple_geometry mSkyboxSphere;
	avk::graphics_pipeline mSkyboxPipeline;
	avk::command_buffer mSkyboxCommandBuffer;

	// ----------------------- ^^^  MEMBER VARIABLES  ^^^ -----------------------
};

//  Main:
//
// +---------------------------------------+
// |                                       |
// |        ARTR 2024 Assignment 2         |
// |                                       |
// +---------------------------------------+
//
//  The second encounter...
// 
int main() 
{
	using namespace avk;

	int result = EXIT_FAILURE;

	try {
		// Create a window, set some configuration parameters (also relevant for its swap chain), and open it:
		auto mainWnd = context().create_window("ARTR 2024 Assignment 2");
		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->set_additional_back_buffer_attachments({
			attachment::declare(vk::Format::eD32Sfloat, on_load::clear, usage::depth_stencil, on_store::dont_care)
		});
		mainWnd->enable_resizing(true);
		mainWnd->request_srgb_framebuffer(true);
		mainWnd->set_presentaton_mode(presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(3u);
		mainWnd->open();

		// Create one single queue which we will submit all command buffers to:
		// (We pass the mainWnd because also presentation shall be submitted to this queue)
		auto& singleQueue = context().create_queue({}, queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->set_queue_family_ownership(singleQueue.family_index());
		mainWnd->set_present_queue(singleQueue);

		// Create an instance of our main class which contains the relevant host code for Assignment 1:
		auto app = assignment2(singleQueue);

		// Create another element for drawing the GUI via the library Dear ImGui:
		auto ui = imgui_manager(singleQueue);

		// Two more utility elements:
		auto holeChecker = hole_checker(singleQueue);
		auto lightsEditor = helpers::create_lightsource_editor(singleQueue, false);
		auto camPresets = helpers::create_camera_presets(singleQueue, false);

		// Pass everything to avk::start and off we go:
		auto composition = configure_and_compose(
			application_name("ARTR 2024 Assignment 2"),
			[](vk::PhysicalDeviceFeatures& features) {
				features.fillModeNonSolid = VK_TRUE; // this device feature is required for wireframe rendering
			},
			mainWnd,
			// Pass the so-called "invokees" which will get their callback methods (such as update() or render()) invoked:
			app, ui, holeChecker, lightsEditor, camPresets			
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

