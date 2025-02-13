                                                                                   |
               Algorithms for Real-Time Rendering (ARTR), 2024S                    |
                                                                                   |
------------------------------------------------------------------------------------
               Assignment 3: Deferred Shading, MSAA, Physically-Based Shading      |
               Deadline: 22.05.2024                                                |
               Submission platform: TUWEL                                          |
====================================================================================

Your Forename: Jackson
Your Surname:  Sam
Your Matriculation Number (8-digit): 12341303


====================================================================================
|                                                                                  |
|                         YOUR DOCUMENTATION FOR ASSIGNMENT 3                      |
|                                                                                  |
====================================================================================
                  First of all, a hint and a very important question:

          HINT:                 Be concise, keep it short!
          IMPORTANT QUESTION:   Do you understand?
          Your answer:          [X] Yes, I understand!     [ ] Nope, Tilt.


                                   Alright, let's go!


==================================== Section Task 1 ================================
                             (Implement Deferred Shading) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Resources Used: Lecture video, learn-opengl tutorial

I dediced there should be 7 additional attachments: 
position, normal, ambient, emmissive, diffuse, specular, shininess as all of these could be computed in the gbuffer pass.
I declared the formats, images, image_views and specified how I wanted them to be used in the render passes. Allthough I'm not too sure if my method is 100% "the right way", the end image looks fine, and I got a 20 frame increase from the forward rendering technique.

====================================================================================

==================================== Section Task 2 ================================
                              (Optimized G-Buffer Layout) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Removed the position from the Gbuffer, obtain the normals using the depth buffer instead.
Pretty straightfoward.

====================================================================================

==================================== Section Task 3 ================================
                         (Enable MSAA and Automatic Resolve) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

That was awful. 

Enabled MSAA resolve attachments, i.e, depth and colour resolves. Made all attachments multisample in the frag shader -> subpassInputMS. Added sample count using a tuple with the formats. Made the additional views/ image views to put in the framebuffer. Took me a VERY long to determine the correct usage flags, in the right order. I ended up resolving in the last subpass, so that the skybox could also get MSAA. Loads of errors along the way. 1/5 - would not recommend. 

====================================================================================

==================================== Section Task 4 ================================
                        (Compute Shader-Based Manual MSAA Resolve) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

That was awful. 
That was awful. 
That was awful. 
That was awful. 
That was awful. 

...

Copied all the resources I needed, so made 2 new renderpasses (one for the buffer, and one for the skybox), and then I also had to define new everything else, such as framebuffers, and then created a new pipeline resources too (mainly for the GBufferPass and the Skybox pipeline). Then I created a new set of commands in render to render the new single subpass pipeline with the compute shader in the middle, followed by another renderpass pipeline for the skybox. I think in order to properly resolve the MSAA, I would need to do the skybox before the lighting pass. 
Finally, afer struggling for what was about 10 hours, I sorted out the transition layouts (and what to do with stored image from the compute shader).
Finally, I updated the compute shader for the render pass, which consisted of copying over the relevant functions from the framgent shader, and manually resolving the samples (by looping over the texture2DMS => texel fetch).

You can switch between compute and graphics pipeline using the GUI, tick "compute shader" option to switch. 

-100 / 5. Are you friend or foe? 
     
====================================================================================

==================================== Section Task 5 ================================
                            (Tile-Based Deferred Shading) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

That was awful.


    ⢀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⣠⣤⣶⣶
    ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⢰⣿⣿⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣀⣀⣾⣿⣿⣿⣿
    ⣿⣿⣿⣿⣿⡏⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿
    ⣿⣿⣿⣿⣿⣿⠀⠀⠀⠈⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠉⠁⠀⣿
    ⣿⣿⣿⣿⣿⣿⣧⡀⠀⠀⠀⠀⠙⠿⠿⠿⠻⠿⠿⠟⠿⠛⠉⠀⠀⠀⠀⠀⣸⣿
    ⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣴⣿⣿⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⢰⣹⡆⠀⠀⠀⠀⠀⠀⣭⣷⠀⠀⠀⠸⣿⣿⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠈⠉⠀⠀⠤⠄⠀⠀⠀⠉⠁⠀⠀⠀⠀⢿⣿⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⢾⣿⣷⠀⠀⠀⠀⡠⠤⢄⠀⠀⠀⠠⣿⣿⣷⠀⢸⣿⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⡀⠉⠀⠀⠀⠀⠀⢄⠀⢀⠀⠀⠀⠀⠉⠉⠁⠀⠀⣿⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿
    ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿


The pain doesn't go away. 

So I followed the slides code to  obtain the min and max depth and the function to test the intersection and culling of the point lights. I set up the fustrum planes, obtained max (top-left) and min (bottom-right) tile coordinates , and converted to VS. to construct the planes used the functions provided, though got a bit of help determining what the direction vectors should be i,e -> vec3 X = vec3(1, 0, 0), Y = vec3(0, 1, 0), Z = vec3(0, 0, 1);, as nothing seemed to work initially.

I have left in the test function which shows the tiles interacting with the point lights.
Just uncomment the code if interested and hope it doesn't crash.

3 turnips/11 - My sanity is compromised.


====================================================================================

==================================== Section Task 6 ================================
                               (Physically-Based Shading) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

That was quite nice actually.

Basically just copied the functions in the slides, and copied the ping-pong implementation, just referencing calc_cook_torrance_contribution instead. Implemented the reflectence, normal distribution and the cook torrence contribution functions and passed in the metallic and roughness! :) 

I posted some nice pictures on discord. 

9 metallic sponza scenes /10.

====================================================================================

================================= Section Bonus Task 1 =============================
                           (More Optimized G-Buffer Layout) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Implemented the spherical coordinates for the normal, using the slides as a starting point. I basically used atan() in GLSL, as a quick search online said this was the one to use. I also compared against the original, and the normals look identical. :) 

For formats, I am pushing metallic and roughness through the same attachment as shininess. No additional attachments needed. 
Most of the attachments are at 8bit precision, apart from the normals which are floating point precision. Generally been using unorm where I can rather than snorm. 

====================================================================================

================================= Section Bonus Task 2 =============================
                                 (Image-Based Lighting) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [X] Not even tried

Description: *What have you done??*

Bonus Task 2 can fuck right off. 

====================================================================================

================================= Section Bonus Task 3 =============================
                                (Optimized MSAA Resolve) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [X] Not even tried

Description: *What have you done??*

And BT3 can do one and all. 

====================================================================================

