                                                                                   |
               Algorithms for Real-Time Rendering (ARTR), 2024S                    |
                                                                                   |
------------------------------------------------------------------------------------
               Assignment 2: Tessellation and Displacement Mapping                 |
               Deadline: 24.04.2024                                                |
               Submission platform: TUWEL                                          |
====================================================================================

Your Forename: Sam
Your Surname:  Jackson
Your Matriculation Number (8-digit): 12341303


====================================================================================
|                                                                                  |
|                         YOUR DOCUMENTATION FOR ASSIGNMENT 2                      |
|                                                                                  |
====================================================================================
                  First of all, a hint and a very important question:

          HINT:                 Be concise, keep it short!
          IMPORTANT QUESTION:   Do you understand?
          Your answer:          [X] Yes, I understand!     [ ] Nope, Tilt.


                                   Alright, let's go!


==================================== Section Task 1 ================================
                                 (Enable Tessellation) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

1]
Added the shaders to the mPipeline creation process:
	tessellation_control_shader("shaders/tess_pn_controlpoints.tesc"),
	tessellation_evaluation_shader("shaders/tess_pn_interp_and_displacement.tese"),

2]
Updated the following line:
	cfg::primitive_topology::triangles -> cfg::primitive_topology::patches

3]
Added the following line under the topology:
	cfg::tessellation_patch_control_points(3),

====================================================================================

==================================== Section Task 2 ================================
                        (Complete PN-Triangles Implementation) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Resources Used: 
(1) Curved PN Triangles, Vlachos et al.
(2) PN Triangles, VU ARTR TUWIen.

1]
Projected the intermediate points onto the plane defined by the (outer) points and its corresponding normal.
N.B only displaced control points, the outer points (0,0,3 ; 0,3,0 ; 3,0,0) are not displaced.
Calculated the centre point as referenced in (2) as VE/2 + E.

2]
Calculated the point vector normals

====================================================================================

==================================== Section Task 3 ================================
                      (Frustum Culling in Tess.-Control Shader) 

Status: [X] Done, [] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Resources Used: 
(1) Youtube - Frustum Culling // Terrain Rendering episode #9 - OGLDev

1] Implimented first using just the patch vertices in clip space. Bug with large triangles where patch dissappears.
2] Strange bug in code, line of visible patches behind user. Probably not a huge deal, I think it's to do with /w.

====================================================================================

==================================== Section Task 4 ================================
                       (Backface Culling in Tess.-Control Shader) 

Status: [X] Done, [] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Resources Used:
https://en.wikipedia.org/wiki/Back-face_culling

1] Implimented back face culling utilising the v.n method outlined in the wiki page.
2] Used bezier control points to determine false positives (using the same v.n method).

====================================================================================

==================================== Section Task 5 ================================
                             (Adaptive Tessellation Modes) 

Status: [X] Done, [] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Resources used:
https://learnopengl.com/Guest-Articles/2021/Tessellation/Tessellation
https://victorbush.com/2015/01/tessellated-terrain/

1] Added distance based adaptive tessellation mode. This varies with the camera.
	Set max to userdefined input. Min to 4.
	Clamped distances between 0 and 1
	Set to tessellation outputs and inputs.
2] Made implimentation 2 vary with angle (currently only when not in quake mode),
it's cracky though, not sure if there is a fix for this...

====================================================================================

==================================== Section Task 6 ================================
                                 (Displacement Mapping) 

Status: [X] Done, [] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Reousrces used:

1] Implimented Displacement mapping. Subtracted -0.5 from the height to keep terrain level. :) 

====================================================================================

================================= Section Bonus Task 1 =============================
                                   (PN-AEN-Triangles) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [X] Not even tried

Description: *What have you done??*

Time has not been good to me :(

====================================================================================

================================= Section Bonus Task 2 =============================
                              (Displacement Anti-Aliasing) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [X] Not even tried

Description: *What have you done??*

Time has not been good to me :(

====================================================================================

================================= Section Bonus Task 3 =============================
                               (Watertight Tessellation) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

