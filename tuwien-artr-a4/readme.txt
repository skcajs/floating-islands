                                                                                   |
               Algorithms for Real-Time Rendering (ARTR), 2024S                    |
                                                                                   |
------------------------------------------------------------------------------------
               Assignment 4: Screen-Space Effects                                  |
               Deadline: 19.06.2024                                                |
               Submission platform: TUWEL                                          |
====================================================================================

Your Forename: Sam
Your Surname:  Jackson
Your Matriculation Number (8-digit): 12341303


====================================================================================
|                                                                                  |
|                         YOUR DOCUMENTATION FOR ASSIGNMENT 4                      |
|                                                                                  |
====================================================================================
               First of all, an instruction and a very important question:

          INSTRUCTION:          Be concise, keep it short!
          IMPORTANT QUESTION:   Do you understand?
          Your answer:          [ ] Yes, I understand!     [ ] Nope, Tilt.


                                   Alright, let's go!


==================================== Section Task 1 ================================
                                (Fix and Optimize SSAO) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Reosources used: 
https://learnopengl.com/Advanced-Lighting/SSAO
Lecture Notes


I have produced SSAO using a random noise filter, and have implemented hemisphere sampling (weighted towards the normal).
I initially tried to use a texture to store the noise and sample from that, but it wasn't working, so instead I have implemented a UBO which just gets iterated through per pixel. This works quite nicely. 

====================================================================================

==================================== Section Task 2 ================================
                             (Blur the Occlusion Factors) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Tried a few different filters, Bilateral, Median and Bilinear (I used to think this was called Billionaire Filtering, because only rich people could use it)

Settled on Billionair Filtering, as it has the most striking difference. 
Bilateral and Median very very subtle, but did lead to a cleaner image overall.

====================================================================================

==================================== Section Task 3 ================================
                                (Reinhard Tone Mapping) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Implemented Reinhard Tone Mapping, using the equations described in the paper. 

====================================================================================

==================================== Section Task 4 ================================
                           (Gradual Adaption of Tone Mapping) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Implemented the adaptive tone mapping. Mapped the buffer values to 1-exp(-at) where a controls how quickly the bightness increases/ decreases. t is the delta time.
I have a configurable parameter for a, so you can increase/ decrease the speed in the gui.

====================================================================================

==================================== Section Task 5 ================================
                              (Screen-Space Reflections) 

Status: [X] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

Implemented ray marching in the comp shader. Grabbed position and moved it along the reflection vector. Converted position to screen space and checked to see if coordinate was in bounds of the texture. Then checked z position against depth buffer (in VS), and if it was "in the vicinity" (epsilon), sampled the value (in SS). 
Added some GUI stuff to control the number of iterations, step size and epsilon (depth)

====================================================================================

==================================== Section Task 6 ================================
                                (Temporal Anti-Aliasing) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [X] Not even tried

Description: *What have you done??*



====================================================================================

============================= Section Bonus Task 1 RTX OFF =========================
                                   (AO Alternative) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 2 RTX OFF =========================
                               (Tone Mapping Alternative) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 3 RTX OFF =========================
                              (Anti-Aliasing Alternative) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 1 RTX ON ==========================
                                 (Ray Traced Shadows) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 2 RTX ON ==========================
                              (Ray Tracing Pipeline Setup) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 3 RTX ON ==========================
                               (Ray Traced Reflections) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 4 RTX ON ==========================
                                (No Skybox Left Behind) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================

============================= Section Bonus Task 5 RTX ON ==========================
                               (Shadows in Reflections) 

Status: [ ] Done, [ ] Partially Done, [ ] Approached, [ ] Failed, [ ] Not even tried

Description: *What have you done??*

====================================================================================


