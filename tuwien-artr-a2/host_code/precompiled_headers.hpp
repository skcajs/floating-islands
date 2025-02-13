// cg_stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//
#pragma once

#include "targetver.hpp"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers

#define PURIFIER_CPU_ONLY

#include <auto_vk_toolkit.hpp>
#include <imgui.h>
#include <random>
#include "utils/lights_editor.hpp"
#include "utils/camera_presets.hpp"
#include "utils/helper_functions.hpp"
#include "utils/hole_checker.hpp"
#include "utils/aen_triangles.hpp"
#include "orbit_camera.hpp"
#include "configure_and_compose.hpp"

