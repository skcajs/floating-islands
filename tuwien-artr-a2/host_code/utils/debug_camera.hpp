#pragma once

#include <auto_vk_toolkit.hpp>

#include "timer_interface.hpp"

class debug_camera : public avk::camera, public avk::invokee
{
public:
	debug_camera(std::string aName = "debug_camera", bool aIsEnabled = true)
		: invokee(std::move(aName), aIsEnabled)
		, mRotationSpeed(0.001f)
	{}

	void update() override {
		// display info about myself
		if (avk::input().key_pressed(avk::key_code::o)
			&& (avk::input().key_down(avk::key_code::left_control) && avk::input().key_down(avk::key_code::right_control))) {
			LOG_INFO(std::format("debug_camera's position: {}", avk::to_string(translation())));
			LOG_INFO(std::format("debug_camera's view-dir: {}", avk::to_string(front(*this))));
			LOG_INFO(std::format("debug_camera's up-vec:   {}", avk::to_string(up(*this))));
			LOG_INFO(std::format("debug_camera's position and orientation:\n{}", avk::to_string(mMatrix)));
			LOG_INFO(std::format("debug_camera's view-mat:\n{}", avk::to_string(view_matrix())));
		}

		auto deltaTime = avk::time().delta_time();

		glm::ivec2 r = { 0,0 };
		if (avk::input().key_down(avk::key_code::i) || avk::input().key_down(avk::key_code::numpad_8)) r.y++;
		if (avk::input().key_down(avk::key_code::k) || avk::input().key_down(avk::key_code::numpad_2)) r.y--;
		if (avk::input().key_down(avk::key_code::j) || avk::input().key_down(avk::key_code::numpad_4)) r.x++;
		if (avk::input().key_down(avk::key_code::l) || avk::input().key_down(avk::key_code::numpad_6)) r.x--;

		if (r.x != 0 || r.y != 0) {
			// accumulate values and create rotation-matrix
			glm::quat rotHoriz = glm::angleAxis(mRotationSpeed * static_cast<float>(r.x), glm::vec3(0.f, 1.f, 0.f));
			glm::quat rotVert = glm::angleAxis(mRotationSpeed * static_cast<float>(r.y), glm::vec3(1.f, 0.f, 0.f));
			set_rotation(rotHoriz * rotation() * rotVert);
		}
	}

private:
	float mRotationSpeed;
};

