#pragma once

#include <glm/glm.hpp>
#include "cuda_runtime.h"

class Camera
{
public:
	Camera(float near, int screenWidth, int screenHeight);
	~Camera();

	void Resize(int screenWidth, int screenHeight);
	glm::mat4 GetInverseView();

	void DebugPixelCenters();

	float* DebugGetPixelCenters();

private:
	void UpdateInverseView();

	int m_screenWidth;
	int m_screenHeight;

	float m_near;
	glm::vec3 m_pos;
	glm::vec3 m_up;
	glm::vec3 m_look;
	float m_fov;
	float m_aspect;

	glm::mat4 m_invView;

	float* d_pixelCenters = nullptr;
	float* h_pixelCenters = nullptr;
};

