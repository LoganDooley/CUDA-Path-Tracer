#pragma once

#include "Camera.cuh"
#include <memory>

class PathTracer
{
public:
	PathTracer(int screenWidth, int screenHeight);
	~PathTracer();

	void Render(int spp);
	void Resize(int screenWidth, int screenHeight);
	float* GetRenderedImage();

private:
	std::unique_ptr<Camera> m_camera;

	int m_screenWidth;
	int m_screenHeight;
	float* d_render = nullptr;
	float* h_render = nullptr;
};