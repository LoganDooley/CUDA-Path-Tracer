#include "PathTracer.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

PathTracer::PathTracer(int screenWidth, int screenHeight):
	m_camera(std::make_unique<Camera>(0.1, screenWidth, screenHeight)),
	m_screenWidth(screenWidth),
	m_screenHeight(screenHeight)
{
	Resize(screenWidth, screenHeight);
}

PathTracer::~PathTracer()
{
}

__global__ void PathTraceScene(float* render, const float* pixelCenters, const int screenWidth)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	int pixelIndex = y * screenWidth + x;
	int redIndex = 3 * pixelIndex;
	int greenIndex = redIndex + 1;
	int blueIndex = greenIndex + 1;
	
	render[redIndex] = pixelCenters[redIndex];
	render[greenIndex] = pixelCenters[greenIndex];
	render[blueIndex] = pixelCenters[blueIndex];
}

void PathTracer::Render(int spp)
{
	// TODO: Make spp work properly and don't use a whole block for one pixel
	PathTraceScene<<<dim3(m_screenWidth, m_screenHeight), 1 >>> (d_render, m_camera->GetDevicePixelCenters(), m_screenWidth);
}

void PathTracer::Resize(int screenWidth, int screenHeight)
{
	m_screenWidth = screenWidth;
	m_screenHeight = screenHeight;

	m_camera->Resize(m_screenWidth, m_screenHeight);

	// Will no-op if m_pixelCenters is nullptr
	cudaFree(d_render);
	cudaMalloc(&d_render, 3 * m_screenWidth * m_screenHeight * sizeof(float));
	cudaFreeHost(h_render);
	cudaMallocHost(&h_render, 3 * m_screenWidth * m_screenHeight * sizeof(float));
}

float* PathTracer::GetRenderedImage()
{
	cudaMemcpy(h_render, d_render, 3 * m_screenWidth * m_screenHeight * sizeof(float), cudaMemcpyDeviceToHost);
	// Ensure h_render has correct data before returning
	cudaDeviceSynchronize();
	return h_render;
}
