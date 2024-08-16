#include "Camera.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/ext.hpp>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

Camera::Camera(float near, int screenWidth, int screenHeight) :
	m_near(near),
	m_pos(glm::vec3(0, 0, 0)),
	m_up(glm::vec3(0, 0, 1)),
	m_look(glm::vec3(0, 1, 0)),
	m_fov(1),
	m_aspect(float(screenWidth) / screenHeight),
	m_invView(glm::mat4(1))
{
	// Maybe redundant while PathTracer calls resize for the camera in its constructor
	Resize(screenWidth, screenHeight);
	UpdateInverseView();
}

Camera::~Camera()
{
	cudaFree(d_pixelCenters);
	cudaFree(h_pixelCenters);
}

__global__ void resizeCamera(float* pixelCenters, const float near, const float halfWidth, const float halfHeight, const int screenWidth, const int screenHeight)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int pixelIndex = j * screenWidth + i;
	
	float x = halfWidth * (float(2 * i) / float(screenWidth) - 1);
	float y = halfHeight * (float(2 * j) / float(screenHeight) - 1);


	pixelCenters[3 * pixelIndex] = x;
	pixelCenters[3 * pixelIndex + 1] = y;
	pixelCenters[3 * pixelIndex + 2] = -near;
}

void Camera::Resize(int screenWidth, int screenHeight) {
	m_aspect = float(screenWidth) / screenHeight;
	float halfWidth = m_near * glm::tan(m_fov / 2);
	float halfHeight = halfWidth / m_aspect;
	
	// Will no-op if m_pixelCenters is nullptr
	cudaFree(d_pixelCenters);
	cudaMalloc(&d_pixelCenters, 3 * screenWidth * screenHeight * sizeof(float));
	cudaFreeHost(h_pixelCenters);
	cudaMallocHost(&h_pixelCenters, 3 * screenWidth * screenHeight * sizeof(float));

	resizeCamera<<<dim3(screenWidth, screenHeight), 1>>> (d_pixelCenters, m_near, halfWidth, halfHeight, screenWidth, screenHeight);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(h_pixelCenters, d_pixelCenters, 3 * screenWidth * screenHeight * sizeof(float), cudaMemcpyDeviceToHost);
}

glm::mat4 Camera::GetInverseView()
{
	return m_invView;
}

float* Camera::GetDevicePixelCenters()
{
	return d_pixelCenters;
}

float* Camera::GetHostPixelCenters()
{
	return h_pixelCenters;
}

void Camera::UpdateInverseView()
{
	m_invView = glm::inverse(glm::lookAt(m_pos + m_look, m_pos, m_up));
}
