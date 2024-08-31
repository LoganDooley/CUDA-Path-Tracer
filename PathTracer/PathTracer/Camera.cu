#include "Camera.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/ext.hpp>
#include <iostream>

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
	cudaFree(d_pixelCorners);
}

__global__ void resizeCamera(float* pixelCorners, const float near, const float halfWidth, const float halfHeight, const int screenWidth, const int screenHeight)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int pixelIndex = j * screenWidth + i;
	
	float x = halfWidth * (float(2 * i) / float(screenWidth) - 1);
	float y = halfHeight * (float(2 * j) / float(screenHeight) - 1);


	pixelCorners[3 * pixelIndex] = x;
	pixelCorners[3 * pixelIndex + 1] = y;
	pixelCorners[3 * pixelIndex + 2] = -near;
}

void Camera::Resize(int screenWidth, int screenHeight) {
	m_aspect = float(screenWidth) / screenHeight;
	float halfWidth = m_near * glm::tan(m_fov / 2);
	float halfHeight = halfWidth / m_aspect;
	
	// Will no-op if m_pixelCenters is nullptr
	cudaFree(d_pixelCorners);
	cudaMalloc(&d_pixelCorners, 3 * screenWidth * screenHeight * sizeof(float));

	resizeCamera<<<dim3(screenWidth, screenHeight), 1>>> (d_pixelCorners, m_near, halfWidth, halfHeight, screenWidth, screenHeight);
}

glm::mat4 Camera::GetInverseView()
{
	return m_invView;
}

float* Camera::GetDevicePixelCorners()
{
	return d_pixelCorners;
}

void Camera::UpdateInverseView()
{
	m_invView = glm::inverse(glm::lookAt(m_pos + m_look, m_pos, m_up));
}
