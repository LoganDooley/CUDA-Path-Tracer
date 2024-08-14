#include "Camera.h"

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

Camera::Camera(int screenWidth, int screenHeight) :
	m_screenWidth(screenWidth),
	m_screenHeight(screenHeight),
	m_pos(glm::vec3(0, 0, 0)),
	m_up(glm::vec3(0, 0, 1)),
	m_look(glm::vec3(0, 1, 0)),
	m_fov(1),
	m_aspect(screenWidth / screenHeight),
	m_invView(glm::mat4(1))
{
	Resize(screenWidth, screenHeight);
	UpdateInverseView();
}

Camera::~Camera()
{
	cudaFree(d_pixelCenters);
	cudaFree(h_pixelCenters);
}

__global__ void resizeCamera(float* pixelCenters, const int screenWidth)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int pixelIndex = j * screenWidth + i;
	float relative = float(i) / screenWidth;
	pixelCenters[pixelIndex] = relative;
}

void Camera::Resize(int screenWidth, int screenHeight) {
	m_screenWidth = screenWidth;
	m_screenHeight = screenHeight;
	m_aspect = float(screenWidth) / screenHeight;
	
	// Will no-op if m_pixelCenters is nullptr
	cudaFree(d_pixelCenters);
	cudaMalloc(&d_pixelCenters, screenWidth * screenHeight * sizeof(float));
	cudaMemset(d_pixelCenters, 0, screenWidth * screenHeight * sizeof(float));
	cudaFreeHost(h_pixelCenters);
	cudaMallocHost(&h_pixelCenters, screenWidth * screenHeight * sizeof(float));

	resizeCamera<<<dim3(screenWidth, screenHeight), 1>>> (d_pixelCenters, screenWidth);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(h_pixelCenters, d_pixelCenters, screenWidth * screenHeight * sizeof(float), cudaMemcpyDeviceToHost);
}

glm::mat4 Camera::GetInverseView()
{
	return m_invView;
}

void Camera::DebugPixelCenters()
{
	if (h_pixelCenters == nullptr) {
		std::cout << "hi" << std::endl;
	}
	for (int i = 0; i < m_screenWidth; i++) {
		float value = h_pixelCenters[i];
		std::cout << value << std::endl;
	}
}

void Camera::UpdateInverseView()
{
	m_invView = glm::inverse(glm::lookAt(m_pos + m_look, m_pos, m_up));
}
