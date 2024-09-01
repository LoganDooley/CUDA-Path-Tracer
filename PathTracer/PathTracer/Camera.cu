#include "Camera.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/ext.hpp>
#include <iostream>
#include <GLFW/glfw3.h>

Camera::Camera(float near, int screenWidth, int screenHeight) :
	m_moveSpeed(1.f),
	m_rotationSpeed(0.001f),
	m_near(near),
	m_pos(glm::vec3(0, 0, 5)),
	m_up(glm::vec3(0, 1, 0)),
	m_look(glm::vec3(0, 0, -1)),
	m_right(glm::cross(m_look, m_up)),
	m_fov(1),
	m_aspect(float(screenWidth) / screenHeight),
	m_invView(glm::mat4(1))
{
	m_keyMap = { 
		{GLFW_KEY_W, false},
		{GLFW_KEY_A, false},
		{GLFW_KEY_S, false},
		{GLFW_KEY_D, false},
		{GLFW_KEY_SPACE, false},
		{GLFW_KEY_LEFT_SHIFT, false},
	};

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

void Camera::Update(double dt)
{
	glm::vec3 translation = glm::vec3(0);
	if (m_keyMap[GLFW_KEY_W]) {
		translation += m_look;
	}
	if (m_keyMap[GLFW_KEY_S]) {
		translation -= m_look;
	}
	if (m_keyMap[GLFW_KEY_D]) {
		translation += m_right;
	}
	if (m_keyMap[GLFW_KEY_A]) {
		translation -= m_right;
	}
	if (m_keyMap[GLFW_KEY_SPACE]) {
		translation += m_up;
	}
	if (m_keyMap[GLFW_KEY_LEFT_SHIFT]) {
		translation -= m_up;
	}

	if (translation != glm::vec3(0, 0, 0)) {
		m_pos += glm::normalize(translation) * m_moveSpeed * float(dt);
		UpdateInverseView();
	}
}

void Camera::SetKeyPressed(int key, bool pressed) {
	if (m_keyMap.find(key) != m_keyMap.end()) {
		m_keyMap[key] = pressed;
	}
}

void Camera::SetRightMouseButtonPressed(bool pressed) {
	m_rightMouseButtonPressed = pressed;
}

void Camera::SetCursorPos(glm::vec2 position)
{
	if (m_rightMouseButtonPressed) {
		glm::vec2 delta = position - m_cursorPos;
		float rightVectorRotation = delta.y * m_rotationSpeed;
		float upVectorRotation = delta.x * m_rotationSpeed;

		m_look = glm::vec3(glm::rotate(glm::mat4(1), upVectorRotation, m_up) * glm::vec4(m_look, 0));
		m_look = glm::vec3(glm::rotate(glm::mat4(1), rightVectorRotation, m_right) * glm::vec4(m_look, 0));
		m_look = glm::normalize(m_look);
		m_right = glm::cross(m_look, m_up);
		UpdateInverseView();
	}
	m_cursorPos = position;
}

void Camera::UpdateInverseView()
{
	m_invView = glm::inverse(glm::lookAt(m_pos, m_pos + m_look, m_up));
}
