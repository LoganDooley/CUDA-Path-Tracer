#pragma once

#include <glm/glm.hpp>
#include <unordered_map>

class Camera
{
public:
	Camera(float near, int screenWidth, int screenHeight);
	~Camera();

	void Resize(int screenWidth, int screenHeight);
	glm::mat4 GetInverseView();

	float* GetDevicePixelCorners();
	void Update(double dt);
	void SetKeyPressed(int key, bool pressed);
	void SetRightMouseButtonPressed(bool pressed);
	void SetCursorPos(glm::vec2 position);

private:
	void UpdateInverseView();

	float m_moveSpeed;
	float m_rotationSpeed;
	float m_near;
	glm::vec3 m_pos;
	glm::vec3 m_up;
	glm::vec3 m_look;
	glm::vec3 m_right;
	float m_fov;
	float m_aspect;
	glm::mat4 m_invView;
	float* d_pixelCorners = nullptr;
	std::unordered_map<int, bool> m_keyMap;
	bool m_rightMouseButtonPressed;
	glm::vec2 m_cursorPos = glm::vec2(0, 0);
};