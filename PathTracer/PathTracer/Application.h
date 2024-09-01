#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include "PathTracer.cuh"

class Application
{
public:
	Application();
	~Application();
	void Run();
	void SetKeyPressed(int key, bool pressed);
	void SetRightMouseButtonPressed(bool pressed);
	void SetCursorPos(glm::vec2 position);

private:
	void Init();
	void InitOpenGLObjects();
	void Update(double dt);
	void Render();
	void End();
	void Resize(int width, int height);

	static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);
	static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);

	GLFWwindow* m_window;
	std::unique_ptr<PathTracer> m_pathTracer;

	int m_screenWidth;
	int m_screenHeight;

	GLuint m_fullscreenVAO = 0;
	GLuint m_fullscreenVBO = 0;
	GLuint m_displayTexture = 0;
	GLuint m_textureShader = 0;
	GLint m_textureLocation = 0;
};

