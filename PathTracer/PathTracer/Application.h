#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <memory>
#include "PathTracer.cuh"

class Application
{
public:
	Application();
	~Application();
	void Run();

private:
	void Init();
	void InitOpenGLObjects();
	void Update();
	void Render();
	void End();
	void Resize(int width, int height);

	static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);

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

