#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <memory>
#include "Camera.h"

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

	GLFWwindow* m_window;
	std::unique_ptr<Camera> m_camera;

	GLuint m_fullscreenVAO;
	GLuint m_fullscreenVBO;
	GLuint m_displayTexture;
	GLuint m_textureShader;
	GLint m_textureLocation;
};

