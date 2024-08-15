#include "Application.h"
#include "ShaderLoader.h"
#include <vector>
#include <iostream>
#include "Debug.h"

Application::Application():
	m_window(nullptr)
{
}

Application::~Application()
{
}

void Application::Run()
{
	Init();
	while (!glfwWindowShouldClose(m_window)) {
		Update();
		Render();
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
	End();
}

void Application::Init()
{
	std::cout << "Init" << std::endl;

	if (!glfwInit()) {
		std::cout << "Failed to initialize GLFW" << std::endl;
		throw std::exception("Failed to init GLFW");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_window = glfwCreateWindow(640, 480, "Temp", NULL, NULL);
	if (!m_window) {
		glfwTerminate();
		std::cout << "Failed to create window" << std::endl;
		throw std::exception("Failed to create window");
	}
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(0);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		glfwTerminate();
		std::cout << "Failed to load GLAD" << std::endl;
		throw std::exception("Failed to load GLAD");
	}

	m_camera = std::make_unique<Camera>(1, 640, 480);

	InitOpenGLObjects();
}

void Application::InitOpenGLObjects() {
	glClearColor(0, 0, 0, 1);

	m_textureShader = ShaderLoader::CreateShaderProgram("Shaders/Texture.vert", "Shaders/Texture.frag");

	std::vector<GLfloat> fullscreenQuadData = {
		-1, 1, 0, 0, 1,
		-1, -1, 0, 0, 0,
		1, -1, 0, 1, 0,
		-1, 1, 0, 0, 1,
		1, -1, 0, 1, 0,
		1, 1, 0, 1, 1,
	};

	GLuint fullscreenVbo;
	glGenBuffers(1, &fullscreenVbo);
	glBindBuffer(GL_ARRAY_BUFFER, fullscreenVbo);
	glBufferData(GL_ARRAY_BUFFER, fullscreenQuadData.size() * sizeof(GLfloat), fullscreenQuadData.data(), GL_STATIC_DRAW);
	glGenVertexArrays(1, &m_fullscreenVAO);
	glBindVertexArray(m_fullscreenVAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

	glGenTextures(1, &m_displayTexture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_displayTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 640, 480, 0, GL_RGB,
		GL_FLOAT, m_camera->DebugGetPixelCenters());
	glGenerateMipmap(GL_TEXTURE_2D);

	glUseProgram(m_textureShader);
	m_textureLocation = glGetUniformLocation(m_textureShader, "tex");
	glUniform1i(m_textureLocation, 0);
}

void Application::Update()
{
}

void Application::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Application::End()
{
	std::cout << "End" << std::endl;
	glfwDestroyWindow(m_window);
	glfwTerminate();
}
