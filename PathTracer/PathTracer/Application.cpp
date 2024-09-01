#include "Application.h"
#include "ShaderLoader.h"
#include <vector>
#include <iostream>
#include "Debug.h"

Application::Application():
	m_window(nullptr),
	m_screenWidth(640),
	m_screenHeight(480)
{
}

Application::~Application()
{
}

void Application::Run()
{
	Init();
	double previousTime = glfwGetTime();
	double lastRender = previousTime;
	while (!glfwWindowShouldClose(m_window)) {
		double currentTime = glfwGetTime();
		Update(currentTime - previousTime);
		std::string title = "UPS: "+std::to_string(1 / (currentTime - previousTime));
		glfwSetWindowTitle(m_window, title.c_str());
		previousTime = currentTime;
		if (currentTime - lastRender > 1 / 360.f) {
			lastRender = glfwGetTime();
			Render();
			glfwSwapBuffers(m_window);
		}
		glfwPollEvents();
	}
	End();
}

void Application::SetKeyPressed(int key, bool pressed)
{
	m_pathTracer->SetKeyPressed(key, pressed);
}

void Application::SetRightMouseButtonPressed(bool pressed)
{
	m_pathTracer->SetRightMouseButtonPressed(pressed);
}

void Application::SetCursorPos(glm::vec2 position)
{
	m_pathTracer->SetCursorPos(position);
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

	m_window = glfwCreateWindow(m_screenWidth, m_screenHeight, "Temp", NULL, NULL);
	if (!m_window) {
		glfwTerminate();
		std::cout << "Failed to create window" << std::endl;
		throw std::exception("Failed to create window");
	}
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(0);

	glfwSetWindowUserPointer(m_window, this);
	glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
	glfwSetFramebufferSizeCallback(m_window, FramebufferResizeCallback);
	glfwSetKeyCallback(m_window, KeyCallback);
	glfwSetMouseButtonCallback(m_window, MouseButtonCallback);
	glfwSetCursorPosCallback(m_window, CursorPosCallback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		glfwTerminate();
		std::cout << "Failed to load GLAD" << std::endl;
		throw std::exception("Failed to load GLAD");
	}

	m_pathTracer = std::make_unique<PathTracer>(m_screenWidth, m_screenHeight);

	glViewport(0, 0, m_screenWidth, m_screenHeight);

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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_screenWidth, m_screenHeight, 0, GL_RGB,
		GL_FLOAT, NULL);

	glUseProgram(m_textureShader);
	m_textureLocation = glGetUniformLocation(m_textureShader, "tex");
	glUniform1i(m_textureLocation, 0);
}

void Application::Update(double dt)
{
	m_pathTracer->Update(dt);
	m_pathTracer->Render(1);
}

void Application::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_screenWidth, m_screenHeight, GL_RGB, GL_FLOAT, m_pathTracer->GetRenderedImage());
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Application::End()
{
	std::cout << "End" << std::endl;
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Application::Resize(int width, int height) 
{
	m_screenWidth = width;
	m_screenHeight = height;

	glViewport(0, 0, m_screenWidth, m_screenHeight);
	m_pathTracer->Resize(m_screenWidth, m_screenHeight);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_screenWidth, m_screenHeight, 0, GL_RGB,
		GL_FLOAT, NULL);
}

void Application::FramebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	Application* app = (Application*)glfwGetWindowUserPointer(window);;
	app->Resize(width, height);
}

void Application::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	Application* app = (Application*)glfwGetWindowUserPointer(window);
	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_ESCAPE) {
			glfwSetWindowShouldClose(window, true);
		}
		app->SetKeyPressed(key, true);
	}
	else if (action == GLFW_RELEASE) {
		app->SetKeyPressed(key, false);
	}
}

void Application::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		Application* app = (Application*)glfwGetWindowUserPointer(window);

		if (action == GLFW_PRESS) {
			app->SetRightMouseButtonPressed(true);
		}
		else if (action == GLFW_RELEASE) {
			app->SetRightMouseButtonPressed(false);
		}
	}
}

void Application::CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Application* app = (Application*)glfwGetWindowUserPointer(window);
	app->SetCursorPos(glm::vec2(xpos, ypos));
}
