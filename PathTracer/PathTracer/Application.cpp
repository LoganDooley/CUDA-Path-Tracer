#include "Application.h"
#include <iostream>

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

	m_camera = std::make_unique<Camera>(640, 480);
}

void Application::Update()
{
}

void Application::Render()
{
}

void Application::End()
{
	std::cout << "End" << std::endl;
	m_camera->DebugPixelCenters();
	glfwDestroyWindow(m_window);
	glfwTerminate();
}
