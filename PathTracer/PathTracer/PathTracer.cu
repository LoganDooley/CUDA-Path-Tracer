#include "PathTracer.cuh"

#include "Scene.cuh"
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

PathTracer::PathTracer(int screenWidth, int screenHeight):
	m_camera(std::make_unique<Camera>(0.1, screenWidth, screenHeight)),
	m_screenWidth(screenWidth),
	m_screenHeight(screenHeight)
{
	m_scene = new Scene();
	((Scene*)m_scene)->AddObject((Object::CreateObject(Shape::Type::Sphere, glm::mat4(1))));

	Resize(screenWidth, screenHeight);
}

PathTracer::~PathTracer()
{
}

__global__ void PathTraceScene(float* render, 
	const float* pixelCorners, 
	const int screenWidth,
	CudaMath::Mat4f invView,
	Scene scene)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	int pixelIndex = y * screenWidth + x;
	int redIndex = 3 * pixelIndex;
	int greenIndex = redIndex + 1;
	int blueIndex = greenIndex + 1;

	float cornerX = pixelCorners[redIndex];
	float cornerY = pixelCorners[greenIndex];
	float cornerZ = pixelCorners[blueIndex];

	// Camera space ray
	Ray r = {
		{make_float4(0, 0, 0, 1)}, 
		{make_float4(cornerX, cornerY, cornerZ, 0)}
	};

	// World space ray
	r.Transform(invView);

	IntersectionData intersectionData = scene.IntersectScene(r);

	bool intersected = intersectionData.intersected;

	render[redIndex] = 0;
	render[greenIndex] = 0;
	render[blueIndex] = 0;

	if (intersected) {
		render[redIndex] = 1;
		render[greenIndex] = 1;
		render[blueIndex] = 1;
		//render[redIndex] = intersectionData.m_normal.m_v4.x;
		//render[greenIndex] = intersectionData.m_normal.m_v4.y;
		//render[blueIndex] = intersectionData.m_normal.m_v4.z;
	}
}

void PathTracer::Update(double dt)
{
	m_camera->Update(dt);
}

void PathTracer::Render(int spp)
{
	Scene* scene = (Scene*)m_scene;

	// TODO: Make spp work properly and don't use a whole block for one pixel
	PathTraceScene<<<dim3(m_screenWidth, m_screenHeight), 1 >>> (d_render, 
		m_camera->GetDevicePixelCorners(),
		m_screenWidth,
		CudaMath::Mat4f::FromGLM(m_camera->GetInverseView()),
		*scene);
	//gpuErrchk(cudaPeekAtLastError());
}

void PathTracer::Resize(int screenWidth, int screenHeight)
{
	m_screenWidth = screenWidth;
	m_screenHeight = screenHeight;

	m_camera->Resize(m_screenWidth, m_screenHeight);

	// Will no-op if m_pixelCenters is nullptr
	cudaFree(d_render);
	cudaMalloc(&d_render, 3 * m_screenWidth * m_screenHeight * sizeof(float));
	cudaFreeHost(h_render);
	cudaMallocHost(&h_render, 3 * m_screenWidth * m_screenHeight * sizeof(float));
}

float* PathTracer::GetRenderedImage()
{
	cudaMemcpy(h_render, d_render, 3 * m_screenWidth * m_screenHeight * sizeof(float), cudaMemcpyDeviceToHost);
	// Ensure h_render has correct data before returning
	cudaDeviceSynchronize();
	return h_render;
}

void PathTracer::SetKeyPressed(int key, bool pressed)
{
	m_camera->SetKeyPressed(key, pressed);
}

void PathTracer::SetRightMouseButtonPressed(bool pressed)
{
	m_camera->SetRightMouseButtonPressed(pressed);
}

void PathTracer::SetCursorPos(glm::vec2 position)
{
	m_camera->SetCursorPos(position);
}
