#include "PathTracer.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Scene.cuh"

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
	Resize(screenWidth, screenHeight);
}

PathTracer::~PathTracer()
{
}

__global__ void PathTraceScene(float* render, const float* pixelCorners, const int screenWidth)
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

	Ray r = { {make_float3(cornerX, cornerY, cornerZ)}, {make_float3(0, 0, 0)}};
	CudaMath::Mat4f m = {
		{make_float4(1, 0, 0, 0)},
		{make_float4(0, 1, 0, 0)},
		{make_float4(0, 0, 1, 0)},
		{make_float4(0, 0, 0, 1)}
	};
	r.Transform(m);

	bool intersected = Shape::Intersect(r, Shape::Type::Sphere);
	
	render[redIndex] = r.m_origin.m_v3.x;
	render[greenIndex] = r.m_origin.m_v3.y;
	render[blueIndex] = r.m_origin.m_v3.z;

	if (intersected) {
		render[redIndex] = 1;
		render[greenIndex] = 1;
		render[blueIndex] = 1;
	}
}

void PathTracer::Render(int spp)
{
	// TODO: Make spp work properly and don't use a whole block for one pixel
	PathTraceScene<<<dim3(m_screenWidth, m_screenHeight), 1 >>> (d_render, m_camera->GetDevicePixelCorners(), m_screenWidth);
	gpuErrchk(cudaPeekAtLastError());
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
