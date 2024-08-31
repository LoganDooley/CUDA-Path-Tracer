#pragma once

#include "CudaMath.cuh"

struct Ray {
	__device__ void Transform(CudaMath::Mat4f matrix) {
		m_origin = (matrix * CudaMath::Vec4f::FromVec3f(m_origin, 1)).ToVec3f();
		m_direction = (matrix * CudaMath::Vec4f::FromVec3f(m_direction, 0)).ToVec3f();
	}

	CudaMath::Vec3f m_origin;
	CudaMath::Vec3f m_direction;
};