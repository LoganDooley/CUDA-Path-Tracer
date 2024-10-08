#pragma once

#include "CudaMath.cuh"

struct Ray {
	__device__ void Transform(const CudaMath::Mat4f& matrix) {
		m_origin = CudaMath::Transform(m_origin, matrix);
		m_origin.m_v4.w = 1;
		m_direction = CudaMath::Transform(m_direction, matrix);
		m_direction.Normalize3();
		m_direction.m_v4.w = 0;
	}

	CudaMath::Vec4f m_origin;
	CudaMath::Vec4f m_direction;
};