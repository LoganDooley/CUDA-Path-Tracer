#pragma once

#include "Ray.cuh"

struct IntersectionData {
	inline __device__ void Transform(const CudaMath::Mat4f& model, const CudaMath::Mat4f& invModel) {
		CudaMath::TransformVec4f(m_position, model);
		m_position.m_v4.w = 1;
		CudaMath::TransformVec4f(m_normal, invModel.GetTranspose());
		m_normal.Normalize3();
		m_normal.m_v4.w = 0;
	}

	bool intersected;
	CudaMath::Vec4f m_position;
	CudaMath::Vec4f m_normal;
};

struct Shape {
	enum Type {
		Sphere,	
	};

	inline __device__ static IntersectionData Intersect(Ray r, Type type) {
		IntersectionData intersectionData = {};
		switch (type) {
		case Type::Sphere:
			IntersectSphere(r, intersectionData);
			break;
		default:
			break;
		}
		return intersectionData;
	}

	inline __device__ static void IntersectSphere(Ray r, IntersectionData& intersectionData) {
		// Unit sphere has radius 0.5 and centered at the origin

		float b = Dot3(r.m_origin, r.m_direction);
		float c = Dot3(r.m_origin, r.m_origin) - 0.25f;

		// Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0) 
		if (c > 0.0f && b > 0.0f) {
			return;
		}

		float discr = b * b - c;

		// A negative discriminant corresponds to ray missing sphere 
		if (discr < 0.0f) {
			return;
		}

		// Ray now found to intersect sphere, compute smallest t value of intersection
		float t = -b - sqrtf(discr);

		// If t is negative, ray started inside sphere so clamp t to zero TODO: FIX THIS SO WE GET INTERNAL INTERSECTION
		if (t < 0.0f) t = 0.0f;

		intersectionData.intersected = true;
		intersectionData.m_position = r.m_origin + r.m_direction * t;
		intersectionData.m_normal = intersectionData.m_position;
		intersectionData.m_position.m_v4.w = 1;
		intersectionData.m_normal.m_v4.w = 0;
		return;
	}
};