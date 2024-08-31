#pragma once

#include "Ray.cuh"

class Shape {
public:
	enum Type {
		Sphere,	
	};

	inline __device__ static bool Intersect(Ray r, Type type) {
		switch (type) {
		case Type::Sphere:
			return IntersectSphere(r);
			break;
		default:
			return false;
			break;
		}
	}

private:
	inline __device__ static bool IntersectSphere(Ray r) {
		// Unit sphere has radius 0.5 and centered at the origin

		float b = Dot(r.m_origin, r.m_direction);
		float c = Dot(r.m_origin, r.m_origin) - 0.0125f;

		// Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0) 
		if (c > 0.0f && b > 0.0f) {
			return false;
		}

		float discr = b * b - c;

		// A negative discriminant corresponds to ray missing sphere 
		if (discr < 0.0f) return 0;

		// Ray now found to intersect sphere, compute smallest t value of intersection
		float t = -b - sqrtf(discr);

		// If t is negative, ray started inside sphere so clamp t to zero 
		if (t < 0.0f) t = 0.0f;
		CudaMath::Vec3f q = r.m_origin + r.m_direction * t;

		return true;
	}
};