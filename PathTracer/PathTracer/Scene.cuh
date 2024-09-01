#pragma once

#include "Shape.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Object {
	__host__ static Object CreateObject(Shape::Type type, glm::mat4 model) {
		Object result = {
			type,
			CudaMath::Mat4f::FromGLM(model),
			CudaMath::Mat4f::FromGLM(glm::inverse(model))
		};
		return result;
	}

	__device__ inline bool IntersectObject(Ray r) {
		r.Transform(m_invModel);
		return Shape::Intersect(r, m_type);
	}

	Shape::Type m_type;
	CudaMath::Mat4f m_model; // Object space -> World space
	CudaMath::Mat4f m_invModel; // World space -> Object space
};

struct Scene {
	__host__ void AddObject(Object object) {
		m_objects.push_back(object);
	}

	__device__ bool IntersectScene(Ray r) {
		for (int i = 0; i < m_objects.size(); i++) {
			Object object = m_objects[i];
			if (object.IntersectObject(r)) {
				return true;
			}
		}
		return false;
	}

	thrust::device_vector<Object> m_objects;
};