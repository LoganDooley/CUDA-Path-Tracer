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

	__device__ inline IntersectionData IntersectObject(Ray r) {
		r.Transform(m_invModel);
		IntersectionData intersectionData = Shape::Intersect(r, m_type);
		intersectionData.Transform(m_model, m_invModel);
		return intersectionData;
	}

	Shape::Type m_type;
	CudaMath::Mat4f m_model; // Object space -> World space
	CudaMath::Mat4f m_invModel; // World space -> Object space
};

struct Scene {
	__host__ void AddObject(Object object) {
		m_objects.push_back(object);
	}

	__device__ IntersectionData IntersectScene(Ray r) {
		for (int i = 0; i < m_objects.size(); i++) {
			Object object = m_objects[i];
			IntersectionData intersectionData = object.IntersectObject(r);
			if (intersectionData.intersected) {
				return intersectionData;
			}
		}
		return {};
	}

	thrust::device_vector<Object> m_objects;
};