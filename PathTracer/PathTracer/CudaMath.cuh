#pragma once

#include <cuda_runtime.h>
#include <math_functions.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>

namespace CudaMath {
	struct Vec3f {
		inline __device__ void Normalize() {
			float length = Length();
			m_v3.x /= length;
			m_v3.y /= length;
			m_v3.z /= length;
		}

		inline __device__ const float Length() {
			return sqrtf(powf(m_v3.x, 2) + powf(m_v3.y, 2) + powf(m_v3.z, 2));
		}

		inline __device__ const Vec3f operator-(const Vec3f& other) {
			Vec3f result = {};
			result.m_v3.x = m_v3.x - other.m_v3.x;
			result.m_v3.y = m_v3.y - other.m_v3.y;
			result.m_v3.z = m_v3.z - other.m_v3.z;
			return result;
		}

		inline __device__ const Vec3f operator+(const Vec3f& other) {
			Vec3f result = {};
			result.m_v3.x = m_v3.x + other.m_v3.x;
			result.m_v3.y = m_v3.y + other.m_v3.y;
			result.m_v3.z = m_v3.z + other.m_v3.z;
			return result;
		}

		inline __device__ const Vec3f operator*(const float& other) {
			Vec3f result = {};
			result.m_v3.x = m_v3.x * other;
			result.m_v3.y = m_v3.y * other;
			result.m_v3.z = m_v3.z * other;
			return result;
		}

		float3 m_v3;
	};

	inline __device__ float Dot(Vec3f a, Vec3f b) {
		return a.m_v3.x * b.m_v3.x + a.m_v3.y * b.m_v3.y + a.m_v3.z * b.m_v3.z;
	}

	struct Vec4f {
		__host__ static Vec4f FromGLM(glm::vec4 v4) {
			Vec4f result = {};
			result.m_v4.x = v4.x;
			result.m_v4.y = v4.y;
			result.m_v4.z = v4.z;
			result.m_v4.w = v4.w;
			return result;
		}

		inline __device__ static Vec4f FromVec3f(Vec3f v3, float w) {
			Vec4f result = {};
			result.m_v4.x = v3.m_v3.x;
			result.m_v4.y = v3.m_v3.y;
			result.m_v4.z = v3.m_v3.z;
			result.m_v4.w = w;
			return result;
		}

		inline __device__ Vec3f ToVec3f() {
			Vec3f v3 = {};
			v3.m_v3.x = m_v4.x;
			v3.m_v3.y = m_v4.y;
			v3.m_v3.z = m_v4.z;
			return v3;
		}

		inline __device__ void Normalize() {
			float length = Length();
			m_v4.x /= length;
			m_v4.y /= length;
			m_v4.z /= length;
			m_v4.w /= length;
		}

		inline __device__ void Normalize3() {
			float length3 = Length3();
			m_v4.x /= length3;
			m_v4.y /= length3;
			m_v4.z /= length3;
		}

		inline __device__ const float Length() {
			return sqrtf(powf(m_v4.x, 2) + powf(m_v4.y, 2) + powf(m_v4.z, 2) + powf(m_v4.w, 2));
		}

		inline __device__ const float Length3() {
			return sqrtf(powf(m_v4.x, 2) + powf(m_v4.y, 2) + powf(m_v4.z, 2));
		}

		inline __device__ const Vec4f operator-(const Vec4f& other) {
			Vec4f result = {};
			result.m_v4.x = m_v4.x - other.m_v4.x;
			result.m_v4.y = m_v4.y - other.m_v4.y;
			result.m_v4.z = m_v4.z - other.m_v4.z;
			result.m_v4.w = m_v4.w - other.m_v4.w;
			return result;
		}

		inline __device__ const Vec4f operator+(const Vec4f& other) {
			Vec4f result = {};
			result.m_v4.x = m_v4.x + other.m_v4.x;
			result.m_v4.y = m_v4.y + other.m_v4.y;
			result.m_v4.z = m_v4.z + other.m_v4.z;
			result.m_v4.w = m_v4.w + other.m_v4.w;
			return result;
		}

		inline __device__ const Vec4f operator*(const float& other) {
			Vec4f result = {};
			result.m_v4.x = m_v4.x * other;
			result.m_v4.y = m_v4.y * other;
			result.m_v4.z = m_v4.z * other;
			result.m_v4.w = m_v4.w * other;
			return result;
		}

		float4 m_v4;
	};

	inline __device__ float Dot(const Vec4f& a, const Vec4f& b) {
		return a.m_v4.x * b.m_v4.x + a.m_v4.y * b.m_v4.y + a.m_v4.z * b.m_v4.z + a.m_v4.w * b.m_v4.w;
	}

	inline __device__ float Dot3(const Vec4f& a, const Vec4f& b) {
		return a.m_v4.x * b.m_v4.x + a.m_v4.y * b.m_v4.y + a.m_v4.z * b.m_v4.z;
	}

	struct Mat4f {
		__host__ static Mat4f FromGLM(glm::mat4 mat4) {
			Mat4f result = {};
			result.m_r0 = Vec4f::FromGLM(glm::row(mat4, 0));
			result.m_r1 = Vec4f::FromGLM(glm::row(mat4, 1));
			result.m_r2 = Vec4f::FromGLM(glm::row(mat4, 2));
			result.m_r3 = Vec4f::FromGLM(glm::row(mat4, 3));
			return result;
		}

		Vec4f m_r0;
		Vec4f m_r1;
		Vec4f m_r2;
		Vec4f m_r3;
	};

	inline __device__ Vec4f Transform(const Mat4f& mat, const Vec4f& vec) {
		Vec4f result = {};
		result.m_v4.x = Dot(mat.m_r0, vec);
		result.m_v4.y = Dot(mat.m_r1, vec);
		result.m_v4.z = Dot(mat.m_r2, vec);
		result.m_v4.w = Dot(mat.m_r3, vec);
		return result;
	}
}