#pragma once
#include "Utils/vec_math.h"
#include "Utils/Exception.h"
#include "Utils/Timing.h"

#include <curand_kernel.h>
#include <limits.h>
#include <random>

namespace mcisph
{

#define START_TIMING(timerName) \
		Timing::startTiming(timerName);

#define STOP_TIMING_AVG \
		{ \
		static int timing_timerId = -1; \
        cudaDeviceSynchronize(); \
		Timing::stopTiming(false, timing_timerId); \
		}

#define PRINT_TIMING_AVG \
	    Timing::printAverageTimes();

#define PRINT_TIMING_SUMS \
		Timing::printTimeSums();

    enum SPHKernelType { CUBIC_SPLINE, SPIKY, POLY6, WENDLAND, GAUSSIAN };
    enum SpatialEstimation {ITERATION, UNIFORM_SAMPLING};
    enum BoundaryEstimation { DIRECTIOANAL_SAMPLING, UNIFORM_AREA_SAMPLING};

    struct DeviceConfig 
    {
        uint32_t path_length;
        uint32_t num_path_sample;
        uint32_t num_neighbor;
        float area_coeff;
    };

    struct FluidModelConfig 
    {
        float max_cfl_dt;
        float min_cfl_dt;
        float cfl_factor;

        float dGdx_reg; 
        float volume_coeff;
        uint32_t num_volume_sample;

        float particle_radius;
        float viscos_coeff;
        float density0;
        float3 gravity;
    };

    __forceinline__ __device__ bool isfinite_vec(const float3& vec) {
        return isfinite(vec.x) && isfinite(vec.y) && isfinite(vec.z);
    }

    __forceinline__ __device__ bool isClosed(const float3& a, const float3& b) {

		float eps = 1e-5f;

        return fabsf(a.x - b.x) < eps && fabsf(a.y - b.y) < eps && fabsf(a.z - b.z) < eps;
    }

    __forceinline__ __device__ bool isZero(const float3& a) {
        return fabsf(a.x) < 1e-16f && fabsf(a.y) < 1e-16f && fabsf(a.z) < 1e-16f;
    }

    __forceinline__ __device__ uint3 unflatten(uint32_t idx, const uint3& grid_res) noexcept {
        return { idx % grid_res.x, (idx / grid_res.x) % grid_res.y, idx / (grid_res.x * grid_res.y) };
    }

    __forceinline__ __device__ uint32_t flatten(const uint3& idx, const uint3& grid_res) noexcept {
        return idx.z * grid_res.y * grid_res.x + idx.y * grid_res.x + idx.x;
    }

    __forceinline__ __device__ float length2(const float3& v) { return dot(v, v); }

    /*the gradient of Poisson integral kernel*/
    __forceinline__ __device__ float3 Poisson_dGdx(const float3& x, const float3& y, float eps)
    {
        float3 r_vec = y - x;
        float r2 = length2(r_vec) + eps;
        float coeff = 1.0f /(4.0f * M_PIf * powf(r2, 1.5f));

        return coeff * r_vec; 
    }

    // Compensated sum algorithm by Kahan
    struct KahanSum 
    {
        float3 sum, c;

        __forceinline__  __device__ KahanSum()
        {
            sum = {};
            c = {};
        };

        __forceinline__  __device__ KahanSum(const KahanSum& other)
        {
            this->sum = other.sum;
            this->c = other.c;
        }

        __forceinline__  __device__ KahanSum& operator+=(const float3& value)
        {
            float3 y = value - c;
            float3 t = sum + y;
            c = (t - sum) - y;
            sum = t;
            return *this;
        };

        __forceinline__  __device__ KahanSum operator-=(const float3& value)
        {
            float3 y = -value - c;
            float3 t = sum + y;
            c = (t - sum) - y;
            sum = t;
            return *this;
        };
    };

    // cuda random
    using RandState = curandStateXORWOW_t;
    __forceinline__ __device__ void random_init(uint64_t seed, curandStateXORWOW_t& state) {
        curand_init(seed, 0, 0, &state);
    }
    __forceinline__ __device__ float rand_uniform(curandStateXORWOW_t& random_state) noexcept {
        return curand_uniform(&random_state);
    }
    __forceinline__ __device__ float rand_normal(curandStateXORWOW_t& random_state) noexcept {
        return curand_normal(&random_state); 
    }
}