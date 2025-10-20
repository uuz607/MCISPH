#pragma once
#include "Common.cuh"

namespace mcisph
{
    template<SPHKernelType T>
    struct SPHKernelCoeff
    {
        float h;
        float coeff;
    };

    using SPHKernelWCoeff = SPHKernelCoeff<CUBIC_SPLINE>;
    using SPHKernelDWCoeff = SPHKernelCoeff<CUBIC_SPLINE>;

    template<SPHKernelType T>
    __forceinline__ __host__ __device__ void getWCoeff(float h, SPHKernelCoeff<T>& wcoeff);

    template<SPHKernelType T>
    __forceinline__ __host__ __device__ void getDWCoeff(float h, SPHKernelCoeff<T>& dwcoeff);

    __forceinline__ __host__ __device__ void getWCoeff(float h, SPHKernelCoeff<CUBIC_SPLINE>& wcoeff)
    {
        wcoeff.h = h;
        float h3 = h * h * h;
        wcoeff.coeff = 8.0f / (M_PIf * h3);
    }

    __forceinline__ __device__ float W(const float3& r_vec, const SPHKernelCoeff<CUBIC_SPLINE>& wcoeff)
    {
        float val = 0.0f;
        float r = length(r_vec);
        const float q = r / wcoeff.h;
        if (q <= 1.0)
        {
            if (q <= 0.5)
            {
                const float q2 = q * q;
                const float q3 = q2 * q;
                val = wcoeff.coeff * (6.0f * q3 - 6.0f * q2 + 1.0f);
            }
            else
            {
                val = wcoeff.coeff * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
            }
        }
        return val;
    }

    __forceinline__ __host__ __device__ void getDWCoeff(float h, SPHKernelCoeff<CUBIC_SPLINE>& dwcoeff)
    {
        dwcoeff.h = h;
        const float h3 = h * h * h;
        dwcoeff.coeff = 48.f / (M_PIf * h3);
    }

    __forceinline__ __device__ float3 dW(const float3& r_vec, const SPHKernelCoeff<CUBIC_SPLINE>& dwcoeff)
    {
        float3 val = make_float3(0.f);
        float r = length(r_vec);
        const float q = r / dwcoeff.h;
        if (r > 1.0e-9f && q <= 1.f)
        {
            float3 gradq = r_vec / r;
            gradq /= dwcoeff.h;

            if (q <= 0.5)
            {
                val = dwcoeff.coeff * q * (3.f * q - 2.f) * gradq;
            }
            else
            {
                float factor = 1.f - q;
                val = dwcoeff.coeff * (-factor * factor) * gradq;
            }
        }
        return val;
    }
}