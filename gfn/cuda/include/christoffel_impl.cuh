
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_RANK 128

// Reduction Utilities
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Optimized Christoffel Device Function
__device__ __forceinline__ void christoffel_device(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma_out, 
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    // Shared Memory Pointers
    float* s_h,          // [rank]
    double* s_E,         // [1] - High Precision
    double* s_P,         // [1] - High Precision
    float* s_M           // [1] - Final Multiplier
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // 1. Reset Shared
    if (tid == 0) {
        *s_E = 0.0;
        *s_P = 0.0;
        *s_M = 1.0f;
    }
    for (int r = tid; r < rank; r += bdim) s_h[r] = 0.0f;
    __syncthreads();

    // 2. Active Inference Multipliers
    if (use_active) {
        float p_E = 0.0f, p_P = 0.0f;
        for (int i = tid; i < dim; i += bdim) {
            if (plasticity != 0.0f) p_E += v[i] * v[i];
            if (x != nullptr && V_w != nullptr) p_P += x[i] * V_w[i];
        }
        // Reduction
        p_E = warpReduceSum(p_E);
        p_P = warpReduceSum(p_P);
        if ((tid & 31) == 0) {
            if (plasticity != 0.0f) atomicAdd(s_E, p_E);
            if (x != nullptr && V_w != nullptr) atomicAdd(s_P, p_P);
        }
        __syncthreads();
        
        if (tid == 0) {
            float m = 1.0f;
            if (plasticity != 0.0f) m *= (1.0f + plasticity * tanhf(*s_E / (float)dim));
            if (x != nullptr && V_w != nullptr) {
                float pot = 1.0f / (1.0f + expf(-*s_P));
                if (pot > sing_thresh) m *= sing_strength;
            }
            *s_M = m;
        }
    }

    // 3. Projection: h = U^T v
    // Coalesced Matrix-Vector Product
    for (int r = 0; r < rank; ++r) {
        float local_h = 0.0f;
        for (int i = tid; i < dim; i += bdim) local_h += v[i] * U[i * rank + r];
        
        local_h = warpReduceSum(local_h);
        if ((tid & 31) == 0) atomicAdd(&s_h[r], local_h);
    }
    __syncthreads();

    // 4. Reconstruction: gamma = W * h^2 * S
    // a) Norm S
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += bdim) local_nsq += s_h[r] * s_h[r];
    local_nsq = warpReduceSum(local_nsq);
    
    // We reuse s_E temporarily to store local_nsq sum
    if (tid == 0) *s_E = 0.0f; 
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(s_E, local_nsq);
    __syncthreads();
    
    float S = 1.0f / (1.0f + sqrtf(*s_E));
    float final_m = *s_M;

    // b) Gamma
    for (int i = tid; i < dim; i += bdim) {
        float g = 0.0f;
        for (int r = 0; r < rank; r++) g += W[i * rank + r] * s_h[r] * s_h[r];
        
        float res = g * S * final_m;
        // FINAL CLAMP AT END
        gamma_out[i] = fminf(fmaxf(res, -5.0f), 5.0f);
    }
}
