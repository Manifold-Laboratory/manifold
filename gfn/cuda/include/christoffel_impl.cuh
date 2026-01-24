
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
                float pot = 1.0f / (1.0f + expf(-fminf(fmaxf(*s_P, -20.0), 20.0))); // Safe Exp
                if (pot > sing_thresh) m *= sing_strength;
            }
            // Absolute clamp for safety
            *s_M = fminf(fmaxf(m, 0.1f), 10.0f);
        }
    }

    // 3. Projection: h = U^T v
    // Coalesced Matrix-Vector Product
    for (int r = tid; r < rank; r += bdim) {
        float local_h = 0.0f;
        for (int i = 0; i < dim; i++) {
            local_h += v[i] * U[i * rank + r];
        }
        s_h[r] = local_h; // Direct assignment since 1 block per head/batch
    }
    __syncthreads();

    // 4. Reconstruction: gamma = W * h^2 * S
    // a) Norm S
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += bdim) local_nsq += s_h[r] * s_h[r];
    local_nsq = warpReduceSum(local_nsq);
    
    if (tid == 0) *s_E = 0.0; 
    __syncthreads();
    
    // Proper shared memory block reduction for S
    if ((tid & 31) == 0) atomicAdd(s_E, (double)local_nsq);
    __syncthreads();
    
    float S = 1.0f / (1.0f + sqrtf((float)*s_E) + 1e-6f);
    float final_m = *s_M;

    // b) Gamma Reconstruction
    for (int i = tid; i < dim; i += bdim) {
        float g = 0.0f;
        for (int r = 0; r < rank; r++) g += W[i * rank + r] * s_h[r] * s_h[r];
        
        float res = g * S * final_m;
        // FINAL CLAMP AT END
        gamma_out[i] = fminf(fmaxf(res, -5.0f), 5.0f);
    }
}
// Backward of Christoffel w.r.t velocity v
// Computes s_gv += (dL/dGamma)^T * (dGamma/dv)
__device__ __forceinline__ void christoffel_v_backward_device(
    const float* __restrict__ s_dGamma,  // [Dim]
    const float* __restrict__ U,         // [Dim, Rank]
    const float* __restrict__ W,         // [Dim, Rank]
    const float* __restrict__ s_h,       // [Rank]
    const float* __restrict__ s_v,       // [Dim]
    float* __restrict__ s_gv,            // [Dim] -> Cumulative update
    const int dim,
    const int rank,
    float plasticity,
    bool use_active,
    // Shared Memory Pointers
    const float S,          // Pre-computed Norm Scaling
    const float M,          // Pre-computed Active Multiplier
    float* s_dL_dh_shared   // Scratch [rank]
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // 1. Compute dL/dh [Rank]
    // dL/dh_r approx = sum_i (dL/dGamma_i * W_ir) * 2 * h_r * S * M
    for (int r = tid; r < rank; r += bdim) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += s_dGamma[i] * W[i * rank + r];
        }
        s_dL_dh_shared[r] = sum * 2.0f * s_h[r] * S * M;
    }
    __syncthreads();

    // 2. Compute dL/dv = dL/dh @ U^T
    // s_gv[i] += sum_r (dL/dh_r * U_ir)
    for (int i = tid; i < dim; i += bdim) {
        float grad_v_i = 0.0f;
        for (int r = 0; r < rank; r++) {
            grad_v_i += s_dL_dh_shared[r] * U[i * rank + r];
        }
        
        // 3. Optional: Reactive Plasticity Gradient dL/dv (Energy part)
        // Gamma = ... * (1 + p * v^2/dim)
        // dL/dv_i_react = dL/dGamma_sum * p * 2v_i/dim ... (approx)
        // For now, we only use the geometric part.
        
        s_gv[i] += grad_v_i;
    }
    __syncthreads();
}
