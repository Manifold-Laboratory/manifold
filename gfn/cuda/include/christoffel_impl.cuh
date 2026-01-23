
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_RANK 128

// Device function for Christoffel calculation.
// Assumes called within a block where threads cover 'dim'.
// Requires shared memory pointers to be passed in to avoid extern shared conflicts.
// Device function for Christoffel calculation.
// Assumes indices are relative to the passed pointers (already offset by batch).
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
    // Shared Memory pointers provided by caller
    float* s_U,          // [rank]
    float* s_energy_sum, // [1]
    float* s_potential_sum, // [1]
    float* s_final_mult  // [1]
) {
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // 1. Init Shared
    if (tid == 0) {
        *s_energy_sum = 0.0f;
        *s_potential_sum = 0.0f;
        *s_final_mult = 1.0f;
    }
    if (tid < rank) {
        s_U[tid] = 0.0f;
    }
    __syncthreads();

    // 2. Active Inference Factors
    if (use_active) {
        float p_energy = 0.0f;
        float p_potential = 0.0f;
        bool calc_plasticity = (plasticity != 0.0f);
        bool calc_singularity = (x != nullptr && V_w != nullptr);

        for (int i = tid; i < dim; i += bdim) {
            if (calc_plasticity) {
                float val_v = v[i];
                p_energy += val_v * val_v;
            }
            if (calc_singularity) {
                p_potential += x[i] * V_w[i];
            }
        }
        
        if (calc_plasticity) atomicAdd(s_energy_sum, p_energy);
        if (calc_singularity) atomicAdd(s_potential_sum, p_potential);
        
        __syncthreads();
        
        if (tid == 0) {
            float mult = 1.0f;
            if (calc_plasticity) {
                float mean_energy = *s_energy_sum / (float)dim;
                mult *= (1.0f + plasticity * tanh(mean_energy));
            }
            if (calc_singularity) {
                float potential = 1.0f / (1.0f + expf(-*s_potential_sum));
                
                // Hard Threshold (Matches Python implementation)
                // Python: (potential > sing_thresh).float()
                // This has ZERO gradient w.r.t potential (Step Function).
                // We replicate this to match behavior and stability.
                
                if (potential > sing_thresh) {
                    mult *= sing_strength;
                }
                
                *s_final_mult = mult;
            } else {
                *s_final_mult = mult;
            }
        }
        __syncthreads();
    }

    // 3. Projections (U^T v)
    if (tid == 0) *s_potential_sum = 0.0f; // Reuse potential for Norm Sum
    __syncthreads();

    float local_norm_sq = 0.0f;
    for (int r = 0; r < rank; r++) {
        float partial = 0.0f;
        for (int i = tid; i < dim; i += bdim) {
            partial += v[i] * U[i * rank + r];
        }
        atomicAdd(&s_U[r], partial);
    }
    __syncthreads();
    
    // Compute Norm of Projections (Soft Saturation Factor)
    // scale = 1 / (1 + ||s_U||)
    if (tid < rank) {
        float u_val = s_U[tid];
        atomicAdd(s_potential_sum, u_val * u_val);
    }
    __syncthreads();
    
    float scale = 1.0f;
    if (tid == 0) {
        float norm = sqrtf(*s_potential_sum);
        scale = 1.0f / (1.0f + norm);
        // Store scale for reuse? We don't have slot.
        // We broadcast via shared memory or compute locally?
        // Reuse s_potential_sum to broadcast scale
        *s_potential_sum = scale;
    }
    __syncthreads();
    scale = *s_potential_sum; // Load broadcasted scale

    // 4. Reconstruction (Gamma = W * proj^2 * scale)
    float final_mult = *s_final_mult;
    for (int i = tid; i < dim; i += bdim) {
        float val = 0.0f;
        for (int r = 0; r < rank; r++) {
            float proj = s_U[r];
            // Soft-Saturated Quadratic Term: (proj^2) / (1 + |proj_vec|)
            val += W[i * rank + r] * proj * proj * scale;
        }
        val = fminf(fmaxf(val, -5.0f), 5.0f);
        
        // Write output
        if (gamma_out != nullptr) {
            gamma_out[i] = val * final_mult;
        }
    }
}
