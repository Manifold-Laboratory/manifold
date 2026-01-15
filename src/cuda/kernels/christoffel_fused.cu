#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define MAX_RANK 128

__global__ void christoffel_fused_kernel(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active
) {
    __shared__ float s_U[MAX_RANK];
    __shared__ float s_energy_sum;
    __shared__ float s_potential_sum;
    __shared__ float s_final_mult;

    const int b = blockIdx.x;
    if (b >= batch) return;
    
    if (threadIdx.x == 0) {
        s_energy_sum = 0.0f;
        s_potential_sum = 0.0f;
        s_final_mult = 1.0f;
    }
    if (threadIdx.x < rank) {
        s_U[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    if (use_active) {
        float p_energy = 0.0f;
        float p_potential = 0.0f;
        
        bool calc_plasticity = (plasticity != 0.0f);
        bool calc_singularity = (x != nullptr && V_w != nullptr);
        
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            if (calc_plasticity) {
                float val_v = v[b * dim + i];
                p_energy += val_v * val_v;
            }
            if (calc_singularity) {
                p_potential += x[b * dim + i] * V_w[i];
            }
        }
        
        if (calc_plasticity) atomicAdd(&s_energy_sum, p_energy);
        if (calc_singularity) atomicAdd(&s_potential_sum, p_potential);
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            float mult = 1.0f;
            if (calc_plasticity) {
                float mean_energy = s_energy_sum / (float)dim;
                mult *= (1.0f + plasticity * tanh(mean_energy));
            }
            if (calc_singularity) {
                float potential = 1.0f / (1.0f + expf(-s_potential_sum));
                if (potential > sing_thresh) mult *= sing_strength;
            }
            s_final_mult = mult;
        }
        __syncthreads();
    }

    for (int r = 0; r < rank; r++) {
        float partial_sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            partial_sum += v[b * dim + i] * U[i * rank + r];
        }
        atomicAdd(&s_U[r], partial_sum);
    }
    __syncthreads();
    
    float final_mult = s_final_mult;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = 0.0f;
        for (int r = 0; r < rank; r++) {
            float proj = s_U[r];
            val += W[i * rank + r] * proj * proj;
        }
        val = fminf(fmaxf(val, -5.0f), 5.0f);
        gamma[b * dim + i] = val * final_mult;
    }
}

// Raw pointer launcher
extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
) {
    const int threads = BLOCK_SIZE;
    const int blocks = batch;
    
    christoffel_fused_kernel<<<blocks, threads, 0, stream>>>(
        v, U, W, gamma, x, V_w,
        batch, dim, rank,
        plasticity, sing_thresh, sing_strength,
        use_active
    );
}
