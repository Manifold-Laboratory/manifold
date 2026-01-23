#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define MAX_RANK 128

__device__ void compute_gamma_block(
    const float* v_global,
    const float* U,
    const float* W,
    float* s_gamma,
    float* s_U,
    int dim,
    int rank
) {
    if (threadIdx.x < rank) s_U[threadIdx.x] = 0.0f;
    __syncthreads();
    
    for (int r = 0; r < rank; r++) {
        float partial = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            partial += v_global[i] * U[i * rank + r];
        }
        atomicAdd(&s_U[r], partial);
    }
    __syncthreads();
    
    // Saturation scale calculation
    __shared__ float s_norm;
    if (threadIdx.x == 0) {
        float n_sq = 0.0f;
        for (int r = 0; r < rank; r++) n_sq += s_U[r] * s_U[r];
        s_norm = sqrtf(n_sq);
    }
    __syncthreads();
    
    float scale = 1.0f / (1.0f + s_norm);
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = 0.0f;
        for (int r = 0; r < rank; r++) {
            float proj = s_U[r];
            val += W[i * rank + r] * proj * proj * scale;
        }
        s_gamma[i] = fminf(fmaxf(val, -5.0f), 5.0f);
    }
    __syncthreads();
}

__global__ void leapfrog_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ x_new,
    float* __restrict__ v_new,
    const float dt,
    const float dt_scale,
    const int batch,
    const int dim,
    const int rank
) {
    const int b = blockIdx.x;
    if (b >= batch) return;
    
    extern __shared__ float shared_mem[];
    float* s_U = shared_mem;
    float* s_gamma = s_U + MAX_RANK;
    float* s_v_half = s_gamma + dim;
    
    const float effective_dt = dt * dt_scale;
    const float* v_b = v + b * dim;
    const float* f_b = f + b * dim;
    const float* x_b = x + b * dim;
    float* x_new_b = x_new + b * dim;
    float* v_new_b = v_new + b * dim;

    compute_gamma_block(v_b, U, W, s_gamma, s_U, dim, rank);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v_val = v_b[i];
        float f_val = f_b[i];
        float g_val = s_gamma[i];
        
        float v_h = v_val + 0.5f * effective_dt * (f_val - g_val);
        s_v_half[i] = v_h;
        
        float x_val = x_b[i];
        x_new_b[i] = x_val + effective_dt * v_h;
    }
    __syncthreads();
    
    compute_gamma_block(s_v_half, U, W, s_gamma, s_U, dim, rank);
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float f_val = f_b[i];
        float g_val = s_gamma[i];
        float v_h = s_v_half[i];
        
        v_new_b[i] = v_h + 0.5f * effective_dt * (f_val - g_val);
    }
}

extern "C" void launch_leapfrog_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    cudaStream_t stream
) {
    const int threads = BLOCK_SIZE;
    const int blocks = batch;
    const int shared_bytes = (MAX_RANK + 2 * dim) * sizeof(float);
    
    leapfrog_fused_kernel<<<blocks, threads, shared_bytes, stream>>>(
        x, v, f, U, W, x_new, v_new,
        dt, dt_scale, batch, dim, rank
    );
}
