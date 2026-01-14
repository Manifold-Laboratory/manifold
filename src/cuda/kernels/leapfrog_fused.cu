/*
 * Fused Leapfrog Integrator Kernel
 * =================================
 * 
 * Computes a complete symplectic Leapfrog step with inline Christoffel:
 * 
 *   v_half = v + 0.5 * dt * dt_scale * (f - Γ(v))
 *   x_new = x + dt * dt_scale * v_half
 *   v_new = v_half + 0.5 * dt * dt_scale * (f - Γ(v_half))
 * 
 * Performance:
 *   - Eliminates 8+ kernel launches per layer
 *   - All intermediate values kept in registers
 *   - Expected speedup: 4-5x
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256
#define MAX_RANK 128

// Helper: Compute Christoffel vector for the block's batch item
// Stores result in s_gamma (size: dim) (assumes s_gamma allocated by caller)
// Uses s_U (size: rank) as temp buffer.
__device__ void compute_gamma_block(
    const float* v_global, // pointer to v[b, 0]
    const float* U,
    const float* W,
    float* s_gamma, // Output to shared memory
    float* s_U,     // Temp buffer for projections [MAX_RANK]
    int dim,
    int rank
) {
    // 1. Compute Projections U^T * v
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
    
    // 2. Compute Gamma = W * (proj^2)
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = 0.0f;
        for (int r = 0; r < rank; r++) {
            float proj = s_U[r];
            val += W[i * rank + r] * proj * proj;
        }
        s_gamma[i] = fminf(fmaxf(val, -5.0f), 5.0f);
    }
    __syncthreads();
}

__global__ void leapfrog_fused_kernel(
    const float* __restrict__ x,      // [batch, dim]
    const float* __restrict__ v,      // [batch, dim]
    const float* __restrict__ f,      // [batch, dim]
    const float* __restrict__ U,      // [dim, rank]
    const float* __restrict__ W,      // [dim, rank]
    float* __restrict__ x_new,        // [batch, dim]
    float* __restrict__ v_new,        // [batch, dim]
    const float dt,
    const float dt_scale,
    const int batch,
    const int dim,
    const int rank
) {
    // One block per batch item
    const int b = blockIdx.x;
    if (b >= batch) return;
    
    // Shared memory layout:
    // s_U [MAX_RANK]
    // s_gamma [dim]  <-- Requires dynamic shared mem if dim is large.
    // For now, let's just use dynamic shared memory for everything.
    
    extern __shared__ float shared_mem[];
    float* s_U = shared_mem;              // [MAX_RANK]
    float* s_gamma = s_U + MAX_RANK;      // [dim]
    // We don't need s_v if we read from global, 
    // BUT we need to store v_half in shared or global because it's input to the second gamma.
    // Let's store v_half in s_gamma space temporarily? No, we need s_gamma.
    // Let's allocate s_v_half [dim]
    float* s_v_half = s_gamma + dim;      // [dim]
    
    const float effective_dt = dt * dt_scale;
    const float* v_b = v + b * dim;
    const float* f_b = f + b * dim;
    const float* x_b = x + b * dim;
    float* x_new_b = x_new + b * dim;
    float* v_new_b = v_new + b * dim;

    // --- Step 1: Gamma(v) ---
    compute_gamma_block(v_b, U, W, s_gamma, s_U, dim, rank);

    if (b == 0 && threadIdx.x == 0) {
       // printf("CUDA DEBUG: v[0]=%f, f[0]=%f, gamma[0]=%f\n", v_b[0], f_b[0], s_gamma[0]);
    }
    
    // --- Step 2: v_half & x_new ---
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v_val = v_b[i];
        float f_val = f_b[i];
        float g_val = s_gamma[i];
        
        float v_h = v_val + 0.5f * effective_dt * (f_val - g_val);
        s_v_half[i] = v_h; // Store for next step
        
        float x_val = x_b[i];
        x_new_b[i] = x_val + effective_dt * v_h;
        
        if (b == 0 && i == 0) {
             // printf("CUDA DEBUG: v_half[0]=%f\n", v_h);
        }
    }
    __syncthreads();
    
    // --- Step 3: Gamma(v_half) ---
    compute_gamma_block(s_v_half, U, W, s_gamma, s_U, dim, rank);
    
    // --- Step 4: v_new ---
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float f_val = f_b[i];
        float g_val = s_gamma[i];
        float v_h = s_v_half[i];
        
        v_new_b[i] = v_h + 0.5f * effective_dt * (f_val - g_val);
    }
}

// Host function
std::vector<torch::Tensor> leapfrog_fused_cuda(
    torch::Tensor x,          // [batch, dim]
    torch::Tensor v,          // [batch, dim]
    torch::Tensor f,          // [batch, dim]
    torch::Tensor U,          // [dim, rank]
    torch::Tensor W,          // [dim, rank]
    float dt,
    float dt_scale
) {
    const int batch = x.size(0);
    const int dim = x.size(1);
    const int rank = U.size(1);
    
    TORCH_CHECK(rank <= MAX_RANK, "Rank exceeds MAX_RANK");
    
    auto x_new = torch::empty_like(x);
    auto v_new = torch::empty_like(v);
    
    const int threads = BLOCK_SIZE;
    const int blocks = batch; // One block per batch item
    
    // Shared mem: MAX_RANK + 2 * dim (s_gamma, s_v_half)
    const int shared_bytes = (MAX_RANK + 2 * dim) * sizeof(float);
    
    leapfrog_fused_kernel<<<blocks, threads, shared_bytes>>>(
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        f.data_ptr<float>(),
        U.data_ptr<float>(),
        W.data_ptr<float>(),
        x_new.data_ptr<float>(),
        v_new.data_ptr<float>(),
        dt, dt_scale,
        batch, dim, rank
    );
    
    return {x_new, v_new};
}
