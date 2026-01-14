/*
 * Fused Christoffel Symbol Kernel
 * ================================
 * 
 * Computes Γ(v,v) = W * (U^T v)^2 in a single fused kernel.
 * 
 * Mathematical Operation:
 *   1. proj = U^T * v      [rank]
 *   2. sq = proj^2         [rank]
 *   3. gamma = W * sq      [dim]
 * 
 * Performance:
 *   - Fuses 3 operations into 1 kernel launch
 *   - Keeps intermediate results in registers/shared memory
 *   - Expected speedup: 2-3x over PyTorch
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <valarray>
#include <forward_list>

#include <torch/extension.h>

#define BLOCK_SIZE 256
#define MAX_RANK 128  // Max rank for shared memory strategy

// Fused kernel: computes gamma[b, d] = sum_r W[d,r] * (sum_i U[i,r] * v[b,i])^2
__global__ void christoffel_fused_kernel(
    const float* __restrict__ v,      // [batch, dim]
    const float* __restrict__ U,      // [dim, rank]
    const float* __restrict__ W,      // [dim, rank]
    float* __restrict__ gamma,        // [batch, dim]
    const int batch,
    const int dim,
    const int rank
) {
    // Shared memory for U projection
    __shared__ float s_U[MAX_RANK];
    
    // Initialize to 0
    if (threadIdx.x < rank) {
        s_U[threadIdx.x] = 0.0f;
    }
    
    // Load v into shared memory for faster reuse? 
    // Actually v is accessed many times (for each r), so caching v is good.
    // But dim might be > BLOCK_SIZE. For now assume dim <= BLOCK_SIZE or handle tiling.
    // Given the constraints, let's just read v from global. L1 cache handles it.
    
    __syncthreads();
    
    const int b = blockIdx.x;  // One block per batch item
    
    if (b >= batch) return;
    
    // Iterate over ranks
    // Each thread computes partial dot product for a subset of dimensions
    for (int r = 0; r < rank; r++) {
        float partial_sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            partial_sum += v[b * dim + i] * U[i * rank + r];
        }
        
        // Atomic reduction for robustness (perf is fine for low rank/dim)
        atomicAdd(&s_U[r], partial_sum);
    }
    
    __syncthreads();
    
    // Now compute gamma output elements
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = 0.0f;
        for (int r = 0; r < rank; r++) {
            float proj = s_U[r];
            val += W[i * rank + r] * proj * proj;
        }
        // Clamp result
        val = fminf(fmaxf(val, -5.0f), 5.0f);
        gamma[b * dim + i] = val;
    }
}

// Host function
torch::Tensor christoffel_fused_cuda(
    torch::Tensor v,  // [batch, dim]
    torch::Tensor U,  // [dim, rank]
    torch::Tensor W   // [dim, rank]
) {
    const int batch = v.size(0);
    const int dim = v.size(1);
    const int rank = U.size(1);
    
    TORCH_CHECK(rank <= MAX_RANK, "Rank exceeds MAX_RANK");
    
    auto gamma = torch::empty({batch, dim}, v.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks = batch;
    
    christoffel_fused_kernel<<<blocks, threads>>>(
        v.data_ptr<float>(),
        U.data_ptr<float>(),
        W.data_ptr<float>(),
        gamma.data_ptr<float>(),
        batch, dim, rank
    );
    
    return gamma;
}
