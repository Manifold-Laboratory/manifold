#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 256

// Helper for index calculation
__device__ int get_idx(int batch_idx, int dim_idx, int dim) {
    return batch_idx * dim + dim_idx;
}

// ------------------------------------------------------------------
// 1. Christoffel Symbols Kernel (Low Rank Approximation)
// Gamma = sum_k (w_k * x) * w_k  (Simplified contraction for demonstration)
// Real metric contraction is more complex, this is a placeholder for the "Beast" optimization
// that maps to existing pytorch naive implementation structure.
// ------------------------------------------------------------------
__global__ void christoffel_kernel(
    const float* __restrict__ x,            // [B, D]
    const float* __restrict__ weights,      // [R, D] (Metric basis)
    float* __restrict__ gamma,              // [B, D]
    int batch_size,
    int dim,
    int rank
) {
    int b = blockIdx.x;
    int d = threadIdx.x;

    if (b < batch_size && d < dim) {
        float sum_gamma = 0.0f;
        
        // Compute contraction loop (naive O(R) per thread)
        // In reality, this should use shared memory tiling for "Beast" speed.
        
        for (int r = 0; r < rank; ++r) {
            // Dot product x . w_r
            float dot = 0.0f;
            for (int k = 0; k < dim; ++k) {
                dot += x[b * dim + k] * weights[r * dim + k];
            }
            
            // Project back: dot * w_r[d]
            sum_gamma += dot * weights[r * dim + d];
        }
        
        gamma[b * dim + d] = sum_gamma;
    }
}

void christoffel_fused_cuda(torch::Tensor x, torch::Tensor weights, torch::Tensor gamma) {
    int batch_size = x.size(0);
    int dim = x.size(1);
    int rank = weights.size(0);

    dim3 blocks(batch_size);
    dim3 threads(dim); 
    // Note: If dim > 1024, need strided loop. Assuming dim <= 1024 for demo.
    
    christoffel_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weights.data_ptr<float>(),
        gamma.data_ptr<float>(),
        batch_size, dim, rank
    );
}

// ------------------------------------------------------------------
// 2. Symplectic Leapfrog Kernel
// Fused update: v = v - dt*grad, x = x + dt*v
// ------------------------------------------------------------------
__global__ void leapfrog_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ force,
    float* __restrict__ out_x,
    float* __restrict__ out_v,
    float dt,
    float friction,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float v_curr = v[idx];
        float x_curr = x[idx];
        float f_curr = force[idx];
        
        // Half-step Velocity (Kick)
        // acc = f - friction*v (simplified)
        float acc = f_curr; 
        // Note: For true manifold, acc needs gamma. 
        // Here we assume 'force' already contains -gamma + external_force (pre-computed)
        // or we fuse christoffel computation here (advanced).
        
        float v_half = v_curr + 0.5f * dt * acc;
        
        // Full-step Position (Drift)
        float x_next = x_curr + dt * v_half;
        
        // Re-eval acceleration at x_next? 
        // For strictly fused kernel without re-evaluating gamma internally, 
        // we usually can only do the first half or full explicit Euler.
        // True Leapfrog needs Gamma(x_next).
        // For this V1 kernel, we'll implement a Symplectic Euler or Velocity Verlet 
        // assuming force is constant or pre-calculated. 
        
        // Let's implement the standard update used in MLayer (velocity verlet-ish)
        // v_next = v_half + 0.5 * dt * acc_next
        
        out_x[idx] = x_next;
        out_v[idx] = v_half; // Return half-step v to allow Python to compute next Gamma
    }
}

void leapfrog_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor force, 
                        torch::Tensor out_x, torch::Tensor out_v, float dt, float friction) {
    int size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    leapfrog_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        force.data_ptr<float>(),
        out_x.data_ptr<float>(),
        out_v.data_ptr<float>(),
        dt, friction, size
    );
}
