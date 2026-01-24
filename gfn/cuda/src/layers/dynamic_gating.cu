#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/**
 * DYNAMIC GATING KERNEL
 * ---------------------
 * Computes curvature-based time-step modulation via 2-layer MLP.
 * Architecture: x -> Linear(dim, dim/4) -> Tanh -> Linear(dim/4, 1) -> Sigmoid
 */

__global__ void dynamic_gating_kernel(
    const float* __restrict__ x,        // [B, D]
    const float* __restrict__ W1,       // [D/4, D] (transposed for efficiency)
    const float* __restrict__ b1,       // [D/4]
    const float* __restrict__ W2,       // [1, D/4] (transposed)
    const float* __restrict__ b2,       // [1]
    float* __restrict__ dt_scale,       // [B, 1]
    const int batch,
    const int dim,
    const int hidden_dim
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (b >= batch) return;
    
    extern __shared__ float s_mem[];
    float* s_x = s_mem;                     // [dim]
    float* s_hidden = s_x + dim;            // [hidden_dim]
    
    // Load input
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x[b * dim + i];
    }
    __syncthreads();
    
    // Layer 1: hidden = tanh(x @ W1^T + b1)
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float sum = b1[i];
        for (int j = 0; j < dim; j++) {
            sum += s_x[j] * W1[i * dim + j];  // W1[i, j]
        }
        s_hidden[i] = tanhf(sum);
    }
    __syncthreads();
    
    // Layer 2: out = sigmoid(hidden @ W2^T + b2)
    if (tid == 0) {
        float sum = b2[0];
        for (int j = 0; j < hidden_dim; j++) {
            sum += s_hidden[j] * W2[j];  // W2[0, j]
        }
        dt_scale[b] = 1.0f / (1.0f + expf(-sum));  // sigmoid
    }
}

extern "C" void launch_dynamic_gating_fused(
    const float* x, const float* W1, const float* b1,
    const float* W2, const float* b2,
    float* dt_scale,
    int batch, int dim, int hidden_dim,
    cudaStream_t stream
) {
    const int shared_bytes = (dim + hidden_dim) * sizeof(float);
    
    dynamic_gating_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x, W1, b1, W2, b2, dt_scale, batch, dim, hidden_dim
    );
}
