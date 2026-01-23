#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/**
 * HEAD MIXING KERNEL
 * ------------------
 * Mixes information between attention heads via linear projections.
 * Input: [H, B, D/H] per head
 * Output: [B, D] mixed state
 */

__global__ void head_mixing_kernel(
    const float* __restrict__ x_heads,  // [H, B, D/H]
    const float* __restrict__ v_heads,  // [H, B, D/H]
    const float* __restrict__ W_x,       // [D, D] weight matrix for x
    const float* __restrict__ W_v,       // [D, D] weight matrix for v
    float* __restrict__ x_out,           // [B, D]
    float* __restrict__ v_out,           // [B, D]
    const int heads,
    const int batch,
    const int dim,
    const int head_dim
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (b >= batch) return;
    
    extern __shared__ float s_mem[];
    float* s_concat_x = s_mem;                    // [D]
    float* s_concat_v = s_concat_x + dim;         // [D]
    float* s_out_x = s_concat_v + dim;            // [D]
    float* s_out_v = s_out_x + dim;               // [D]
    
    // Step 1: Concatenate heads [H, B, D/H] -> [B, D]
    for (int i = tid; i < dim; i += blockDim.x) {
        int h = i / head_dim;
        int d = i % head_dim;
        s_concat_x[i] = x_heads[(h * batch + b) * head_dim + d];
        s_concat_v[i] = v_heads[(h * batch + b) * head_dim + d];
    }
    __syncthreads();
    
    // Step 2: Linear projection x_out = concat_x @ W_x^T
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum_x = 0.0f;
        float sum_v = 0.0f;
        
        for (int j = 0; j < dim; j++) {
            sum_x += s_concat_x[j] * W_x[i * dim + j];  // W_x[i, j]
            sum_v += s_concat_v[j] * W_v[i * dim + j];  // W_v[i, j]
        }
        
        s_out_x[i] = sum_x;
        s_out_v[i] = sum_v;
    }
    __syncthreads();
    
    // Step 3: Write to global memory
    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_out_x[i];
        v_out[b * dim + i] = s_out_v[i];
    }
}

extern "C" void launch_head_mixing_fused(
    const float* x_heads, const float* v_heads,
    const float* W_x, const float* W_v,
    float* x_out, float* v_out,
    int heads, int batch, int dim,
    cudaStream_t stream
) {
    const int head_dim = dim / heads;
    const int shared_bytes = 4 * dim * sizeof(float);
    
    head_mixing_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x_heads, v_heads, W_x, W_v, x_out, v_out,
        heads, batch, dim, head_dim
    );
}
