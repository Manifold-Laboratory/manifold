#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level scan for small sequences (< 32 elements)
__device__ void warp_scan(
    float& a_local,
    float& x_local,
    const int lane_id
) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float a_prev = __shfl_up_sync(0xffffffff, a_local, offset);
        float x_prev = __shfl_up_sync(0xffffffff, x_local, offset);
        
        if (lane_id >= offset) {
            // Operator composition: (a2, x2) ⊗ (a1, x1) = (a2*a1, a2*x1 + x2)
            x_local = a_local * x_prev + x_local;
            a_local = a_local * a_prev;
        }
    }
}

// Block-level scan using Blelloch algorithm (work-efficient)
__device__ void block_scan(
    float* s_a,
    float* s_x,
    const int n,
    const int tid
) {
    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            // Compose operators
            float a_new = s_a[ai] * s_a[bi];
            float x_new = s_a[bi] * s_x[ai] + s_x[bi];
            
            s_a[bi] = a_new;
            s_x[bi] = x_new;
        }
        offset *= 2;
    }
    
    // Clear last element (identity)
    if (tid == 0) {
        s_a[n - 1] = 1.0f;
        s_x[n - 1] = 0.0f;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            float a_tmp = s_a[ai];
            float x_tmp = s_x[ai];
            
            s_a[ai] = s_a[bi];
            s_x[ai] = s_x[bi];
            
            s_a[bi] = a_tmp * s_a[bi];
            s_x[bi] = a_tmp * s_x[bi] + x_tmp;
        }
    }
    __syncthreads();
}

// Main parallel scan kernel
__global__ void parallel_scan_fused_kernel(
    const float* __restrict__ a,    // Decay factors [B, L, D]
    const float* __restrict__ x,    // Additive terms [B, L, D]
    float* __restrict__ y,          // Output [B, L, D]
    const int batch,
    const int seq_len,
    const int dim,
    const float plasticity
) {
    // Each block handles one batch and one dimension
    const int b = blockIdx.x;
    const int d = blockIdx.y;
    
    if (b >= batch || d >= dim) return;
    
    const int base_idx = (b * seq_len + 0) * dim + d;
    const int stride = dim;
    
    // Shared memory for scan (padded to avoid bank conflicts)
    extern __shared__ float shared_mem[];
    float* s_a = shared_mem;
    float* s_x = s_a + (BLOCK_SIZE + BLOCK_SIZE / 32);  // Padding
    
    // Process sequence in chunks of BLOCK_SIZE
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += BLOCK_SIZE) {
        const int tid = threadIdx.x;
        const int seq_idx = chunk_start + tid;
        const int chunk_size = min(BLOCK_SIZE, seq_len - chunk_start);
        
        // Load data into shared memory
        if (tid < chunk_size) {
            const int idx = base_idx + seq_idx * stride;
            float a_val = a[idx];
            
            // Active Inference Modulation:
            // A_new = A_old * (1 + plasticity * tanh(x^2))? 
            // Simplified: A_new = A_old + plasticity * ...
            // Let's assume input 'a' is raw decay.
            // We apply plasticity here if needed.
            // For now, identity pass-through as placeholder or simple scaling.
            if (plasticity != 0.0f) {
                // Example: modulate forget gate by energy of input x
                // float energy = tanh(x[idx] * x[idx]);
                 // a_val = a_val * (1.0f + plasticity * energy);
            }
            
            s_a[tid] = a_val;
            s_x[tid] = x[idx];
        } else {
            // Padding with identity
            s_a[tid] = 1.0f;
            s_x[tid] = 0.0f;
        }
        __syncthreads();
        
        // Perform block-level scan
        if (chunk_size <= WARP_SIZE) {
            // Use warp-level scan for small chunks
            const int lane_id = tid % WARP_SIZE;
            if (tid < chunk_size) {
                float a_local = s_a[tid];
                float x_local = s_x[tid];
                warp_scan(a_local, x_local, lane_id);
                s_a[tid] = a_local;
                s_x[tid] = x_local;
            }
            __syncthreads();
        } else {
            // Use Blelloch scan for larger chunks
            block_scan(s_a, s_x, BLOCK_SIZE, tid);
        }
        
        // Apply carry from previous chunk
        if (chunk_start > 0 && tid < chunk_size) {
            // Get carry from last element of previous chunk
            // This is stored in y[chunk_start-1]
            const int carry_idx = base_idx + (chunk_start - 1) * stride;
            float a_carry = a[carry_idx];  // We need to accumulate this
            float x_carry = y[carry_idx];
            
            // Compose with current element
            s_x[tid] = a_carry * s_x[tid] + x_carry;
        }
        __syncthreads();
        
        // Write results
        if (tid < chunk_size) {
            const int idx = base_idx + seq_idx * stride;
            y[idx] = s_x[tid];
        }
    }
}

// Launcher function
extern "C" void launch_parallel_scan_fused(
    const float* a,
    const float* x,
    float* y,
    int batch,
    int seq_len,
    int dim,
    float plasticity,
    cudaStream_t stream
) {
    // Grid: (batch, dim)
    // Block: BLOCK_SIZE threads
    dim3 grid(batch, dim);
    dim3 block(BLOCK_SIZE);
    
    // Shared memory: 2 * (BLOCK_SIZE + padding) * sizeof(float)
    const int padding = BLOCK_SIZE / 32;
    const int shared_bytes = 2 * (BLOCK_SIZE + padding) * sizeof(float);
    
    parallel_scan_fused_kernel<<<grid, block, shared_bytes, stream>>>(
        a, x, y, batch, seq_len, dim, plasticity
    );
}
