
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 512

/**
 * ULTRA-PERFORMANCE Recurrent Manifold Fused Kernel (v2.1)
 * -----------------------------------------------
 * Consolidates the loop over sequence length AND the loop over layers.
 * 
 * Audit Fixes:
 * 1. Read force from SHARED memory (s_force) inside the layer loop.
 * 2. Optimized bank alignment for shared doubles.
 * 3. Verified indexing for batch/seq/dim dimensions.
 */
__global__ void recurrent_manifold_fused_kernel(
    float* __restrict__ x_inout,      // [B, D] - Initial/Final state
    float* __restrict__ v_inout,      // [B, D] - Initial/Final state
    const float* __restrict__ forces,  // [B, T, D] - External forces
    const float* __restrict__ U_stack, // [L, D, R] - Stacked projection matrices
    const float* __restrict__ W_stack, // [L, D, R] - Stacked weighting matrices
    float* __restrict__ x_out_seq,    // [B, T, D] - Trajectory output (Optional)
    const int batch,
    const int seq_len,
    const int dim,
    const int rank,
    const int num_layers,
    const float dt,
    const float dt_scale
) {
    extern __shared__ float s_mem_f[];
    
    // Memory Budget for dim=512: 4*512*4 + rank*4 = 8KB + 512B = 8.5KB.
    float* s_x     = s_mem_f;
    float* s_v     = s_x + dim;
    float* s_gamma = s_v + dim;
    float* s_force = s_gamma + dim;
    float* s_h     = s_force + dim;
    
    // Aligned double storage for active inference reductions
    double* s_double = (double*)(s_h + rank + (rank % 2));
    double* s_E = s_double;
    double* s_P = s_E + 1;
    float* s_M  = (float*)(s_P + 1);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    if (b >= batch) return;

    // 1. Initial Load of State (x0, v0)
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_inout[b * dim + i];
        s_v[i] = v_inout[b * dim + i];
    }
    __syncthreads();

    const float eff_dt = dt * dt_scale;

    // === OUTER LOOP: SEQUENCE TIME ===
    for (int t = 0; t < seq_len; t++) {
        
        // a) Load force for this token ONCE into shared memory
        // This is a major win: we read 'forces' from global memory once,
        // then reuse it for all L layers in the inner loop.
        for (int i = tid; i < dim; i += blockDim.x) {
            s_force[i] = forces[(b * seq_len + t) * dim + i];
        }
        __syncthreads();

        // b) INNER LOOP: LAYERS
        for (int l = 0; l < num_layers; l++) {
            
            const float* U_l = U_stack + (l * dim * rank);
            const float* W_l = W_stack + (l * dim * rank);

            // Compute Manifold Geometry (Γ)
            christoffel_device(
                s_v, U_l, W_l, s_gamma, s_x, nullptr, 
                dim, rank, 0.0f, 1.0f, 1.0f, false, 
                s_h, s_E, s_P, s_M
            );
            __syncthreads();

            // Unified State Update (Recurrent Evolution)
            for (int i = tid; i < dim; i += blockDim.x) {
                // v[t+1] = v[t] + dt * (F[t] - Gamma[t])
                // Note: we use the shared s_force loaded at the start of this timestep
                s_v[i] += eff_dt * (s_force[i] - s_gamma[i]);
                // x[t+1] = x[t] + dt * v[t+1]
                s_x[i] += eff_dt * s_v[i];
            }
            __syncthreads();
        }

        // c) Stream back frame to trajectory buffer if requested (for Readout)
        if (x_out_seq != nullptr) {
            for (int i = tid; i < dim; i += blockDim.x) {
                x_out_seq[(b * seq_len + t) * dim + i] = s_x[i];
            }
        }
    }

    // 2. Final Write Back of sequence-updated state
    for (int i = tid; i < dim; i += blockDim.x) {
        x_inout[b * dim + i] = s_x[i];
        v_inout[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, float dt_scale,
    cudaStream_t stream
) {
    const int shared_bytes = (4 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    recurrent_manifold_fused_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x_state, v_state, forces, U_stack, W_stack, x_out_seq,
        batch, seq_len, dim, rank, num_layers, dt, dt_scale
    );
}
