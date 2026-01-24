
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 512

/**
 * RECURRENT MANIFOLD FUSED KERNEL v5.0 (MULTI-HEAD MIXING)
 * ========================================================
 * 1. Execution Order: TIME-MAJOR (Corrects Deep RNN architecture)
 * 2. Integrator: SYMPLECTIC LEAPFROG (Kick-Drift-Kick)
 * 3. Multi-Head Support: ALL heads processed in ONE block per batch item.
 *    Allows inter-head mixing via shared memory.
 */

// Device function for Head Mixing (Linear Projection)
__device__ void head_mixing_device(
    float* s_x,           // [Heads * DimHead] (concatenated in shared mem)
    float* s_v,           // [Heads * DimHead] (concatenated in shared mem)
    const float* W_x,     // [Dim, Dim]
    const float* W_v,     // [Dim, Dim]
    float* s_temp_x,      // Scratch buffer [Dim]
    float* s_temp_v,      // Scratch buffer [Dim]
    int dim,
    int tid
) {
    // Shared Memory Barrier before reading s_x/s_v
    __syncthreads();

    // Parallel Matrix Multiplication: s_temp = s_concat @ W^T
    // Each thread computes one output coordinate (or part of it)
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum_x = 0.0f;
        float sum_v = 0.0f;
        
        for (int j = 0; j < dim; j++) {
            // W is row-major [Dim, Dim]
            // We want y = x @ W^T => y_i = sum_j (x_j * W_ij)
            // But usually Linear is y = x @ W^T where W is [Out, In].
            // Let's assume W is stored as [Out, In] (standard PyTorch Linear weight).
            // So y_i = sum_j (x_j * W[i, j])
            
            float w_val_x = W_x[i * dim + j];
            float w_val_v = W_v[i * dim + j];
            
            sum_x += s_x[j] * w_val_x;
            sum_v += s_v[j] * w_val_v;
        }
        
        s_temp_x[i] = sum_x;
        s_temp_v[i] = sum_v;
    }
    __syncthreads();
    
    // Copy back to main buffers
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = s_temp_x[i];
        s_v[i] = s_temp_v[i];
    }
    __syncthreads();
}

__global__ void recurrent_manifold_fused_kernel(
    float* __restrict__ x_state,      // [B, D]
    float* __restrict__ v_state,      // [B, D]
    const float* __restrict__ forces,  // [B, T, D] 
    const float* __restrict__ U_stack, // [L*H, D/H, R]
    const float* __restrict__ W_stack, // [L*H, D/H, R]
    float* __restrict__ x_out_seq,    // [B, T, D]
    float* __restrict__ reg_loss_out, // [B]
    const float* __restrict__ W_mix_x, // [D, D] (Optional)
    const float* __restrict__ W_mix_v, // [D, D] (Optional)
    const int batch,
    const int seq_len,
    const int dim,
    const int rank,
    const int num_layers,
    const int num_heads,
    const float dt,
    const float* __restrict__ dt_scales,
    const float* __restrict__ forget_rates,
    const float plasticity,
    const float sing_thresh,
    const float sing_strength
) {
    extern __shared__ float s_mem_f[];
    
    // Memory Layout (Total Dimensionality D = dim)
    const int dim_per_head = dim / num_heads;
    const int head_rank = rank / num_heads;
    
    // Buffers
    float* s_x     = s_mem_f;           
    float* s_v     = s_mem_f + dim;     
    float* s_gamma = s_v + dim;         
    float* s_force = s_gamma + dim;     
    float* s_temp  = s_force + dim;     
    float* s_temp2 = s_temp + dim;      
    float* s_h     = s_temp2 + dim;     
    
    double* s_double_base = (double*)(s_h + num_heads * head_rank + 32); 
    
    const int b = blockIdx.x; 
    const int tid = threadIdx.x;
    
    if (b >= batch) return;

    // Load Initial State
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_state[b * dim + i];
        s_v[i] = v_state[b * dim + i];
    }
    __syncthreads();
    
    const float depth_scale = 1.0f / sqrtf((float)num_layers);

    // === TIME LOOP ===
    for (int t = 0; t < seq_len; t++) {
        for (int i = tid; i < dim; i += blockDim.x) s_force[i] = forces[(b * seq_len + t) * dim + i];
        __syncthreads();
        
        for (int l = 0; l < num_layers; l++) {
            // LEVEL 12: SYMMETRY RESTORATION
            // Perform Unit-Norm stabilization at the START of the layer.
            // This ensures the Adjoint BPTT (Backward) matches exactly.
            for (int h = 0; h < num_heads; h++) {
                int head_offset = h * dim_per_head;
                float* s_v_h = s_v + head_offset;
                float* s_P_scr = (float*)((double*)(s_h + num_heads * head_rank + 32) + 1); 

                float local_v_nsq = 0.0f;
                for (int i = tid; i < dim_per_head; i += blockDim.x) local_v_nsq += s_v_h[i] * s_v_h[i];
                
                if (tid == 0) *s_P_scr = 0.0f;
                __syncthreads();
                
                atomicAdd(s_P_scr, local_v_nsq);
                __syncthreads();
                
                float v_norm_val = sqrtf(*s_P_scr + 1e-6f);
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_v_h[i] /= v_norm_val;
                __syncthreads();
            }

            for (int h = 0; h < num_heads; h++) {
                int head_offset = h * dim_per_head;
                int layer_head_idx = l * num_heads + h;
                
                const float h_dt_scale = dt_scales ? dt_scales[h] : 1.0f;
                const float h_mu = forget_rates ? forget_rates[h] : 0.05f;
                const float eff_dt = dt * h_dt_scale * depth_scale;
                const float half_dt = 0.5f * eff_dt;
                
                float* s_v_h = s_v + head_offset;
                float* s_x_h = s_x + head_offset;
                float* s_gamma_h = s_gamma + head_offset;
                const float* U_h = U_stack + (layer_head_idx * dim_per_head * head_rank);
                const float* W_h = W_stack + (layer_head_idx * dim_per_head * head_rank);
                
                float* s_h_scratch = s_h + h * head_rank; 
                double* s_E_scr = s_double_base; 
                double* s_P_scr = s_E_scr + 1;
                float* s_M_scr = (float*)(s_P_scr + 1);

                // --- INTEGRATOR (Leapfrog Kick-Drift-Kick with Spring) ---
                christoffel_device(s_v_h, U_h, W_h, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, s_h_scratch, s_E_scr, s_P_scr, s_M_scr);
                __syncthreads();
                
                // Kick 1: v = v(1 - 0.5*mu*dt) + half_dt * (F - Gamma)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float g = s_gamma_h[i];
                    float sf = s_force[head_offset + i];
                    s_v_h[i] = s_v_h[i] * (1.0f - half_dt * h_mu) + half_dt * (sf - g);
                }
                __syncthreads();
                
                // Drift
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_x_h[i] += eff_dt * s_v_h[i];
                __syncthreads();
                
                // Kick 2
                christoffel_device(s_v_h, U_h, W_h, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, s_h_scratch, s_E_scr, s_P_scr, s_M_scr);
                __syncthreads();
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float g = s_gamma_h[i]; 
                    float sf = s_force[head_offset + i];
                    s_v_h[i] = s_v_h[i] * (1.0f - half_dt * h_mu) + half_dt * (sf - g);
                }
                __syncthreads();

                __syncthreads();
            } 
            
            // Mixed path
            if (num_heads > 1 && W_mix_x != nullptr && l < num_layers - 1) {
                head_mixing_device(s_x, s_v, W_mix_x, W_mix_v, s_temp, s_temp2, dim, tid);
            }
        } 
        
        __syncthreads();

        // Write Output Sequence
        if (x_out_seq != nullptr) {
            for (int i = tid; i < dim; i += blockDim.x) x_out_seq[(b * seq_len + t) * dim + i] = s_x[i];
        }
        __syncthreads();
        
    } // End Time Loop
    
    // Write Final State
    for (int i = tid; i < dim; i += blockDim.x) {
        x_state[b * dim + i] = s_x[i];
        v_state[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* reg_loss,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, const float* dt_scales, const float* forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    const float* W_mix_x, const float* W_mix_v, 
    cudaStream_t stream
) {
    const int num_blocks = batch; 
    const int shared_bytes = (6 * dim + rank + 128) * sizeof(float) + 8 * sizeof(double);
    
    cudaMemsetAsync(reg_loss, 0, batch * sizeof(float), stream);
    
    recurrent_manifold_fused_kernel<<<num_blocks, BLOCK_SIZE, shared_bytes, stream>>>(
        x_state, v_state, forces, U_stack, W_stack, x_out_seq, reg_loss, W_mix_x, W_mix_v,
        batch, seq_len, dim, rank, num_layers, num_heads, dt, dt_scales, forget_rates,
        plasticity, sing_thresh, sing_strength
    );
}

