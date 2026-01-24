#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 512

/**
 * RECURRENT MANIFOLD BACKWARD KERNEL v5.0 (WITH MIXING GRADS)
 * ===========================================================
 * Handles Time-Major Backprop with:
 * 1. BPTT through Leapfrog (Symplectic Reversibility)
 * 2. Multi-Head Mixing Gradients (dL/dW_mix)
 * 3. Layer Iteration (L_out -> L_in)
 */

// Device function: Backward of Head Mixing
// y = x @ W^T
// dL/dx = dL/dy @ W
// dL/dW = (dL/dy)^T @ x
__device__ void head_mixing_backward_device(
    float* s_gx,          // [Dim] Gradients of output (dL/dy) -> Updated to dL/dx
    float* s_gv,          // [Dim]
    const float* s_x,     // [Dim] Input state x (reconstructed)
    const float* s_v,     // [Dim] Input state v
    const float* W_x,     // [Dim, Dim]
    const float* W_v,     // [Dim, Dim]
    float* grad_W_mix_x,  // [Dim, Dim] Global accumulation buffer
    float* grad_W_mix_v,  // [Dim, Dim] Global accumulation buffer
    float* s_temp_x,      // Scratch [Dim]
    float* s_temp_v,      // Scratch [Dim]
    int dim,
    int tid
) {
    __syncthreads();
    
    // 1. Compute dL/dW (Outer Product accumulation)
    for (int i = tid; i < dim * dim; i += blockDim.x) {
        int r = i / dim;
        int c = i % dim;
        // grad_W[r, c] += gy[r] * x[c]
        float val_x = s_gx[r] * s_x[c];
        float val_v = s_gv[r] * s_v[c];
        
        if (grad_W_mix_x) atomicAdd(&grad_W_mix_x[i], val_x);
        if (grad_W_mix_v) atomicAdd(&grad_W_mix_v[i], val_v);
    }
    __syncthreads();
    
    // 2. Compute dL/dx = dL/dy @ W
    for (int j = tid; j < dim; j += blockDim.x) {
        float sum_x = 0.0f;
        float sum_v = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_x += s_gx[i] * W_x[i * dim + j];
            sum_v += s_gv[i] * W_v[i * dim + j];
        }
        s_temp_x[j] = sum_x;
        s_temp_v[j] = sum_v;
    }
    __syncthreads();
    
    // Write back gradient
    for (int i = tid; i < dim; i += blockDim.x) {
        s_gx[i] = s_temp_x[i];
        s_gv[i] = s_temp_v[i];
    }
    __syncthreads();
}

// Device function to accumulate gradients for U and W
__device__ void christoffel_grads_device(
    const float* s_dGamma,  // [Dim] Gradient of Loss w.r.t Gamma
    const float* s_v,       // [Dim] Input velocity
    const float* s_h,       // [Rank] Pre-computed h = U^T v
    const float* U,         // [Dim, Rank]
    const float* W,         // [Dim, Rank]
    float* grad_U,          // [Dim, Rank] Global
    float* grad_W,          // [Dim, Rank] Global
    int dim,
    int rank,
    int tid,
    float* s_temp_float   // Shared mem for reduction (Float for atomic compatibility)
) {
    // 1. Recompute S (Normalization)
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += blockDim.x) local_nsq += s_h[r] * s_h[r];
    
    // Warp Reduction
    for (int offset = 16; offset > 0; offset /= 2) local_nsq += __shfl_down_sync(0xffffffff, local_nsq, offset);
    
    if (tid == 0) *s_temp_float = 0.0f;
    __syncthreads();
    
    if ((tid & 31) == 0) atomicAdd(s_temp_float, local_nsq);
    __syncthreads();
    
    float h_norm = sqrtf(*s_temp_float);
    float S = 1.0f / (1.0f + h_norm);
    
    // 2. Gradients w.r.t W
    // Gamma = W * (h^2) * S
    // dL/dW_ir = dGamma_i * (h_r^2 * S)
    for (int i = tid; i < dim * rank; i += blockDim.x) {
        int r = i % rank;
        int d = i / rank;
        float val = s_dGamma[d] * s_h[r] * s_h[r] * S;
        atomicAdd(&grad_W[i], val);
    }
    __syncthreads();
    
    // 3. Gradients w.r.t h (Approx: Ignore dS/dh)
    // dL/dh_r approx = sum_i (dGamma_i * W_ir) * 2 * h_r * S
    for (int r = tid; r < rank; r += blockDim.x) {
        float sum_dw = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_dw += s_dGamma[i] * W[i * rank + r];
        }
        float dL_dh = sum_dw * 2.0f * s_h[r] * S;
        
        // 4. Gradient w.r.t U
        // dL/dU_jr = dL/dh_r * v_j
        for (int j = 0; j < dim; j++) {
            atomicAdd(&grad_U[j * rank + r], dL_dh * s_v[j]);
        }
    }
    __syncthreads();
}

__global__ void recurrent_manifold_backward_kernel(
    const float* __restrict__ grad_x_seq,    
    const float* __restrict__ grad_x_final,  
    const float* __restrict__ grad_v_final,  
    const float* __restrict__ x_final,       
    const float* __restrict__ v_final,       
    const float* __restrict__ forces,        
    const float* __restrict__ U_stack,       
    const float* __restrict__ W_stack,       
    float* __restrict__ grad_x_init,         
    float* __restrict__ grad_v_init,         
    float* __restrict__ grad_forces,         
    float* __restrict__ grad_U,              
    float* __restrict__ grad_W,
    const float* __restrict__ W_mix_x,
    const float* __restrict__ W_mix_v,
    float* __restrict__ grad_W_mix_x,
    float* __restrict__ grad_W_mix_v,
    const int batch, 
    const int seq_len,
    const int dim,         
    const int rank,
    const int num_layers,
    const int num_heads,
    const float dt,
    const float* __restrict__ dt_scales,
    const float* __restrict__ forget_rates,
    float* __restrict__ grad_forget_rates,
    const float plasticity,
    const float sing_thresh,
    const float sing_strength
) {
    extern __shared__ float s_mem[];
    
    // Shared Memory Layout
    const int dim_per_head = dim / num_heads;
    const int head_rank = rank / num_heads;
    
    float* s_x = s_mem;                
    float* s_v = s_mem + dim;            
    float* s_gx = s_v + dim;           
    float* s_gv = s_gx + dim;          
    float* s_gamma = s_gv + dim;       
    float* s_temp1 = s_gamma + dim;    
    float* s_temp2 = s_temp1 + dim;    
    float* s_h = s_temp2 + dim;        
    
    float* s_v_half = s_temp1;
    double* s_double_base = (double*)(s_h + num_heads * head_rank + 32); 

    const int b = blockIdx.x; 
    const int tid = threadIdx.x;
    
    if (b >= batch) return;

    const float depth_scale = 1.0f / sqrtf((float)num_layers);
    
    // 1. Initialize State
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_final[b * dim + i];
        s_v[i] = v_final[b * dim + i];
        s_gx[i] = grad_x_final ? grad_x_final[b * dim + i] : 0.0f;
        s_gv[i] = grad_v_final ? grad_v_final[b * dim + i] : 0.0f;
    }
    __syncthreads();
    
    // 2. Iterate Backwards through Time
    for (int t = seq_len - 1; t >= 0; t--) {
        
        if (grad_x_seq != nullptr) {
             for (int i = tid; i < dim; i += blockDim.x) s_gx[i] += grad_x_seq[(b * seq_len + t) * dim + i];
             __syncthreads();
        }
        
        for (int l = num_layers - 1; l >= 0; l--) {
            // LEVEL 12: ADJOINT NORMALIZATION
            // Differentiate through v = v_raw / |v_raw|
            // dL/dv_raw = (dL/dv - (dL/dv . v) * v) / |v_raw|
            for (int h = num_heads - 1; h >= 0; h--) {
                int head_offset = h * dim_per_head;
                float* s_gv_h = s_gv + head_offset;
                float* s_v_h = s_v + head_offset;

                float local_dot = 0.0f;
                float local_v_nsq = 0.0f;
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    local_dot += s_gv_h[i] * s_v_h[i];
                    local_v_nsq += s_v_h[i] * s_v_h[i];
                }
                
                if (tid == 0) {
                    s_double_base[2] = 0.0;
                    s_double_base[3] = 0.0;
                }
                __syncthreads();
                
                atomicAdd((double*)&s_double_base[2], (double)local_dot);
                atomicAdd((double*)&s_double_base[3], (double)local_v_nsq);
                __syncthreads();
                
                float dot_val = (float)s_double_base[2];
                float v_nsq = (float)s_double_base[3] + 1e-6f;
                float v_norm = sqrtf(v_nsq);
                
                // Project gradient onto the sphere tangent plane and scale
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_gv_h[i] = (s_gv_h[i] - dot_val * s_v_h[i]) / v_norm;
                }
                __syncthreads();
            }

            // Mixing
            if (num_heads > 1 && W_mix_x != nullptr && l < num_layers - 1) {
                head_mixing_backward_device(s_gx, s_gv, s_x, s_v, W_mix_x, W_mix_v, grad_W_mix_x, grad_W_mix_v, s_temp1, s_temp2, dim, tid);
            }
            
            // Backward integrator loop
            for (int h = num_heads - 1; h >= 0; h--) {
                const int head_offset = h * dim_per_head;
                const long long layer_param_idx = l * num_heads + h;
                
                const float h_dt_scale = dt_scales ? dt_scales[h] : 1.0f;
                const float h_mu = forget_rates ? forget_rates[h] : 0.05f;
                const float eff_dt = dt * h_dt_scale * depth_scale;
                
                float* grad_U_h = grad_U + (layer_param_idx * dim_per_head * head_rank);
                float* grad_W_h = grad_W + (layer_param_idx * dim_per_head * head_rank);
                
                const float* U_l = U_stack + (layer_param_idx * dim_per_head * head_rank);
                const float* W_l = W_stack + (layer_param_idx * dim_per_head * head_rank);
                const float* f_t = &forces[(b * seq_len + t) * dim + head_offset];
                
                float* s_v_h = s_v + head_offset;
                float* s_x_h = s_x + head_offset;
                float* s_gx_h = s_gx + head_offset;
                float* s_gv_h = s_gv + head_offset;
                float* s_gamma_h = s_gamma + head_offset;
                float* s_v_half_h = s_v_half + head_offset;
                float* s_h_scr = s_h + h * head_rank; 
                
                double* s_E_scr = s_double_base; 
                double* s_P_scr = s_E_scr+1; 
                float* s_M_scr = (float*)(s_P_scr+1);
                
                // --- STEP 1: RESTORE MIDPOINT (REVERSE LEAPFROG) ---
                // christoffel_device(s_v_h, U_l, W_l, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, s_h_scr, s_E_scr, s_P_scr, s_M_scr);
                // __syncthreads();
                // v_h = v_mid + 0.5 dt (F - G - mu*v) => v_mid = v_h / (1 + 0.5 dt mu) - ... simplified damping approx
                // We use standard damping: v_new = v_old * (1 - mu*dt) + acc*dt
                // For simplicity in reverse: v_mid approx v_h * (1 + 0.5*eff_dt*h_mu) - 0.5*eff_dt*(F-G)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_v_half_h[i] = s_v_h[i] * (1.0f + 0.5f * eff_dt * h_mu) - 0.5f * eff_dt * (f_t[i] - s_gamma_h[i]);
                }
                __syncthreads();
                
                // x_0 = x_1 - dt v_mid
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_x_h[i] -= eff_dt * s_v_half_h[i];
                __syncthreads();
                
                // --- STEP 2: RESTORE INITIAL v0 ---
                // christoffel_device(s_v_half_h, U_l, W_l, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, s_h_scr, s_E_scr, s_P_scr, s_M_scr);
                // __syncthreads();
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_v_h[i] = s_v_half_h[i] * (1.0f + 0.5f * eff_dt * h_mu) - 0.5f * eff_dt * (f_t[i] - s_gamma_h[i]);
                }
                __syncthreads();

                // --- ADJOINT PROPAGATION ---
                // We currently have x0, v0 in s_x_h, s_v_h.
                // We have v_mid in s_v_half_h.
                
                // 1. Advance x to x1 for K2 Backward
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_x_h[i] += eff_dt * s_v_half_h[i];
                }
                __syncthreads();
                
                // Recompute K2 (using x1)
                // christoffel_device(s_v_half_h, U_l, W_l, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, s_h_scr, s_E_scr, s_P_scr, s_M_scr);
                // __syncthreads();
                
                // BWD K2 to Spring k
                float local_gk = 0.0f;
                const float half_dt_val = 0.5f * eff_dt;
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float gv_val = s_gv_h[i];
                    // dL/dmu = -dL/dv * v * 0.5 * dt (approx)
                    local_gk += gv_val * (-0.5f * eff_dt * s_v_h[i]);
                }
                // Reduce and Add (Lane 0 of each warp adds)
                float warp_sum_mu2 = warpReduceSum(local_gk);
                if ((tid & 31) == 0) {
                    if (grad_forget_rates) atomicAdd(&grad_forget_rates[h], warp_sum_mu2);
                }
                
                // Backprop Friction through Velocity: dL/dv_old = dL/dv_new * (1 - 0.5*dt*mu)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_gv_h[i] *= (1.0f - 0.5f * eff_dt * h_mu);
                }
                __syncthreads();

                // (Continue original gradient logic for U, W, forces...)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float gv = s_gv_h[i];
                    if (grad_forces) atomicAdd(&grad_forces[(b*seq_len+t)*dim+head_offset+i], gv * 0.5f * eff_dt);
                    s_gamma_h[i] = -gv * 0.5f * eff_dt; 
                }
                __syncthreads();
                christoffel_grads_device(s_gamma_h, s_v_half_h, s_h_scr, U_l, W_l, grad_U_h, grad_W_h, dim_per_head, head_rank, tid, (float*)s_E_scr);
                {
                    float local_nsq = 0.0f;
                    for (int r = tid; r < head_rank; r += blockDim.x) local_nsq += s_h_scr[r] * s_h_scr[r];
                    // FIX: warpReduceSum called by ALL threads
                    float reduced_nsq = warpReduceSum(local_nsq);
                    if (tid == 0) *((float*)s_E_scr) = reduced_nsq;
                    __syncthreads();
                    float S = 1.0f / (1.0f + sqrtf(*((float*)s_E_scr)) + 1e-6f);
                    christoffel_v_backward_device(s_gamma_h, U_l, W_l, s_h_scr, s_v_half_h, s_gv_h, dim_per_head, head_rank, plasticity, false, S, 1.0f, (float*)s_P_scr);
                }
                __syncthreads();
                
                // Propagate dL/dx1 to dL/dv_mid (Drift)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_gv_h[i] += s_gx_h[i] * eff_dt;
                }
                __syncthreads();
                
                // Sping Gradient Interaction with Drift?
                // x1 = x0 + dt * v_mid
                // v_mid = v0 + 0.5*dt*(f - g1 - k*x0)
                // dx1/dk = dt * dv_mid/dk = dt * (-0.5 * dt * x0)
                // dL/dk += dL/dx1 * dx1/dk
                
                // Note: s_x_h here is x1. x0 = x1 - dt*v_mid.
                // We simplify by waiting until x is restored to x0 or computing x0 locally.
                // Let's compute approx term later or now?
                // It relies on x0. Let's do it after retracting x.
                
                // 2. Retract x to x0 for K1 Backward
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_x_h[i] -= eff_dt * s_v_half_h[i];
                }
                __syncthreads();
                
                // Add Spring Gradient K1 Part (-0.5 * dt * x0)
                // AND Drift part (which depends on x0)
                float local_gk_mu1 = 0.0f;
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    // Friction dL/dmu block
                    local_gk_mu1 += s_gv_h[i] * (-0.5f * eff_dt * s_v_half_h[i]);
                    s_gv_h[i] *= (1.0f - 0.5f * eff_dt * h_mu);
                }
                float warp_sum_mu1 = warpReduceSum(local_gk_mu1);
                if ((tid & 31) == 0) {
                    if (grad_forget_rates) atomicAdd(&grad_forget_rates[h], warp_sum_mu1); 
                }
                __syncthreads();
                
                // BACK TO K1 (using x0)
                // K1 Recompute
                // christoffel_device(s_v_h, U_l, W_l, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, s_h_scr, s_E_scr, s_P_scr, s_M_scr);
                // __syncthreads();
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float gv = s_gv_h[i];
                    if (grad_forces) atomicAdd(&grad_forces[(b*seq_len+t)*dim+head_offset+i], gv * 0.5f * eff_dt);
                    s_gamma_h[i] = -gv * 0.5f * eff_dt; // dL/dG1
                }
                __syncthreads();
                christoffel_grads_device(s_gamma_h, s_v_h, s_h_scr, U_l, W_l, grad_U_h, grad_W_h, dim_per_head, head_rank, tid, (float*)s_E_scr);
                {
                    float local_nsq = 0.0f;
                    for (int r = tid; r < head_rank; r += blockDim.x) local_nsq += s_h_scr[r] * s_h_scr[r];
                    float reduced_nsq = warpReduceSum(local_nsq);
                    if (tid == 0) *((float*)s_E_scr) = reduced_nsq;
                    __syncthreads();
                    float S = 1.0f / (1.0f + sqrtf(*((float*)s_E_scr)) + 1e-6f);
                    christoffel_v_backward_device(s_gamma_h, U_l, W_l, s_h_scr, s_v_h, s_gv_h, dim_per_head, head_rank, plasticity, false, S, 1.0f, (float*)s_P_scr);
                }
                __syncthreads();
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_gv_h[i] += s_gx_h[i] * eff_dt;
                }
                __syncthreads();
                {
                    float local_nsq = 0.0f;
                    for (int r = tid; r < head_rank; r += blockDim.x) local_nsq += s_h_scr[r] * s_h_scr[r];
                    float reduced_nsq = warpReduceSum(local_nsq);
                    if (tid == 0) *((float*)s_E_scr) = reduced_nsq;
                    __syncthreads();
                    float S = 1.0f / (1.0f + sqrtf(*((float*)s_E_scr)) + 1e-6f);
                    christoffel_v_backward_device(s_gamma_h, U_l, W_l, s_h_scr, s_v_h, s_gv_h, dim_per_head, head_rank, plasticity, false, S, 1.0f, (float*)s_P_scr);
                }
                __syncthreads();
                
                // Clamp here leads to Adjoint instability. 
                // Relying on Hamiltonian stabilization.
                __syncthreads();
            }
        }
    }
    
    // 3. Write Initial Gradients
    for (int i = tid; i < dim; i += blockDim.x) {
        grad_x_init[b * dim + i] = s_gx[i];
        grad_v_init[b * dim + i] = s_gv[i];
    }
}

extern "C" void launch_recurrent_manifold_backward(
    const float* grad_x_seq, const float* grad_x_final, const float* grad_v_final,
    const float* x_final, const float* v_final,
    const float* forces, const float* U_stack, const float* W_stack,
    float* grad_x_init, float* grad_v_init, float* grad_forces,
    float* grad_U, float* grad_W,
    const float* W_mix_x, const float* W_mix_v,
    float* grad_W_mix_x, float* grad_W_mix_v,
    int batch_total, int seq_len, int dim, int rank, int num_layers, int num_heads,
    float dt, const float* dt_scales, const float* forget_rates, float* grad_forget_rates,
    float plasticity, float sing_thresh, float sing_strength,
    cudaStream_t stream
) {
    const int shared_bytes = (7 * dim + rank + 128) * sizeof(float) + 8 * sizeof(double);
    const int num_blocks = batch_total;
    
    recurrent_manifold_backward_kernel<<<num_blocks, BLOCK_SIZE, shared_bytes, stream>>>(
        grad_x_seq, grad_x_final, grad_v_final,
        x_final, v_final,
        forces, U_stack, W_stack,
        grad_x_init, grad_v_init, grad_forces,
        grad_U, grad_W,
        W_mix_x, W_mix_v, grad_W_mix_x, grad_W_mix_v,
        batch_total, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales, forget_rates, grad_forget_rates,
        plasticity, sing_thresh, sing_strength
    );
}
