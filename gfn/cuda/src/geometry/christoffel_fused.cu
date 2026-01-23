#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

__global__ void christoffel_fused_kernel(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active
) {
    __shared__ float s_U[MAX_RANK];
    __shared__ float s_energy_sum;
    __shared__ float s_potential_sum;
    __shared__ float s_final_mult;

    const int b = blockIdx.x;
    if (b >= batch) return;
    
    christoffel_device(
        v + b * dim, U, W, gamma + b * dim, 
        (x != nullptr) ? (x + b * dim) : nullptr, V_w, 
        dim, rank, 
        plasticity, sing_thresh, sing_strength, use_active,
        s_U, &s_energy_sum, &s_potential_sum, &s_final_mult
    );
}

// Raw pointer launcher
extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
) {
    const int threads = BLOCK_SIZE;
    const int blocks = batch;
    
    christoffel_fused_kernel<<<blocks, threads, 0, stream>>>(
        v, U, W, gamma, x, V_w,
        batch, dim, rank,
        plasticity, sing_thresh, sing_strength,
        use_active
    );
}

// --------------------------------------------------------------------------------
// BACKWARD KERNEL
// --------------------------------------------------------------------------------

__global__ void christoffel_backward_kernel(
    const float* __restrict__ grad_gamma,
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    float* __restrict__ grad_v,
    float* __restrict__ grad_U,
    float* __restrict__ grad_W,
    float* __restrict__ grad_x,
    float* __restrict__ grad_V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active
) {
    // Shared Accumulators
    extern __shared__ float s_mem[];
    float* s_h = s_mem;             // [rank] Projections
    float* s_grad_h = s_mem + rank; // [rank] dL/dh
    
    // Active Inference Shared
    float* s_E = s_grad_h + rank;   // [1] Energy
    float* s_P = s_E + 1;           // [1] Potential
    float* s_M = s_P + 1;           // [1] M
    float* s_dL_dM = s_M + 1;       // [1] dL/dM
    
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;
    
    // Offsets
    const float* v_b = v + b * dim;
    const float* gg_b = grad_gamma + b * dim;
    const float* x_b = (x) ? (x + b * dim) : nullptr;
    float* gv_b = grad_v + b * dim;
    float* gx_b = (grad_x) ? (grad_x + b * dim) : nullptr;
    
    // 1. Init Shared
    if (tid < rank) {
        s_h[tid] = 0.0f;
        s_grad_h[tid] = 0.0f;
    }
    if (tid == 0) {
        *s_E = 0.0f;
        *s_P = 0.0f;
        *s_M = 1.0f;
        *s_dL_dM = 0.0f;
    }
    __syncthreads();
    
    // 2. Recompute Forward State (h and Active Factors)
    // a) h[r]
    for (int r = 0; r < rank; r++) {
         float p = 0.0f;
         for (int i = tid; i < dim; i += blockDim.x) {
             p += v_b[i] * U[i * rank + r];
         }
         atomicAdd(&s_h[r], p);
    }
    
    // b) Active Factors
    if (use_active) {
        bool calc_plast = (plasticity != 0.0f);
        bool calc_sing = (x_b != nullptr && V_w != nullptr);
        float local_E = 0.0f;
        float local_P = 0.0f;
        
        for (int i = tid; i < dim; i += blockDim.x) {
             if (calc_plast) local_E += v_b[i] * v_b[i];
             if (calc_sing) local_P += x_b[i] * V_w[i];
        }
        if (calc_plast) atomicAdd(s_E, local_E);
        if (calc_sing) atomicAdd(s_P, local_P);
    }
    __syncthreads();
    
    // 3. Compute M
    if (tid == 0 && use_active) {
        float m_plast = 1.0f;
        float m_sing = 1.0f;
        
        if (plasticity != 0.0f) {
            float mean_E = *s_E / (float)dim;
            m_plast = 1.0f + plasticity * tanh(mean_E);
        }
        if (x_b != nullptr && V_w != nullptr) {
            float pot = 1.0f / (1.0f + expf(-*s_P));
            if (pot > sing_thresh) m_sing = sing_strength;
        }
        *s_M = m_plast * m_sing;
    }
    __syncthreads();
    
    float M = use_active ? *s_M : 1.0f;
    
    // 4. Compute Gradients w.r.t W and h, and collect dL/dM
    
    // 4. Compute Gradients w.r.t W and Projected Gradient G_r
    
    float local_dL_dM = 0.0f;
    
    // First Pass: Accumulate G_r (Raw Projected Gradient) and dL/dM components
    // reusing s_grad_h for G_r temporarily
    if (tid < rank) s_grad_h[tid] = 0.0f;
    __syncthreads();
    
    for (int j = tid; j < dim; j += blockDim.x) {
        float gg = gg_b[j];
        
        // Reconstruct Gamma_static[j] for dL/dM check
        // NOTE: This reconstruction must match Forward exactly (including scale!)
        // But we don't have 'scale' handy easily? 
        // We can recompute norm(s_h).
        // Let's compute norm once before this loop.
        
        // ... (This architecture is tricky safely. Let's compute Norm first).
        // Moving reconstruction inside loop is inefficient.
        
        // Minimal logic: 
        // dL/dGamma_raw = gg
        // Check clamp? We need gamma_val.
        // It's expensive to recompute gamma here.
        // Assumption: If user didn't save gamma, we recompute cost.
        // But for now, let's assume clamp logic is handled or we ignore it for stability?
        // Wait, "missing clamp derivative" was a bug I fixed.
        // I need to keep it.
        
        // Let's compute Norm scale first.
        
    }
    
    // START REWRITE
    // A. Compute Norm S
    // Use s_M as temp buffer (M is cached in register 'M')
    // CRITICAL: Do NOT overwrite s_P (Potential) or s_E (Energy) needed later!
    if (tid == 0) *s_M = 0.0f; 
    __syncthreads();
    
    if (tid < rank) {
        float h_val = s_h[tid];
        atomicAdd(s_M, h_val * h_val);
    }
    __syncthreads();
    
    float norm_h = sqrtf(*s_M);
    float S = 1.0f / (1.0f + norm_h);
    
    // B. Accumulate G_r (Projected Gradient)
    // dL/d(W * Z) => G_r = Sum_j (gg_j * M * W_jr)
    
    for (int j = tid; j < dim; j += blockDim.x) {
        float gg = gg_b[j];
        
        // Recompute Gamma_j to check Clamp
        float gamma_recon = 0.0f;
        for (int r = 0; r < rank; r++) {
            float hr = s_h[r];
            gamma_recon += W[j * rank + r] * hr * hr;
        }
        gamma_recon *= S; // Apply Scale
        
        float val_raw = gamma_recon * M;
        
        // Clamp Derivative
        if (val_raw <= -5.0f || val_raw >= 5.0f) {
            gg = 0.0f; 
        }
        
        // dL/dM Accumulation
        local_dL_dM += gg * gamma_recon; 
        
        // dL/dW Accumulation
        // Gamma_j = M * Sum W_jr * Z_r
        // dL/dW_jr = gg * M * Z_r
        // Z_r = h_r^2 * S
        float gg_st = gg * M;
        for (int r = 0; r < rank; r++) {
            float hr = s_h[r];
            float Zr = hr * hr * S;
            atomicAdd(&grad_W[j * rank + r], gg_st * Zr);
            
            // Accumulate G_r
            atomicAdd(&s_grad_h[r], gg_st * W[j * rank + r]);
        }
    }
    
    if (use_active) atomicAdd(s_dL_dM, local_dL_dM);
    __syncthreads();
    
    // C. Convert G_r to dL/dh_r (dL/dP_r)
    // Formula: dL/dh_r = S * h_r * (2 * G_r - C/N)
    // C = Sum(G_k * Z_k)
    // Z_k = h_k^2 * S
    
    // Compute C (Reduction)
    if (tid == 0) *s_M = 0.0f; // Reuse M for C
    __syncthreads();
    
    if (tid < rank) {
        float Gr = s_grad_h[tid];
        float hr = s_h[tid];
        float Zr = hr * hr * S;
        atomicAdd(s_M, Gr * Zr);
    }
    __syncthreads();
    
    float C_val = *s_M;
    
    // Final dL/dh update
    if (tid < rank) {
        float Gr = s_grad_h[tid];
        float hr = s_h[tid];
        float term = 2.0f * Gr - C_val / (norm_h + 1e-6f);
        s_grad_h[tid] = S * hr * term;
    }
    __syncthreads();
    
    // 5. Gradients w.r.t v, U, x, V_w
    // a) From Static part (via h)
    for (int k = tid; k < dim; k += blockDim.x) {
        float vk = v_b[k];
        
        // dL/dv_k = Sum_r (dL/dh_r * U_kr)
        float dL_dv_static = 0.0f;
        for (int r = 0; r < rank; r++) {
            float dhr = s_grad_h[r];
            dL_dv_static += dhr * U[k * rank + r];
            
            // dL/dU_kr = dL/dh_r * v_k
            atomicAdd(&grad_U[k * rank + r], dhr * vk);
        }
        
        // Singularity gradients are ZERO for Hard Threshold (step function has no derivative)
        // Only accumulate the static curvature gradient
        gv_b[k] = dL_dv_static;
    }
}

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active,
    cudaStream_t stream
) {
    int shared_bytes = (2 * rank + 5) * sizeof(float); // +5 for safetey
    christoffel_backward_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        grad_gamma, v, U, W, x, V_w,
        grad_v, grad_U, grad_W, grad_x, grad_V_w,
        batch, dim, rank,
        plasticity, sing_thresh, sing_strength, use_active
    );
}
