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
    __shared__ float s_h[MAX_RANK];
    __shared__ double s_E;
    __shared__ double s_P;
    __shared__ float s_M;

    const int b = blockIdx.x;
    if (b >= batch) return;
    
    christoffel_device(
        v + b * dim, U, W, gamma + b * dim, 
        (x != nullptr) ? (x + b * dim) : nullptr, V_w, 
        dim, rank, 
        plasticity, sing_thresh, sing_strength, use_active,
        s_h, &s_E, &s_P, &s_M
    );
}

extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
) {
    christoffel_fused_kernel<<<batch, BLOCK_SIZE, 0, stream>>>(
        v, U, W, gamma, x, V_w, batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active
    );
}


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
    extern __shared__ float s_mem[];
    float* s_h = s_mem;  // [rank]
    
    // All the rest as doubles for maximum precision
    double* s_double = (double*)(s_mem + rank + (rank % 2));
    double* s_grad_h = s_double;           // [rank] - KEEP AS DOUBLE
    double* s_E = s_grad_h + rank;
    double* s_P = s_E + 1;
    double* s_M = s_P + 1;
    double* s_dL_dM = s_M + 1;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    const float* v_b = v + b * dim;
    const float* gg_b = grad_gamma + b * dim;
    const float* x_b = (x) ? (x + b * dim) : nullptr;
    float* gv_b = grad_v + b * dim;

    // Initialize
    if (tid == 0) {
        *s_E = 0.0;
        *s_P = 0.0;
        *s_M = 0.0;
        *s_dL_dM = 0.0;
    }
    for (int r = tid; r < rank; r += blockDim.x) {
        s_h[r] = 0.0f;
        s_grad_h[r] = 0.0;
    }
    __syncthreads();

    // Recompute forward: h = U^T v
    for (int r = 0; r < rank; ++r) {
        float local_h = 0.0f;
        for (int i = tid; i < dim; i += blockDim.x) {
            local_h += v_b[i] * U[i * rank + r];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            local_h += __shfl_down_sync(0xffffffff, local_h, offset);
        }
        if (tid % 32 == 0) atomicAdd(&s_h[r], local_h);
    }
    
    // Active Inference
    if (use_active) {
        double local_E = 0.0, local_P = 0.0;
        for (int i = tid; i < dim; i += blockDim.x) {
            if (plasticity != 0.0f) {
                local_E += (double)v_b[i] * (double)v_b[i];
            }
            if (x_b != nullptr && V_w != nullptr) {
                local_P += (double)x_b[i] * (double)V_w[i];
            }
        }
        if (plasticity != 0.0f) atomicAdd(s_E, local_E);
        if (x_b != nullptr && V_w != nullptr) atomicAdd(s_P, local_P);
    }
    __syncthreads();

    // Multipliers
    double M_plast = 1.0, M_sing = 1.0;
    if (use_active) {
        if (plasticity != 0.0f) {
            double mean_E = *s_E / (double)dim;
            M_plast = 1.0 + (double)plasticity * tanh(mean_E);
        }
        if (x_b != nullptr && V_w != nullptr) {
            double pot = 1.0 / (1.0 + exp(-*s_P));
            if (pot > (double)sing_thresh) {
                M_sing = (double)sing_strength;
            }
        }
    }
    double M = M_plast * M_sing;

    // Norm and S
    double local_nsq = 0.0;
    for (int r = tid; r < rank; r += blockDim.x) {
        local_nsq += (double)s_h[r] * (double)s_h[r];
    }
    atomicAdd(s_M, local_nsq);
    __syncthreads();
    
    double norm_h = sqrt(*s_M);
    double S = 1.0 / (1.0 + norm_h);

    // Backward - accumulate grad_W and dL/dh
    double local_dL_dM = 0.0;
    for (int j = tid; j < dim; j += blockDim.x) {
        double gamma_static = 0.0;
        for (int r = 0; r < rank; r++) {
            gamma_static += (double)W[j * rank + r] * (double)s_h[r] * (double)s_h[r];
        }
        gamma_static *= S;
        
        double gg = (double)gg_b[j];
        // Clamp derivative
        double val_final = gamma_static * M;
        if (val_final <= -5.0 || val_final >= 5.0) {
            gg = 0.0;
        }
        
        local_dL_dM += gg * gamma_static;
        
        double gg_M = gg * M;
        for (int r = 0; r < rank; r++) {
            double Zr = (double)s_h[r] * (double)s_h[r] * S;
            atomicAdd(&grad_W[j * rank + r], (float)(gg_M * Zr));
            atomicAdd(&s_grad_h[r], gg_M * (double)W[j * rank + r]);  // DOUBLE ACCUMULATION
        }
    }
    if (use_active) atomicAdd(s_dL_dM, local_dL_dM);
    __syncthreads();

    // Apply S-modulation to grad_h
    if (tid == 0) *s_M = 0.0;
    __syncthreads();
    
    double local_C = 0.0;
    for (int r = tid; r < rank; r += blockDim.x) {
        local_C += s_grad_h[r] * (double)s_h[r] * (double)s_h[r] * S;
    }
    atomicAdd(s_M, local_C);
    __syncthreads();
    
    double C_val = *s_M;
    for (int r = tid; r < rank; r += blockDim.x) {
        double Gr = s_grad_h[r];
        double hr = (double)s_h[r];
        s_grad_h[r] = S * hr * (2.0 * Gr - C_val / (norm_h + 1e-12));
    }
    __syncthreads();

    // Final gradients
    double dL_dM_val = *s_dL_dM;
    double plast_grad_mult = 0.0;
    if (use_active && plasticity != 0.0f) {
        double mean_E = *s_E / (double)dim;
        double t = tanh(mean_E);
        plast_grad_mult = dL_dM_val * M_sing * (double)plasticity * (1.0 - t * t) * (2.0 / (double)dim);
    }

    for (int k = tid; k < dim; k += blockDim.x) {
        double vk = (double)v_b[k];
        double dv = plast_grad_mult * vk;
        for (int r = 0; r < rank; r++) {
            double dhr = s_grad_h[r];
            dv += dhr * (double)U[k * rank + r];
            atomicAdd(&grad_U[k * rank + r], (float)(dhr * vk));
        }
        gv_b[k] = (float)dv;
    }
}

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
) {
    int shared_floats = rank + (rank % 2);
    int shared = shared_floats * sizeof(float) + (rank + 4) * sizeof(double);
    christoffel_backward_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        grad_gamma, v, U, W, x, V_w, grad_v, grad_U, grad_W, grad_x, grad_V_w,
        batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active
    );
}
