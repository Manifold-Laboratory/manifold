#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

// Omelyan coefficients (optimized 2nd order)
#define OM_LAMBDA 0.1931833275037836f
#define OM_C1 OM_LAMBDA
#define OM_C2 ((1.0f - 2.0f * OM_LAMBDA) / 2.0f)
#define OM_C3 OM_C2
#define OM_C4 OM_C2
#define OM_C5 OM_LAMBDA
#define OM_D1 0.5f
#define OM_D2 (1.0f - OM_LAMBDA)
#define OM_D3 0.5f

__global__ void omelyan_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ V_w,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale_scalar,
    const float* __restrict__ dt_scale_tensor,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    const int steps
) {
    extern __shared__ float s_mem_f[];
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_gamma = s_v + dim;
    float* s_h = s_gamma + dim;
    
    double* s_mem_d = (double*)(s_h + rank + (rank % 2));
    double* s_E = s_mem_d;
    double* s_P = s_E + 1;
    float* s_M = (float*)(s_P + 1);
    
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;
    
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();
    
    float scale = (dt_scale_tensor != nullptr) ? dt_scale_tensor[b] : dt_scale_scalar;
    float eff_dt = dt * scale;
    
    for (int s = 0; s < steps; s++) {
        // Substep 1: x += λ * dt * v
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += OM_C1 * eff_dt * s_v[i];
        __syncthreads();
        
        // v += 0.5 * dt * a
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += OM_D1 * eff_dt * (f_val - s_gamma[i]);
        }
        __syncthreads();
        
        // Substep 2: x += (1-2λ)/2 * dt * v
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += OM_C2 * eff_dt * s_v[i];
        __syncthreads();
        
        // v += (1-λ) * dt * a
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += OM_D2 * eff_dt * (f_val - s_gamma[i]);
        }
        __syncthreads();
        
        // Substep 3: x += (1-2λ)/2 * dt * v
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += OM_C3 * eff_dt * s_v[i];
        __syncthreads();
        
        // v += 0.5 * dt * a
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += OM_D3 * eff_dt * (f_val - s_gamma[i]);
        }
        __syncthreads();
        
        // Substep 4 (final): x += λ * dt * v
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += OM_C5 * eff_dt * s_v[i];
        __syncthreads();
    }
    
    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_omelyan_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W, const float* V_w,
    float* x_new, float* v_new,
    float dt, float dt_scale_scalar,
    const float* dt_scale_tensor,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, 
    int steps,
    cudaStream_t stream
) {
    int shared = (3 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    omelyan_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, V_w, x_new, v_new, dt, dt_scale_scalar, dt_scale_tensor,
        batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active, steps
    );
}
