
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

// Yoshida Coefficients
#define Y_W0 -1.7024143839193153f
#define Y_W1 1.3512071919596578f

__global__ void yoshida_fused_kernel(
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
    double* s_mem_d = (double*)(s_mem_f + 2 * dim + rank);

    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_h = s_v + dim;
    
    double* s_E = s_mem_d;
    double* s_P = s_E + 1;
    float* s_M = (float*)(s_P + 1);
    float* s_gamma = s_M + 1;
    
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
    
    float c1 = Y_W1 / 2.0f;
    float c2 = (Y_W0 + Y_W1) / 2.0f;
    float c3 = c2; 
    float c4 = c1;
    float d1 = Y_W1;
    float d2 = Y_W0;
    float d3 = Y_W1;
    
    for (int s = 0; s < steps; s++) {
        // === Substep 1 ===
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += c1 * eff_dt * s_v[i];
        __syncthreads();
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, s_h, s_E, s_P, s_M);
        __syncthreads(); 
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += d1 * eff_dt * (f_val - s_gamma[i]);
        }
        __syncthreads();
        
        // === Substep 2 ===
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += c2 * eff_dt * s_v[i];
        __syncthreads();
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += d2 * eff_dt * (f_val - s_gamma[i]);
        }
        __syncthreads();
        
         // === Substep 3 ===
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += c3 * eff_dt * s_v[i];
        __syncthreads();
        christoffel_device(s_v, U, W, s_gamma, s_x, V_w, dim, rank, plasticity, sing_thresh, sing_strength, use_active, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_v[i] += d3 * eff_dt * (f_val - s_gamma[i]);
        }
        __syncthreads();
        
        for (int i = tid; i < dim; i += blockDim.x) s_x[i] += c4 * eff_dt * s_v[i];
        __syncthreads();
    }
    
    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_yoshida_fused(
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
    int shared = (3 * dim + MAX_RANK + 1) * sizeof(float) + 2 * sizeof(double);
    yoshida_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, V_w, x_new, v_new, dt, dt_scale_scalar, dt_scale_tensor,
        batch, dim, rank, plasticity, sing_thresh, sing_strength, use_active, steps
    );
}
