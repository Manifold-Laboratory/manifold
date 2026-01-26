
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

__global__ void leapfrog_backward_kernel(
    const float* __restrict__ grad_x_new,
    const float* __restrict__ grad_v_new,
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ grad_x,
    float* __restrict__ grad_v,
    float* __restrict__ grad_f,
    float* __restrict__ grad_U,
    float* __restrict__ grad_W,
    const int batch,
    const int dim,
    const int rank,
    float dt,
    float dt_scale,
    int steps,
    int topology,
    float R_val,
    float r_val
) {
    extern __shared__ float s_mem[];
    float* s_x = s_mem;
    float* s_v = s_x + dim;
    float* s_v_half = s_v + dim;
    float* s_gamma1 = s_v_half + dim;
    float* s_gamma2 = s_gamma1 + dim;
    float* s_h = s_gamma2 + dim;
    
    double* s_double = (double*)(s_h + rank + (rank % 2));
    double* s_E = s_double;
    double* s_P = s_E + 1;
    float* s_M_f = (float*)(s_P + 1);
    
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;
    
    const float eff_dt = dt * dt_scale;
    for (int i = tid; i < dim; i += blockDim.x) { s_x[i] = x[b * dim + i]; s_v[i] = v[b * dim + i]; grad_x[b * dim + i] = 0.0f; grad_v[b * dim + i] = 0.0f; if (grad_f) grad_f[b * dim + i] = 0.0f; }
    __syncthreads();
    
    for (int step = 0; step < steps; step++) {
        christoffel_device(s_v, U, W, s_gamma1, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, topology, s_h, s_E, s_P, s_M_f, R_val, r_val);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) { s_v_half[i] = s_v[i] + 0.5f * eff_dt * ((f?f[b*dim+i]:0.0f) - s_gamma1[i]); s_x[i] += eff_dt * s_v_half[i]; }
        __syncthreads();
        christoffel_device(s_v_half, U, W, s_gamma2, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, topology, s_h, s_E, s_P, s_M_f, R_val, r_val);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) s_v[i] = s_v_half[i] + 0.5f * eff_dt * ((f?f[b*dim+i]:0.0f) - s_gamma2[i]);
        __syncthreads();
    }
    
    // ADJOINT REVERSAL
    float* s_gx = s_gamma1, *s_gv = s_gamma2;
    for (int i = tid; i < dim; i += blockDim.x) { s_gx[i] = grad_x_new[b * dim + i]; s_gv[i] = grad_v_new[b * dim + i]; }
    __syncthreads();
    
    const float half_dt = 0.5f * eff_dt;

    for (int step = steps - 1; step >= 0; step--) {
        for (int i = tid; i < dim; i += blockDim.x) {
            float gvn = s_gv[i];
            
            // 1. Reverse Kick 2
            if (grad_f) atomicAdd(&grad_f[b*dim+i], gvn * half_dt);
            // dL/dGamma2 = -gvn * half_dt (handled by Christoffel backward)
            
            // 2. Reverse Drift
            s_gx[i] = s_gx[i]; // No x gradient update needed for drift v contribution? 
            // x_new = x_old + dt*v -> dL/dv = dL/dx * dt
            s_gv[i] += s_gx[i] * eff_dt; 
            
            // 3. Reverse Kick 1
            if (grad_f) atomicAdd(&grad_f[b*dim+i], s_gv[i] * half_dt);
        }
        __syncthreads();
    }
    for (int i = tid; i < dim; i += blockDim.x) { atomicAdd(&grad_x[b * dim + i], s_gx[i]); atomicAdd(&grad_v[b * dim + i], s_gv[i]); }
}

extern "C" void launch_leapfrog_backward(
    const float* grad_x_new, const float* grad_v_new, const float* x, const float* v, const float* f, const float* U, const float* W,
    float* grad_x, float* grad_v, float* grad_f, float* grad_U, float* grad_W,
    int batch, int dim, int rank, float dt, float dt_scale, int steps, int topology, float R_val, float r_val, cudaStream_t stream
) {
    int shared = (5 * dim + rank + 16) * sizeof(float) + (rank + 4) * sizeof(double);
    leapfrog_backward_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        grad_x_new, grad_v_new, x, v, f, U, W, grad_x, grad_v, grad_f, grad_U, grad_W, batch, dim, rank, dt, dt_scale, steps, topology, R_val, r_val
    );
}
