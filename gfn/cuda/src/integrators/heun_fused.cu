#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

__global__ void heun_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale,
    const int batch,
    const int dim,
    const int rank,
    const int steps
) {
    extern __shared__ float s_mem_f[];
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_x_pred = s_v + dim;
    float* s_v_pred = s_x_pred + dim;
    float* s_gamma = s_v_pred + dim;
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

    const float eff_dt = dt * dt_scale;

    for (int s = 0; s < steps; s++) {
        // Predictor: Euler step
        christoffel_device(s_v, U, W, s_gamma, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            float acc = f_val - s_gamma[i];
            s_v_pred[i] = s_v[i] + eff_dt * acc;
            s_x_pred[i] = s_x[i] + eff_dt * s_v[i];
        }
        __syncthreads();

        // Corrector: Average of initial and predicted accelerations
        christoffel_device(s_v_pred, U, W, s_gamma + dim, s_x_pred, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            float acc_init = f_val - s_gamma[i];
            float acc_pred = f_val - s_gamma[dim + i];
            s_v[i] = s_v[i] + 0.5f * eff_dt * (acc_init + acc_pred);
            s_x[i] = s_x[i] + 0.5f * eff_dt * (s_v[i] + s_v_pred[i]);
        }
        __syncthreads();
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_heun_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
) {
    int shared = (6 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    heun_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, x_new, v_new, dt, dt_scale, batch, dim, rank, steps
    );
}
