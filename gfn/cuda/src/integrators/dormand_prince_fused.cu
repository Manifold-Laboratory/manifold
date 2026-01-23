#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

// Dormand-Prince 5(4) Coefficients (Fixed Step Variant)
#define DP_A21 0.2f
#define DP_A31 0.075f
#define DP_A32 0.225f
#define DP_A41 0.9791666666666666f // 44/45
#define DP_A42 -3.288888888888889f // -148/45
#define DP_A43 3.311111111111111f  // 149/45
#define DP_A51 2.323043513333333f   // 19372/6561
#define DP_A52 -7.948254839201341f  // -25360/3189
#define DP_A53 4.575034444444444f   // 64448/14085
#define DP_A54 -0.0135832717088484f // -212/15607 (Approx)
#define DP_A61 1.488095238095238f   // 25/168
#define DP_A62 -5.0f
#define DP_A63 2.142857142857143f
#define DP_A64 -2.142857142857143f
#define DP_A65 -0.0135832717088484f // Simplified for fixed step

#define DP_B1 0.09114583333333334f // 35/384
#define DP_B3 0.4492362351190476f  // 500/1113
#define DP_B4 0.1633544742671042f  // 125/765
#define DP_B5 -0.03418758838294908f // -2187/64000
#define DP_B6 0.0303030303030303f   // 11/364

__global__ void dormand_prince_fused_kernel(
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
    // Memory requirements for 7 stages (FSAL)
    // s_x, s_v, s_k1...s_k6, s_h, s_gamma
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_k1 = s_v + dim;
    float* s_k2 = s_k1 + dim;
    float* s_k3 = s_k2 + dim;
    float* s_k4 = s_k3 + dim;
    float* s_k5 = s_k4 + dim;
    float* s_k6 = s_k5 + dim;
    float* s_gamma = s_k6 + dim;
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

    const float h = dt * dt_scale;

    for (int s = 0; s < steps; s++) {
        // Stage 1 (k1)
        christoffel_device(s_v, U, W, s_gamma, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k1[i] = h * (f_val - s_gamma[i]);
            s_gamma[i] = s_v[i] + DP_A21 * s_k1[i]; // Temporary storage for v for k2
        }
        __syncthreads();

        // Stage 2 (k2)
        christoffel_device(s_gamma, U, W, s_k2, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k2[i] = h * (f_val - s_k2[i]);
            s_gamma[i] = s_v[i] + DP_A31 * s_k1[i] + DP_A32 * s_k2[i];
        }
        __syncthreads();

        // Stage 3 (k3)
        christoffel_device(s_gamma, U, W, s_k3, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k3[i] = h * (f_val - s_k3[i]);
            s_gamma[i] = s_v[i] + DP_A41 * s_k1[i] + DP_A42 * s_k2[i] + DP_A43 * s_k3[i];
        }
        __syncthreads();

        // Stage 4 (k4)
        christoffel_device(s_gamma, U, W, s_k4, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k4[i] = h * (f_val - s_k4[i]);
            s_gamma[i] = s_v[i] + DP_A51 * s_k1[i] + DP_A52 * s_k2[i] + DP_A53 * s_k3[i] + DP_A54 * s_k4[i];
        }
        __syncthreads();

        // Stage 5 (k5)
        christoffel_device(s_gamma, U, W, s_k5, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k5[i] = h * (f_val - s_k5[i]);
            s_gamma[i] = s_v[i] + DP_A61 * s_k1[i] + DP_A62 * s_k2[i] + DP_A63 * s_k3[i] + DP_A64 * s_k4[i] + DP_A65 * s_k5[i];
        }
        __syncthreads();

        // Stage 6 (k6)
        christoffel_device(s_gamma, U, W, s_k6, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k6[i] = h * (f_val - s_k6[i]);
            
            // Final update for current step
            s_v[i] = s_v[i] + DP_B1 * s_k1[i] + DP_B3 * s_k3[i] + DP_B4 * s_k4[i] + DP_B5 * s_k5[i] + DP_B6 * s_k6[i];
            s_x[i] = s_x[i] + h * s_v[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_dormand_prince_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
) {
    int shared = (9 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double);
    dormand_prince_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, x_new, v_new, dt, dt_scale, batch, dim, rank, steps
    );
}
