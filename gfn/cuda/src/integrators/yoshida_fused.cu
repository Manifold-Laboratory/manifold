
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
    const float* __restrict__ V_w, // For active inference
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale_scalar,
    const float* __restrict__ dt_scale_tensor, // [batch] or nullptr
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active
) {
    // Shared Memory Layout:
    // ...
    extern __shared__ float s_mem[];
    
    // Pointers
    float* s_x = s_mem;                // [dim]
    float* s_v = s_x + dim;            // [dim]
    float* s_U = s_v + dim;            // [MAX_RANK]
    float* s_energy = s_U + MAX_RANK;  // [1]
    float* s_potential = s_energy + 1; // [1]
    float* s_mult = s_potential + 1;   // [1]
    float* s_gamma = s_mult + 1;       // [dim]
    
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (b >= batch) return;
    
    // 0. Load State
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();
    
    // Determine Effective DT
    float scale = (dt_scale_tensor != nullptr) ? dt_scale_tensor[b] : dt_scale_scalar;
    float eff_dt = dt * scale;
    
    float c1 = Y_W1 / 2.0f;
    float c2 = (Y_W0 + Y_W1) / 2.0f;
    float c3 = c2; 
    float c4 = c1;
    float d1 = Y_W1;
    float d2 = Y_W0;
    float d3 = Y_W1;
    
    // === Substep 1 ===
    // x1 = x + c1 * dt * v
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] += c1 * eff_dt * s_v[i];
    }
    __syncthreads();
    
    // gamma1 = Christoffel(v) -- Note: depends on x1?
    // gfn code: self.christoffel(v, x1). Yes.
    // We pass s_v and s_x (pointers to shared memory!)
    christoffel_device(
        s_v, U, W, s_gamma, s_x, V_w,
        dim, rank, plasticity, sing_thresh, sing_strength, use_active,
        s_U, s_energy, s_potential, s_mult
    );
    __syncthreads(); // Wait for s_gamma
    
    // v1 = v + d1 * dt * (f - gamma1)
    // Assuming 'f' is constant external force? Yes.
    for (int i = tid; i < dim; i += blockDim.x) {
        float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
        s_v[i] += d1 * eff_dt * (f_val - s_gamma[i]);
    }
    __syncthreads();
    
    // === Substep 2 ===
    // x2 = x1 + c2 * dt * v1
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] += c2 * eff_dt * s_v[i];
    }
    __syncthreads();
    
    // gamma2 = Christoffel(v1, x2)
    christoffel_device(
        s_v, U, W, s_gamma, s_x, V_w,
        dim, rank, plasticity, sing_thresh, sing_strength, use_active,
        s_U, s_energy, s_potential, s_mult
    );
    __syncthreads();
    
    // v2 = v1 + d2 * dt * (f - gamma2)
    for (int i = tid; i < dim; i += blockDim.x) {
        float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
        s_v[i] += d2 * eff_dt * (f_val - s_gamma[i]);
    }
    __syncthreads();
    
     // === Substep 3 ===
    // x3 = x2 + c3 * dt * v2
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] += c3 * eff_dt * s_v[i];
    }
    __syncthreads();
    
    // gamma3 = Christoffel(v2, x3)
    christoffel_device(
        s_v, U, W, s_gamma, s_x, V_w,
        dim, rank, plasticity, sing_thresh, sing_strength, use_active,
        s_U, s_energy, s_potential, s_mult
    );
    __syncthreads();
    
    // v3 = v2 + d3 * dt * (f - gamma3)
    for (int i = tid; i < dim; i += blockDim.x) {
        float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
        s_v[i] += d3 * eff_dt * (f_val - s_gamma[i]);
    }
    __syncthreads();
    
    // === Final Position ===
    // x_new = x3 + c4 * dt * v3
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] += c4 * eff_dt * s_v[i];
        
        // Write to Global
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_yoshida_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W, const float* V_w,
    float* x_new, float* v_new,
    float dt, float dt_scale_scalar,
    const float* dt_scale_tensor, // Added
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
) {
    // Calculate shared memory size
    int shared_bytes = (3 * dim + MAX_RANK + 3) * sizeof(float);
    
    yoshida_fused_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x, v, f, U, W, V_w,
        x_new, v_new,
        dt, dt_scale_scalar, dt_scale_tensor,
        batch, dim, rank,
        plasticity, sing_thresh, sing_strength, use_active
    );
}
