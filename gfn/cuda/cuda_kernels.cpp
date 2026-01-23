
#ifdef _WIN32
#define NOMINMAX
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CRITICAL FIX for CUDA + PyTorch conflict:
#include <ATen/cuda/CUDAContext.h>

// Forward declarations
extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
);

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
);

extern "C" void launch_leapfrog_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
);

extern "C" void launch_euler_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
);

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
);

extern "C" void launch_verlet_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
);

extern "C" void launch_forest_ruth_fused(
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
);

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
);

extern "C" void launch_heun_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
);

extern "C" void launch_rk4_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
);

extern "C" void launch_dormand_prince_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps,
    cudaStream_t stream
);

extern "C" void launch_parallel_scan_fused(
    const float* a, const float* x, float* y,
    int batch, int seq_len, int dim,
    cudaStream_t stream
);

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, float dt_scale,
    cudaStream_t stream
);

// Wrappers
torch::Tensor christoffel_fused_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength) {
    auto gamma = torch::empty_like(v);
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (x_ptr != nullptr && V_ptr != nullptr);
    launch_christoffel_fused(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), x_ptr, V_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> christoffel_backward_cuda(torch::Tensor grad_gamma, torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength) {
    auto grad_v = torch::zeros_like(v);
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    torch::Tensor grad_x, grad_V_w;
    const float *x_ptr = nullptr, *V_ptr = nullptr;
    float *gx_ptr = nullptr, *gV_ptr = nullptr;
    bool use_active = false;
    if (x.numel() > 0) { grad_x = torch::zeros_like(x); x_ptr = x.data_ptr<float>(); gx_ptr = grad_x.data_ptr<float>(); use_active = true; }
    if (V_w.numel() > 0) { grad_V_w = torch::zeros_like(V_w); V_ptr = V_w.data_ptr<float>(); gV_ptr = grad_V_w.data_ptr<float>(); use_active = true; }
    if (plasticity != 0.0f) use_active = true;
    launch_christoffel_backward(grad_gamma.data_ptr<float>(), v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_ptr, V_ptr, grad_v.data_ptr<float>(), grad_U.data_ptr<float>(), grad_W.data_ptr<float>(), gx_ptr, gV_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, at::cuda::getCurrentCUDAStream());
    return {grad_v, grad_U, grad_W, grad_x, grad_V_w};
}

std::vector<torch::Tensor> euler_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    launch_euler_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> leapfrog_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    launch_leapfrog_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> yoshida_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor V_w, float dt, float dt_scale, float plasticity, float sing_thresh, float sing_strength, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (V_ptr != nullptr);
    launch_yoshida_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), V_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, nullptr, x.size(0), x.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> verlet_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    launch_verlet_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> forest_ruth_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor V_w, float dt, float dt_scale, float plasticity, float sing_thresh, float sing_strength, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (V_ptr != nullptr);
    launch_forest_ruth_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), V_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, nullptr, x.size(0), x.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> omelyan_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, torch::Tensor V_w, float dt, float dt_scale, float plasticity, float sing_thresh, float sing_strength, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (V_ptr != nullptr);
    launch_omelyan_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), V_ptr, x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, nullptr, x.size(0), x.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> heun_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    launch_heun_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> rk4_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    launch_rk4_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> dormand_prince_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    launch_dormand_prince_fused(x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

torch::Tensor parallel_scan_fused_cuda(torch::Tensor a, torch::Tensor x) {
    auto y = torch::empty_like(x);
    launch_parallel_scan_fused(a.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), a.size(0), a.size(1), a.size(2), at::cuda::getCurrentCUDAStream());
    return y;
}

std::vector<torch::Tensor> recurrent_manifold_fused_cuda(
    torch::Tensor x_state, torch::Tensor v_state,
    torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, float dt_scale
) {
    int batch = x_state.size(0);
    int dim = x_state.size(1);
    int seq_len = forces.size(1);
    int rank = U_stack.size(2);
    int num_layers = U_stack.size(0);
    
    auto x_out_seq = torch::empty({batch, seq_len, dim}, x_state.options());
    
    launch_recurrent_manifold_fused(
        x_state.data_ptr<float>(), v_state.data_ptr<float>(),
        forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(),
        x_out_seq.data_ptr<float>(),
        batch, seq_len, dim, rank, num_layers, dt, dt_scale,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {x_state, v_state, x_out_seq};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda);
    m.def("christoffel_backward", &christoffel_backward_cuda);
    m.def("euler_fused", &euler_fused_cuda);
    m.def("leapfrog_fused", &leapfrog_fused_cuda);
    m.def("yoshida_fused", &yoshida_fused_cuda);
    m.def("verlet_fused", &verlet_fused_cuda);
    m.def("forest_ruth_fused", &forest_ruth_fused_cuda);
    m.def("omelyan_fused", &omelyan_fused_cuda);
    m.def("heun_fused", &heun_fused_cuda);
    m.def("rk4_fused", &rk4_fused_cuda);
    m.def("dormand_prince_fused", &dormand_prince_fused_cuda);
    m.def("parallel_scan_fused", &parallel_scan_fused_cuda);
    m.def("recurrent_manifold_fused", &recurrent_manifold_fused_cuda);
}
