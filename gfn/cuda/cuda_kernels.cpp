#ifdef _WIN32
#define NOMINMAX
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of raw CUDA launchers (from .cu files)
extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
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
    cudaStream_t stream
);

extern "C" void launch_parallel_scan_fused(
    const float* a, const float* x, float* y,
    int batch, int seq_len, int dim,
    cudaStream_t stream
);

// PyTorch Wrappers
torch::Tensor christoffel_fused_cuda(
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    float plasticity,
    float sing_thresh,
    float sing_strength
) {
    const int batch = v.size(0);
    const int dim = v.size(1);
    const int rank = U.size(1);
    
    auto gamma = torch::empty_like(v);
    
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (x_ptr != nullptr && V_ptr != nullptr);

    launch_christoffel_fused(
        v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(),
        x_ptr, V_ptr, batch, dim, rank,
        plasticity, sing_thresh, sing_strength, use_active,
        at::cuda::getCurrentCUDAStream()
    );
    
    return gamma;
}

std::vector<torch::Tensor> leapfrog_fused_cuda(
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor f,
    torch::Tensor U,
    torch::Tensor W,
    float dt,
    float dt_scale
) {
    const int batch = x.size(0);
    const int dim = x.size(1);
    const int rank = U.size(1);
    
    auto x_new = torch::empty_like(x);
    auto v_new = torch::empty_like(v);

    launch_leapfrog_fused(
        x.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(),
        U.data_ptr<float>(), W.data_ptr<float>(),
        x_new.data_ptr<float>(), v_new.data_ptr<float>(),
        dt, dt_scale, batch, dim, rank,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {x_new, v_new};
}

torch::Tensor parallel_scan_fused_cuda(
    torch::Tensor a,
    torch::Tensor x
) {
    const int batch = a.size(0);
    const int seq_len = a.size(1);
    const int dim = a.size(2);
    
    auto y = torch::empty_like(x);
    
    launch_parallel_scan_fused(
        a.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(),
        batch, seq_len, dim,
        at::cuda::getCurrentCUDAStream()
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda, "Fused Christoffel computation (CUDA)");
    m.def("leapfrog_fused", &leapfrog_fused_cuda, "Fused Leapfrog integration (CUDA)");
    m.def("parallel_scan_fused", &parallel_scan_fused_cuda, "Fused Parallel Scan (CUDA)");
}
