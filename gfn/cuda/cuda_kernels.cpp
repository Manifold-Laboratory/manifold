#ifdef _WIN32
#define NOMINMAX
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CRITICAL FIX for CUDA 13.0 + PyTorch conflict:
// Include ATen/cuda/CUDAContext.h AFTER torch/extension.h
// This header defines cusparseGetErrorString which conflicts with CUDA 13.0
// We need it for getCurrentCUDAStream() but must include it carefully
#include <ATen/cuda/CUDAContext.h>

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

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
);

std::vector<torch::Tensor> christoffel_backward_cuda(
    torch::Tensor grad_gamma,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    float plasticity,
    float sing_thresh,
    float sing_strength
) {
    const auto batch = v.size(0);
    const auto dim = v.size(1);
    const auto rank = U.size(1);
    
    // Gradient Arrays
    auto grad_v = torch::zeros_like(v);
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    
    // Optional Outputs
    torch::Tensor grad_x;
    torch::Tensor grad_V_w;
    
    // Active Logic
    bool use_active = false;
    const float* x_ptr = nullptr;
    const float* V_w_ptr = nullptr;
    float* grad_x_ptr = nullptr;
    float* grad_V_w_ptr = nullptr;
    
    if (x.defined() && x.numel() > 0 && V_w.defined() && V_w.numel() > 0) {
        grad_x = torch::zeros_like(x);
        grad_V_w = torch::zeros_like(V_w);
        x_ptr = x.data_ptr<float>();
        V_w_ptr = V_w.data_ptr<float>();
        grad_x_ptr = grad_x.data_ptr<float>();
        grad_V_w_ptr = grad_V_w.data_ptr<float>();
        use_active = true;
    }
    
    if (plasticity != 0.0f) use_active = true;

    launch_christoffel_backward(
        grad_gamma.data_ptr<float>(),
        v.data_ptr<float>(),
        U.data_ptr<float>(),
        W.data_ptr<float>(),
        x_ptr,
        V_w_ptr,
        grad_v.data_ptr<float>(),
        grad_U.data_ptr<float>(),
        grad_W.data_ptr<float>(),
        grad_x_ptr,
        grad_V_w_ptr,
        batch, dim, rank, 
        plasticity, sing_thresh, sing_strength, use_active,
        at::cuda::getCurrentCUDAStream()
    );
    
    // Return gradients. If optional ones are undefined, we return them (will be None in Python?)
    // PyBind11 converts undefined Tensor to None? No, it might crash or return empty tensor.
    // Better return consistent list. Python side handles None unwrapping.
    // Actually, PyTorch autograd expects Tensors.
    // If grad_x is not defined, we should return empty tensor?
    // Let's ensure we return valid objects.
    
    if (!grad_x.defined()) grad_x = torch::Tensor(); // Undefined
    if (!grad_V_w.defined()) grad_V_w = torch::Tensor(); // Undefined
    
    return {grad_v, grad_U, grad_W, grad_x, grad_V_w};
}

extern "C" void launch_yoshida_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W, const float* V_w,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    const float* dt_scale_tensor, // Added
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, cudaStream_t stream
);

std::vector<torch::Tensor> yoshida_fused_cuda(
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor f,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor V_w, // Optional
    float dt,
    py::object dt_scale_arg, // Accepts float or Tensor
    float plasticity,
    float sing_thresh,
    float sing_strength
) {
    const auto batch = x.size(0);
    const auto dim = x.size(1);
    const auto rank = U.size(1);
    
    auto x_new = torch::zeros_like(x);
    auto v_new = torch::zeros_like(v);
    
    bool use_active = false;
    const float* V_ptr = nullptr;
    if (plasticity != 0.0f || (V_w.defined() && V_w.numel() > 0)) {
        use_active = true;
    }
    if (V_w.defined() && V_w.numel() > 0) {
        V_ptr = V_w.data_ptr<float>();
    }
    
    // Handle dt_scale (Scalar or Tensor)
    float dt_scale_scalar = 1.0f;
    const float* dt_scale_ptr = nullptr;
    
    // Check if it's a Tensor
    if (py::isinstance<torch::Tensor>(dt_scale_arg)) {
        torch::Tensor t = dt_scale_arg.cast<torch::Tensor>();
        // Ensure continuous & float
        if (t.defined() && t.numel() > 0) {
            // If it's a scalar tensor
            if (t.numel() == 1) {
                dt_scale_scalar = t.item<float>();
            } else {
                dt_scale_ptr = t.data_ptr<float>();
            }
        } else {
            // Empty tensor -> usage default 1.0 or error? 
            // Assume default.
        }
    } else {
        // Assume float
        try {
            dt_scale_scalar = dt_scale_arg.cast<float>();
        } catch (...) {
            dt_scale_scalar = 1.0f;
        }
    }

    launch_yoshida_fused(
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        f.data_ptr<float>(),
        U.data_ptr<float>(),
        W.data_ptr<float>(),
        V_ptr,
        x_new.data_ptr<float>(),
        v_new.data_ptr<float>(),
        dt, dt_scale_scalar, dt_scale_ptr,
        batch, dim, rank,
        plasticity, sing_thresh, sing_strength, use_active,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {x_new, v_new};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda, "Christoffel Fused Forward");
    m.def("christoffel_backward", &christoffel_backward_cuda, "Christoffel Backward");
    m.def("leapfrog_fused", &leapfrog_fused_cuda, "Leapfrog Fused Integrator");
    m.def("parallel_scan_fused", &parallel_scan_fused_cuda, "Parallel Scan Fused");
    m.def("yoshida_fused", &yoshida_fused_cuda, "Yoshida Fused Integrator");
}
