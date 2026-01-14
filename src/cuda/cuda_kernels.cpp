/*
 * CUDA Kernel Bindings
 * ====================
 * PyTorch C++ extension API bindings for GFN kernels.
 */

#include <valarray>
#include <torch/extension.h>

// Forward declarations
torch::Tensor christoffel_fused_cuda(
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W
);

std::vector<torch::Tensor> leapfrog_fused_cuda(
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor f,
    torch::Tensor U,
    torch::Tensor W,
    float dt,
    float dt_scale
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda, "Fused Christoffel computation (CUDA)");
    m.def("leapfrog_fused", &leapfrog_fused_cuda, "Fused Leapfrog integration (CUDA)");
}
