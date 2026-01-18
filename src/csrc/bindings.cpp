#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernels
void christoffel_fused_cuda(
    torch::Tensor x,
    torch::Tensor metric_weights,
    torch::Tensor output_gamma
);

void leapfrog_fused_cuda(
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor force,
    torch::Tensor out_x,
    torch::Tensor out_v,
    float dt,
    float friction
);

// C++ Interface
void christoffel_fused(torch::Tensor x, torch::Tensor metric_weights, torch::Tensor output_gamma) {
    christoffel_fused_cuda(x, metric_weights, output_gamma);
}

void leapfrog_fused(torch::Tensor x, torch::Tensor v, torch::Tensor force, 
                   torch::Tensor out_x, torch::Tensor out_v, float dt, float friction) {
    leapfrog_fused_cuda(x, v, force, out_x, out_v, dt, friction);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused, "Fused Christoffel Calculation (CUDA)");
    m.def("leapfrog_fused", &leapfrog_fused, "Fused Symplectic Leapfrog Integrator (CUDA)");
}
