
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

extern "C" void launch_lowrank_christoffel_forward(
    const float* v, const float* U, const float* W, float* gamma,
    int batch, int dim, int rank, cudaStream_t stream
);

extern "C" void launch_lowrank_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W,
    float* grad_v, float* grad_U, float* grad_W,
    int batch, int dim, int rank, cudaStream_t stream
);

extern "C" void launch_reactive_christoffel_forward(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    cudaStream_t stream
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

extern "C" void launch_leapfrog_backward(
    const float* grad_x_new, const float* grad_v_new,
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* grad_x, float* grad_v, float* grad_f,
    float* grad_U, float* grad_W,
    int batch, int dim, int rank,
    float dt, float dt_scale, int steps,
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
    float plasticity,
    cudaStream_t stream
);

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* reg_loss,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, const float* dt_scales, const float* forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    const float* W_mix_x, const float* W_mix_v,
    cudaStream_t stream
);

extern "C" void launch_recurrent_manifold_backward(
    const float* grad_x_seq, const float* grad_x_final, const float* grad_v_final,
    const float* x_final, const float* v_final,
    const float* forces, const float* U_stack, const float* W_stack,
    float* grad_x_init, float* grad_v_init, float* grad_forces,
    float* grad_U, float* grad_W,
    const float* W_mix_x, const float* W_mix_v,
    float* grad_W_mix_x, float* grad_W_mix_v,
    int batch_total, int seq_len, int dim, int rank, int num_layers, int num_heads,
    float dt, const float* dt_scales, const float* forget_rates, float* grad_forget_rates,
    float plasticity, float sing_thresh, float sing_strength,
    cudaStream_t stream
);

extern "C" void launch_manifold_step_fused(
    const float* x_in, const float* v_in, const float* force,
    const float* U, const float* W,
    float* x_out, float* v_out, float* christoffel_out,
    int batch, int dim, int rank,
    float dt, float dt_scale,
    float plasticity, float sing_thresh, float sing_strength,
    cudaStream_t stream
);

extern "C" void launch_head_mixing_fused(
    const float* x_heads, const float* v_heads,
    const float* W_x, const float* W_v,
    float* x_out, float* v_out,
    int heads, int batch, int dim,
    cudaStream_t stream
);

extern "C" void launch_dynamic_gating_fused(
    const float* x, const float* W1, const float* b1,
    const float* W2, const float* b2,
    float* dt_scale,
    int batch, int dim, int hidden_dim,
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

torch::Tensor lowrank_christoffel_forward_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W) {
    auto gamma = torch::empty_like(v);
    launch_lowrank_christoffel_forward(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), v.size(0), v.size(1), U.size(1), at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> lowrank_christoffel_backward_cuda(torch::Tensor grad_gamma, torch::Tensor v, torch::Tensor U, torch::Tensor W) {
    auto grad_v = torch::zeros_like(v);
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    launch_lowrank_christoffel_backward(grad_gamma.data_ptr<float>(), v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), grad_v.data_ptr<float>(), grad_U.data_ptr<float>(), grad_W.data_ptr<float>(), v.size(0), v.size(1), U.size(1), at::cuda::getCurrentCUDAStream());
    return {grad_v, grad_U, grad_W};
}

torch::Tensor reactive_christoffel_forward_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength) {
    auto gamma = torch::empty_like(v);
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    launch_reactive_christoffel_forward(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), x_ptr, V_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> leapfrog_backward_cuda(
    torch::Tensor grad_x_new, torch::Tensor grad_v_new,
    torch::Tensor x, torch::Tensor v, torch::Tensor f,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale, int steps
) {
    auto grad_x = torch::zeros_like(x);
    auto grad_v = torch::zeros_like(v);
    auto grad_f = torch::zeros_like(f);
    auto grad_U = torch::zeros_like(U);
    auto grad_W = torch::zeros_like(W);
    
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    float* gf_ptr = (f.numel() > 0) ? grad_f.data_ptr<float>() : nullptr;
    
    launch_leapfrog_backward(
        grad_x_new.data_ptr<float>(), grad_v_new.data_ptr<float>(),
        x.data_ptr<float>(), v.data_ptr<float>(), f_ptr,
        U.data_ptr<float>(), W.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_v.data_ptr<float>(), gf_ptr,
        grad_U.data_ptr<float>(), grad_W.data_ptr<float>(),
        x.size(0), x.size(1), U.size(1),
        dt, dt_scale, steps,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {grad_x, grad_v, grad_f, grad_U, grad_W};
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

torch::Tensor parallel_scan_fused_cuda(torch::Tensor a, torch::Tensor x, float plasticity) {
    auto y = torch::empty_like(x);
    launch_parallel_scan_fused(a.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), a.size(0), a.size(1), a.size(2), plasticity, at::cuda::getCurrentCUDAStream());
    return y;
}

// (Declarations removed here, consolidated at top)

std::vector<torch::Tensor> recurrent_manifold_fused_cuda(
    torch::Tensor x_state, torch::Tensor v_state,
    torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, torch::Tensor dt_scales, torch::Tensor forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    torch::Tensor mix_x, torch::Tensor mix_v 
) {
    int batch = x_state.size(0);
    int dim = x_state.size(1);
    int seq_len = forces.size(1);
    int rank = U_stack.size(2);
    int num_layers = U_stack.size(0) / num_heads;  // Account for heads in stack
    
    auto x_out_seq = torch::empty({batch, seq_len, dim}, x_state.options());
    auto reg_loss = torch::zeros({batch}, x_state.options());
    
    const float* mix_x_ptr = (mix_x.numel() > 0) ? mix_x.data_ptr<float>() : nullptr;
    const float* mix_v_ptr = (mix_v.numel() > 0) ? mix_v.data_ptr<float>() : nullptr;
    const float* dt_scales_ptr = (dt_scales.numel() > 0) ? dt_scales.data_ptr<float>() : nullptr;
    const float* forget_rates_ptr = (forget_rates.numel() > 0) ? forget_rates.data_ptr<float>() : nullptr;
    
    launch_recurrent_manifold_fused(
        x_state.data_ptr<float>(), v_state.data_ptr<float>(),
        forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(),
        x_out_seq.data_ptr<float>(), reg_loss.data_ptr<float>(),
        batch, seq_len, dim, rank, num_layers,
        dt, dt_scales_ptr, forget_rates_ptr, num_heads,
        plasticity, sing_thresh, sing_strength,
        mix_x_ptr, mix_v_ptr,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {x_state, v_state, x_out_seq, reg_loss};
}

// (Declarations removed here, consolidated at top)

std::vector<torch::Tensor> recurrent_manifold_backward_cuda(
    torch::Tensor grad_x_seq, torch::Tensor grad_x_final, torch::Tensor grad_v_final,
    torch::Tensor x_final, torch::Tensor v_final,
    torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, torch::Tensor dt_scales, torch::Tensor forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    torch::Tensor mix_x, torch::Tensor mix_v
) {
    int batch_total = x_final.size(0);
    int dim = x_final.size(1);
    int seq_len = forces.size(1);
    int rank = U_stack.size(2);
    int num_layers = U_stack.size(0) / num_heads;

    auto grad_x_init = torch::zeros_like(x_final);
    auto grad_v_init = torch::zeros_like(v_final);
    auto grad_forces = torch::zeros_like(forces);
    auto grad_U = torch::zeros_like(U_stack);
    auto grad_W = torch::zeros_like(W_stack);
    
    // Mixing Logic
    const bool use_mixing = (mix_x.numel() > 0 && mix_v.numel() > 0);
    auto grad_mix_x = use_mixing ? torch::zeros_like(mix_x) : torch::empty(0, x_final.options());
    auto grad_mix_v = use_mixing ? torch::zeros_like(mix_v) : torch::empty(0, x_final.options());
    
    const float* mix_x_ptr = use_mixing ? mix_x.data_ptr<float>() : nullptr;
    const float* mix_v_ptr = use_mixing ? mix_v.data_ptr<float>() : nullptr;
    float* g_mix_x_ptr = use_mixing ? grad_mix_x.data_ptr<float>() : nullptr;
    float* g_mix_v_ptr = use_mixing ? grad_mix_v.data_ptr<float>() : nullptr;
    const float* dt_scales_ptr = (dt_scales.numel() > 0) ? dt_scales.data_ptr<float>() : nullptr;
    const float* forget_rates_ptr = (forget_rates.numel() > 0) ? forget_rates.data_ptr<float>() : nullptr;
    auto grad_forget_rates = torch::zeros_like(forget_rates);

    const float* gx_seq_ptr = (grad_x_seq.numel() > 0) ? grad_x_seq.data_ptr<float>() : nullptr;
    const float* gx_final_ptr = (grad_x_final.numel() > 0) ? grad_x_final.data_ptr<float>() : nullptr;
    const float* gv_final_ptr = (grad_v_final.numel() > 0) ? grad_v_final.data_ptr<float>() : nullptr;

    launch_recurrent_manifold_backward(
        gx_seq_ptr, gx_final_ptr, gv_final_ptr,
        x_final.data_ptr<float>(), v_final.data_ptr<float>(),
        forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(),
        grad_x_init.data_ptr<float>(), grad_v_init.data_ptr<float>(), grad_forces.data_ptr<float>(),
        grad_U.data_ptr<float>(), grad_W.data_ptr<float>(),
        mix_x_ptr, mix_v_ptr, g_mix_x_ptr, g_mix_v_ptr,
        batch_total, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales_ptr, forget_rates_ptr, grad_forget_rates.data_ptr<float>(),
        plasticity, sing_thresh, sing_strength,
        at::cuda::getCurrentCUDAStream()
    );

    return {grad_x_init, grad_v_init, grad_forces, grad_U, grad_W, grad_mix_x, grad_mix_v, grad_forget_rates};
}

std::vector<torch::Tensor> manifold_step_fused_cuda(
    torch::Tensor x_in, torch::Tensor v_in, torch::Tensor force,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale,
    float plasticity, float sing_thresh, float sing_strength,
    bool return_christoffel
) {
    int batch = x_in.size(0);
    int dim = x_in.size(1);
    int rank = U.size(1);
    
    auto x_out = torch::empty_like(x_in);
    auto v_out = torch::empty_like(v_in);
    torch::Tensor christoffel_out = torch::empty(0, x_in.options());
    
    float* c_ptr = nullptr;
    if (return_christoffel) {
        christoffel_out = torch::empty_like(x_in);
        c_ptr = christoffel_out.data_ptr<float>();
    }
    
    launch_manifold_step_fused(
        x_in.data_ptr<float>(), v_in.data_ptr<float>(), force.data_ptr<float>(),
        U.data_ptr<float>(), W.data_ptr<float>(),
        x_out.data_ptr<float>(), v_out.data_ptr<float>(), c_ptr,
        batch, dim, rank, dt, dt_scale,
        plasticity, sing_thresh, sing_strength,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {x_out, v_out, christoffel_out};
}

std::vector<torch::Tensor> head_mixing_fused_cuda(
    torch::Tensor x_heads, torch::Tensor v_heads,
    torch::Tensor W_x, torch::Tensor W_v
) {
    int heads = x_heads.size(0);
    int batch = x_heads.size(1);
    int head_dim = x_heads.size(2);
    int dim = heads * head_dim;
    
    auto x_out = torch::empty({batch, dim}, x_heads.options());
    auto v_out = torch::empty({batch, dim}, v_heads.options());
    
    launch_head_mixing_fused(
        x_heads.data_ptr<float>(), v_heads.data_ptr<float>(),
        W_x.data_ptr<float>(), W_v.data_ptr<float>(),
        x_out.data_ptr<float>(), v_out.data_ptr<float>(),
        heads, batch, dim,
        at::cuda::getCurrentCUDAStream()
    );
    
    return {x_out, v_out};
}

torch::Tensor dynamic_gating_fused_cuda(
    torch::Tensor x,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2
) {
    int batch = x.size(0);
    int dim = x.size(1);
    int hidden_dim = W1.size(0);
    
    auto dt_scale = torch::empty({batch, 1}, x.options());
    
    launch_dynamic_gating_fused(
        x.data_ptr<float>(),
        W1.data_ptr<float>(), b1.data_ptr<float>(),
        W2.data_ptr<float>(), b2.data_ptr<float>(),
        dt_scale.data_ptr<float>(),
        batch, dim, hidden_dim,
        at::cuda::getCurrentCUDAStream()
    );
    
    return dt_scale;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda);
    m.def("christoffel_backward", &christoffel_backward_cuda);
    m.def("lowrank_christoffel_forward", &lowrank_christoffel_forward_cuda);
    m.def("lowrank_christoffel_backward", &lowrank_christoffel_backward_cuda);
    m.def("reactive_christoffel_forward", &reactive_christoffel_forward_cuda);
    m.def("leapfrog_backward", &leapfrog_backward_cuda);
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
    m.def("recurrent_manifold_fused", &recurrent_manifold_fused_cuda, "Fused Recurrent Manifold Kernel with Multi-Head Mixing",
          py::arg("x_state"), py::arg("v_state"), py::arg("forces"), py::arg("U_stack"), py::arg("W_stack"),
          py::arg("dt"), py::arg("dt_scales"), py::arg("forget_rates"), py::arg("num_heads"),
          py::arg("plasticity")=0.0, py::arg("sing_thresh")=1.0, py::arg("sing_strength")=1.0,
          py::arg("mix_x")=torch::Tensor(), py::arg("mix_v")=torch::Tensor());
    m.def("recurrent_manifold_backward", &recurrent_manifold_backward_cuda, "Backprop for Recurrent Manifold Fused Kernel",
          py::arg("grad_x_seq"), py::arg("grad_x_final"), py::arg("grad_v_final"),
          py::arg("x_final"), py::arg("v_final"),
          py::arg("forces"), py::arg("U_stack"), py::arg("W_stack"),
          py::arg("dt"), py::arg("dt_scales"), py::arg("forget_rates"), py::arg("num_heads"),
          py::arg("plasticity")=0.0, py::arg("sing_thresh")=1.0, py::arg("sing_strength")=1.0,
          py::arg("mix_x")=torch::Tensor(), py::arg("mix_v")=torch::Tensor());
    m.def("manifold_step_fused", &manifold_step_fused_cuda);
    m.def("head_mixing_fused", &head_mixing_fused_cuda);
    m.def("dynamic_gating_fused", &dynamic_gating_fused_cuda);
}
