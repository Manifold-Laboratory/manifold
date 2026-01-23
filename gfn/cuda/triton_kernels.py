"""
Triton-based CUDA kernels for Manifold GFN
Replaces problematic C++/CUDA kernels with Triton (works on Windows!)
"""

import torch
import triton
import triton.language as tl

@triton.jit
def christoffel_fused_kernel(
    # Pointers
    v_ptr, U_ptr, W_ptr, gamma_ptr,
    x_ptr, V_w_ptr,
    # Dimensions
    batch, dim, rank,
    # Physics params
    plasticity, sing_thresh, sing_strength,
    use_active: tl.constexpr,
    # Strides
    stride_vb, stride_vd,
    stride_Ud, stride_Ur,
    stride_Wd, stride_Wr,
    stride_gb, stride_gd,
    stride_xb, stride_xd,
    stride_Vd, stride_Vr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Christoffel symbol computation with active inference.
    Replaces christoffel_fused.cu
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    # Offsets
    offs_r = tl.arange(0, BLOCK_SIZE)
    mask_r = offs_r < rank
    
    # Load v[batch, dim]
    v_idx = pid_batch * stride_vb + pid_dim * stride_vd
    v_val = tl.load(v_ptr + v_idx)
    
    # Load U[dim, :] and W[dim, :]
    U_base = pid_dim * stride_Ud
    W_base = pid_dim * stride_Wd
    
    U_vals = tl.load(U_ptr + U_base + offs_r * stride_Ur, mask=mask_r, other=0.0)
    W_vals = tl.load(W_ptr + W_base + offs_r * stride_Wr, mask=mask_r, other=0.0)
    
    # Compute gamma = U @ (W^T @ v)
    # First: W^T @ v (reduce over rank)
    Wv = tl.sum(W_vals * v_val, axis=0)
    
    # Then: U @ Wv
    gamma = tl.sum(U_vals * Wv, axis=0)
    
    # Active inference (if enabled)
    if use_active:
        # Load x[batch, dim]
        x_idx = pid_batch * stride_xb + pid_dim * stride_xd
        x_val = tl.load(x_ptr + x_idx)
        
        # Load V_w[dim, :]
        V_base = pid_dim * stride_Vd
        V_vals = tl.load(V_w_ptr + V_base + offs_r * stride_Vr, mask=mask_r, other=0.0)
        
        # Compute friction = sigmoid(V @ x) * v
        Vx = tl.sum(V_vals * x_val, axis=0)
        friction = tl.sigmoid(Vx) * v_val
        
        # Apply plasticity and singularities
        x_norm = tl.sqrt(x_val * x_val + 1e-8)
        if x_norm > sing_thresh:
            sing_factor = 1.0 + sing_strength * tl.exp(-(x_norm - sing_thresh))
            friction = friction * sing_factor
        
        gamma = gamma + plasticity * friction
    
    # Store result
    gamma_idx = pid_batch * stride_gb + pid_dim * stride_gd
    tl.store(gamma_ptr + gamma_idx, gamma)


def christoffel_fused_triton(v, U, W, x=None, V_w=None, plasticity=0.1, sing_thresh=5.0, sing_strength=10.0):
    """
    Triton wrapper for Christoffel computation.
    100% compatible with original CUDA kernel API.
    """
    batch, dim = v.shape
    rank = U.shape[1]
    
    # Output
    gamma = torch.empty_like(v)
    
    # Grid
    grid = lambda meta: (batch, dim)
    
    # Launch kernel
    christoffel_fused_kernel[grid](
        v, U, W, gamma,
        x, V_w if V_w is not None else torch.zeros(1, device=v.device),
        batch, dim, rank,
        plasticity, sing_thresh, sing_strength,
        use_active=(x is not None and V_w is not None),
        # Strides
        v.stride(0), v.stride(1),
        U.stride(0), U.stride(1),
        W.stride(0), W.stride(1),
        gamma.stride(0), gamma.stride(1),
        x.stride(0) if x is not None else 0, x.stride(1) if x is not None else 0,
        V_w.stride(0) if V_w is not None else 0, V_w.stride(1) if V_w is not None else 0,
        BLOCK_SIZE=triton.next_power_of_2(rank),
    )
    
    return gamma


# Export for use in gfn/cuda/ops.py
__all__ = ['christoffel_fused_triton']
