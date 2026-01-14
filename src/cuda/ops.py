"""
Python interface for GFN CUDA kernels with fallback to PyTorch.
"""

import torch
import os

# Try to load CUDA extension
try:
    from torch.utils.cpp_extension import load
    
    # Build path
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load extension (JIT compilation on first import)
    # Try loading pre-compiled extension first
    try:
        # Implicit relative import for when installed as package
        from . import gfn_cuda
    except ImportError:
        try:
            # Absolute import
            import gfn_cuda
        except ImportError:
             # Fallback to JIT compilation if pre-compiled not found
             # (This is useful for development but fragile on Windows)
             print("[GFN CUDA] Pre-compiled extension not found, attempting JIT compilation...")
             gfn_cuda = load(
                name='gfn_cuda',
                sources=[
                    os.path.join(cuda_dir, 'cuda_kernels.cpp'),
                    os.path.join(cuda_dir, 'kernels', 'christoffel_fused.cu'),
                    os.path.join(cuda_dir, 'kernels', 'leapfrog_fused.cu'),
                ],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                extra_cflags=['/std:c++17', '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/permissive-'],
                verbose=False
            )
    
    CUDA_AVAILABLE = True
    print("[GFN CUDA] Custom kernels loaded successfully")
    
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"[GFN CUDA] Failed to load custom kernels: {e}")
    print("[GFN CUDA] Falling back to PyTorch implementation")


def christoffel_fused(v, U, W):
    """
    Fused Christoffel symbol computation: Γ(v,v) = W * (U^T v)^2
    
    Args:
        v: Velocity tensor [batch, dim]
        U: Left projection matrix [dim, rank]
        W: Right projection matrix [dim, rank]
        
    Returns:
        gamma: Christoffel symbols [batch, dim]
    """
    if CUDA_AVAILABLE and v.is_cuda:
        return gfn_cuda.christoffel_fused(v, U, W)
    else:
        # PyTorch fallback
        proj = torch.matmul(v, U)  # [batch, rank]
        sq = proj * proj            # [batch, rank]
        gamma = torch.matmul(sq, W.t())  # [batch, dim]
        return torch.clamp(gamma, -5.0, 5.0)


def leapfrog_fused(x, v, f, U, W, dt, dt_scale=1.0):
    """
    Fused Leapfrog integration step with inline Christoffel computation.
    
    Args:
        x: Position [batch, dim]
        v: Velocity [batch, dim]
        f: Force [batch, dim]
        U, W: Christoffel matrices [dim, rank]
        dt: Time step
        dt_scale: Adaptive time scaling (gate)
        
    Returns:
        x_new, v_new: Updated position and velocity
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.leapfrog_fused(x, v, f, U, W, dt, dt_scale)
    else:
        # PyTorch fallback
        effective_dt = dt * dt_scale
        
        # Half-step velocity
        gamma_v = christoffel_fused(v, U, W)
        v_half = v + 0.5 * effective_dt * (f - gamma_v)
        
        # Full-step position
        x_new = x + effective_dt * v_half
        
        # Half-step velocity again
        gamma_v_half = christoffel_fused(v_half, U, W)
        v_new = v_half + 0.5 * effective_dt * (f - gamma_v_half)
        
        return x_new, v_new
