"""
Python interface for GFN CUDA kernels with fallback to PyTorch.
"""

import torch
import os
import sys
import platform

# Try to load CUDA extension
try:
    from torch.utils.cpp_extension import load
    
    # Build path (Allow override via env var for custom builds/deployments)
    cuda_dir = os.environ.get('GFN_CUDA_PATH')
    if cuda_dir is None:
        cuda_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        print(f"[GFN CUDA] Using custom CUDA source path: {cuda_dir}")

    # Platform-specific compiler setup
    is_windows = platform.system() == 'Windows'
    
    # Load extension logic
    # 1. Try direct import (e.g. if compiled via setup.py install/develop or in CWD)
    try:
        import gfn_cuda
        CUDA_AVAILABLE = True
        print("[GFN CUDA] Custom kernels loaded successfully (direct import)")
    except ImportError:
         # 2. JIT fallback
         print("[GFN CUDA] Pre-compiled extension not found, attempting JIT compilation...")
         
         # Build path (Allow override via env var for custom builds/deployments)
         cuda_dir = os.environ.get('GFN_CUDA_PATH')
         if cuda_dir is None:
            cuda_dir = os.path.dirname(os.path.abspath(__file__))
         else:
            print(f"[GFN CUDA] Using custom CUDA source path: {cuda_dir}")

         # Platform-specific compilation flags
         if is_windows:
             extra_cflags = ['/O2']
             extra_cuda_cflags = [
                 '-O3', 
                 '--use_fast_math', 
                 '-allow-unsupported-compiler',
                 '-D_ALLOW_COMPILER_SUBSTITUTIONS'
             ]
         else:
             extra_cflags = ['-O3', '-fPIC']
             extra_cuda_cflags = ['-O3', '--use_fast_math', '--compiler-options', "'-fPIC'"]
         
         
         gfn_cuda = load(
            name='gfn_cuda',
            sources=[
                os.path.join(cuda_dir, 'cuda_kernels.cpp'),
                os.path.join(cuda_dir, 'src', 'geometry', 'christoffel_fused.cu'),
                os.path.join(cuda_dir, 'src', 'integrators', 'leapfrog_fused.cu'),
                os.path.join(cuda_dir, 'src', 'layers', 'parallel_scan_fused.cu'),
            ],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            verbose=True
        )
         CUDA_AVAILABLE = True
         print("[GFN CUDA] JIT compilation successful")
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"[GFN CUDA] Failed to load custom kernels: {e}")
    
    # Try Triton as fallback (works on Windows!)
    try:
        import triton
        from .triton_kernels import christoffel_fused_triton
        print("[GFN CUDA] Using Triton kernels (Windows-compatible, ~50% speed of native CUDA)")
        TRITON_AVAILABLE = True
    except ImportError:
        print("[GFN CUDA] Triton not available, falling back to PyTorch implementation")
        TRITON_AVAILABLE = False


def christoffel_fused(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    """
    Fused Christoffel symbol computation with Active Inference support.
    
    Γ(v,v) = W * (U^T v)^2
    
    If Active Inference is enabled:
    1. Plasticity: Γ = Γ * (1 + plasticity * tanh(energy))
    2. Singularities: If sigmoid(V_w * x) > thresh, Γ = Γ * strength
    
    Args:
        v: Velocity tensor [batch, dim]
        U: Left projection matrix [dim, rank]
        W: Right projection matrix [dim, rank]
        x: Position tensor [batch, dim] (Optional, for Singularities)
        V_w: Gravity well projection [1, dim] (Optional, for Singularities)
        plasticity: Alpha coefficient for reactive curvature (0.0 = disabled)
        sing_thresh: Threshold for singularity activation (0.0-1.0)
        sing_strength: Multiplier for singularity gravity
        
    Returns:
        gamma: Christoffel symbols [batch, dim]
    """
    if CUDA_AVAILABLE and v.is_cuda:
        # Check C++ extension signature compatibility.
        # Future updates will strictly enforce tensor arguments.
        # We pass empty tensors as placeholders if needed. 
        # C++ usually needs explicit tensors.
        if x is None:
            x = torch.empty(0, device=v.device, dtype=v.dtype)
        if V_w is None:
            V_w = torch.empty(0, device=v.device, dtype=v.dtype)
            
        # Use Autograd-compatible wrapper for both Training and Inference
        from .autograd import christoffel_fused_autograd
        return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)
    else:
        # PyTorch fallback
        proj = torch.matmul(v, U)  # [batch, rank]
        sq = proj * proj            # [batch, rank]
        gamma = torch.matmul(sq, W.t())  # [batch, dim]
        
        # 1. Reactive Plasticity
        if plasticity != 0.0:
            energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
            gamma = gamma * (1.0 + plasticity * energy)
            
        # 2. Singularities
        if x is not None and x.numel() > 0 and V_w is not None and V_w.numel() > 0:
            # V(x) -> scalar
            # V_w is [1, dim] usually (or [dim, 1]?)
            # Assuming V_w from nn.Linear(dim, 1) is [1, dim]
            potential = torch.sigmoid(torch.matmul(x, V_w.t()))
            is_singularity = (potential > sing_thresh).float()
            gamma = gamma * (1.0 + is_singularity * (sing_strength - 1.0))

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
        # CRITICAL: Kernels assume contiguous memory
        x = x.contiguous()
        v = v.contiguous()
        f = f.contiguous()
        U = U.contiguous()
        W = W.contiguous()
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
        
def yoshida_fused(x, v, f, U, W, dt, dt_scale=1.0, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    """
    Fused Yoshida 4th-order integration step.
    """
    if CUDA_AVAILABLE and x.is_cuda:
        from .autograd import yoshida_fused_autograd
        return yoshida_fused_autograd(
            x, v, f=f, U=U, W=W, V_w=V_w, 
            dt=dt, dt_scale=dt_scale, 
            plasticity=plasticity, sing_thresh=sing_thresh, sing_strength=sing_strength
        )
    else:
        # PyTorch fallback (Legacy manual steps)
        # Yoshida coefficients
        w0 = -1.7024143839193153
        w1 = 1.3512071919596578
        c1 = c4 = w1 / 2.0
        c2 = c3 = (w0 + w1) / 2.0
        d1 = d3 = w1
        d2 = w0
        eff_dt = dt * dt_scale
        
        # Substep 1
        x1 = x + c1 * eff_dt * v
        gamma1 = christoffel_fused(v, U, W, x1, V_w, plasticity, sing_thresh, sing_strength)
        v1 = v + d1 * eff_dt * (f - gamma1)
        
        # Substep 2
        x2 = x1 + c2 * eff_dt * v1
        gamma2 = christoffel_fused(v1, U, W, x2, V_w, plasticity, sing_thresh, sing_strength)
        v2 = v1 + d2 * eff_dt * (f - gamma2)
        
        # Substep 3
        x3 = x2 + c3 * eff_dt * v2
        gamma3 = christoffel_fused(v2, U, W, x3, V_w, plasticity, sing_thresh, sing_strength)
        v3 = v2 + d3 * eff_dt * (f - gamma3)
        
        # Final
        x_new = x3 + c4 * eff_dt * v3
        v_new = v3
        
        return x_new, v_new
