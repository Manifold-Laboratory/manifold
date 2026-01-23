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

    # Ensure MSVC is in PATH for PyTorch build system
    msvc_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
    if os.path.exists(msvc_path) and msvc_path not in os.environ['PATH']:
        print(f"[GFN CUDA] Adding MSVC to PATH: {msvc_path}")
        os.environ['PATH'] = msvc_path + os.pathsep + os.environ['PATH']
    
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
                name='gfn_cuda_v2_6_professional',
                sources=[
                    os.path.join(cuda_dir, 'cuda_kernels.cpp'),
                    os.path.join(cuda_dir, 'src', 'geometry', 'christoffel_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'leapfrog_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'yoshida_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'euler_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'layers', 'parallel_scan_fused.cu'),
                ],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-m64', '-ccbin', r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'],
                extra_cflags=['/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/Zc:twoPhase-', '/I' + os.path.join(cuda_dir, 'include')],
                verbose=True
            )
    
    CUDA_AVAILABLE = True
    print("[GFN CUDA] Professional kernels loaded successfully (with Autograd and Parallel Scan)")
    
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"[GFN CUDA] Failed to load custom kernels: {e}")
    print("[GFN CUDA] Falling back to PyTorch implementation")


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
        from .autograd import christoffel_fused_autograd
        res = christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)
        if res is not None:
            return res
    else:
        # PyTorch fallback (MUST match LowRankChristoffel exactly)
        proj = torch.matmul(v, U)  # [batch, rank]
        
        # Stability: Norm-based saturation
        norm = torch.norm(proj, dim=-1, keepdim=True)
        scale = 1.0 / (1.0 + norm)
        sq = (proj * proj) * scale
        
        gamma = torch.matmul(sq, W.t())  # [batch, dim]
        
        # 1. Reactive Plasticity
        if plasticity != 0.0:
            energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
            gamma = gamma * (1.0 + plasticity * energy)
            
        # 2. Singularities
        if x is not None and x.numel() > 0 and V_w is not None and V_w.numel() > 0:
            potential = torch.sigmoid(torch.matmul(x, V_w.t()))
            is_singularity = (potential > sing_thresh).float()
            gamma = gamma * (1.0 + is_singularity * (sing_strength - 1.0))

        return torch.clamp(gamma, -5.0, 5.0)


def leapfrog_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused Leapfrog integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.leapfrog_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x), 
                                      U.contiguous(), W.contiguous(), dt, dt_scale, steps)
    else:
        # PyTorch fallback loop
        curr_x, curr_v = x, v
        for _ in range(steps):
            effective_dt = dt * dt_scale
            gamma_v = christoffel_fused(curr_v, U, W)
            v_half = curr_v + 0.5 * effective_dt * ((f if f is not None else 0) - gamma_v)
            curr_x = curr_x + effective_dt * v_half
            gamma_v_half = christoffel_fused(v_half, U, W)
            curr_v = v_half + 0.5 * effective_dt * ((f if f is not None else 0) - gamma_v_half)
        return curr_x, curr_v

def euler_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused Euler integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.euler_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x), 
                                   U.contiguous(), W.contiguous(), dt, dt_scale, steps)
    else:
        # PyTorch fallback loop
        curr_x, curr_v = x, v
        for _ in range(steps):
             effective_dt = dt * dt_scale
             gamma_v = christoffel_fused(curr_v, U, W)
             curr_v = curr_v + effective_dt * ((f if f is not None else 0) - gamma_v)
             curr_x = curr_x + effective_dt * curr_v
        return curr_x, curr_v

def yoshida_fused(x, v, f, U, W, dt, dt_scale=1.0, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, steps=1):
    """
    Fused Yoshida 4th-order integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.yoshida_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                     U.contiguous(), W.contiguous(), V_w.contiguous() if V_w is not None else torch.empty(0, device=x.device),
                                     dt, dt_scale, plasticity, sing_thresh, sing_strength, steps)
    else:
        # Fallback would require complex Yoshida loop in Python
        return None

def parallel_scan_fused(a, x):
    """
    Fused Associative Scan for Parallel Geodesic Flow.
    """
    if CUDA_AVAILABLE and a.is_cuda:
         return gfn_cuda.parallel_scan_fused(a.contiguous(), x.contiguous())
    else:
         # Standard PyTorch implementation for Associative Scan
         # y_t = a_t * y_{t-1} + x_t
         # This is less efficient but biologically correct fallback
         from torch.distributions.utils import broadcast_all
         a, x = broadcast_all(a, x)
         y = torch.zeros_like(x)
         prev = torch.zeros_like(x[:, 0])
         for t in range(x.size(1)):
             y[:, t] = a[:, t] * prev + x[:, t]
             prev = y[:, t]
         return y
