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
                name='gfn_cuda',
                sources=[
                    os.path.join(cuda_dir, 'cuda_kernels.cpp'),
                    os.path.join(cuda_dir, 'kernels', 'christoffel_fused.cu'),
                    os.path.join(cuda_dir, 'kernels', 'leapfrog_fused.cu'),
                ],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-m64', '-ccbin', r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'],
                extra_cflags=['/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/Zc:twoPhase-'],
                verbose=True
            )
    
    CUDA_AVAILABLE = True
    print("[GFN CUDA] Custom kernels loaded successfully")
    
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
        # Check if C++ extension supports new signature (todo check capability)
        # For now assume we update C++ to match key-word args or strict position
        # We pass None as empty tensors if needed? 
        # C++ usually needs explicit tensors.
        if x is None:
            x = torch.empty(0, device=v.device, dtype=v.dtype)
        if V_w is None:
            V_w = torch.empty(0, device=v.device, dtype=v.dtype)
            
        return gfn_cuda.christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)
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
