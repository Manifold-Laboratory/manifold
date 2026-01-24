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
                name='gfn_cuda_v5_0',
                sources=[
                    os.path.join(cuda_dir, 'cuda_kernels.cpp'),
                    os.path.join(cuda_dir, 'src', 'geometry', 'christoffel_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'geometry', 'lowrank_christoffel.cu'),
                    os.path.join(cuda_dir, 'src', 'geometry', 'reactive_christoffel.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'leapfrog_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'euler_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'heun_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'rk4_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'yoshida_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'forest_ruth_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'omelyan_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'verlet_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'dormand_prince_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'recurrent_manifold_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'recurrent_manifold_backward.cu'),
                    os.path.join(cuda_dir, 'src', 'integrators', 'parallel_scan_fused.cu'),
                    os.path.join(cuda_dir, 'src', 'layers', 'manifold_step.cu'),
                    os.path.join(cuda_dir, 'src', 'layers', 'head_mixing.cu'),
                    os.path.join(cuda_dir, 'src', 'layers', 'dynamic_gating.cu'),
                ],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-m64', '-ccbin', r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'],
                extra_cflags=['/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/Zc:twoPhase-', '/I' + os.path.join(cuda_dir, 'include')],
                verbose=True
             )
             print(f"[GFN CUDA] JIT Compilation v5_0 SUCCESS.")
        except Exception as e:
            print(f"[GFN CUDA] JIT Compilation FAILED: {e}")
            CUDA_AVAILABLE = False
    
    CUDA_AVAILABLE = True
    print("[GFN CUDA] Professional kernels loaded successfully (Autograd, Parallel Scan, Trajectory Fusion)")
    
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

def lowrank_christoffel_fused(v, U, W):
    """
    MODULAR: Pure Low-Rank Christoffel geometry.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        from .autograd import lowrank_christoffel_autograd
        res = lowrank_christoffel_autograd(v, U, W)
        if res is not None:
            return res
            
    # Fallback to standard logic (simpler version of the one above)
    proj = torch.matmul(v, U)
    norm = torch.norm(proj, dim=-1, keepdim=True)
    scale = 1.0 / (1.0 + norm)
    sq = (proj * proj) * scale
    return torch.matmul(sq, W.t())

def reactive_christoffel_fused(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    """
    MODULAR: Active Inference Christoffel geometry.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        from .autograd import reactive_christoffel_autograd
        res = reactive_christoffel_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)
        if res is not None:
            return res
            
    # Fallback uses the monolithic logic already present in ops.py
    return christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)


def leapfrog_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused Leapfrog integration step. (Supports Recurrent Fusion if steps > 1)
    Uses autograd-enabled backward pass during training.
    """
    if CUDA_AVAILABLE and x.is_cuda:
        # Use autograd version if gradients are needed
        if x.requires_grad or U.requires_grad or W.requires_grad:
            from .autograd import leapfrog_fused_autograd
            result = leapfrog_fused_autograd(x, v, f, U, W, dt, dt_scale, steps)
            if result is not None:
                return result
            # If autograd failed but we need grads, DO NOT use raw kernel. 
            # Return None to trigger Python fallback in Integrator.
            return None
        
        # Inference-only path (no gradients)
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

def verlet_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused Verlet integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.verlet_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                    U.contiguous(), W.contiguous(), dt, dt_scale, steps)
    else:
        # Fallback loop
        curr_x, curr_v = x, v
        for _ in range(steps):
            eff_dt = dt * dt_scale
            gamma = christoffel_fused(curr_v, U, W)
            a = (f if f is not None else 0.0) - gamma
            x_half = curr_x + 0.5 * eff_dt * curr_v
            curr_x = x_half + 0.5 * eff_dt * (curr_v + eff_dt * a) # Approx
            curr_v = curr_v + eff_dt * a
        return curr_x, curr_v

def forest_ruth_fused(x, v, f, U, W, dt, dt_scale=1.0, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, steps=1):
    """
    Fused Forest-Ruth 4th-order integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.forest_ruth_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                         U.contiguous(), W.contiguous(), V_w.contiguous() if V_w is not None else torch.empty(0, device=x.device),
                                         dt, dt_scale, plasticity, sing_thresh, sing_strength, steps)
    else:
        return None

def omelyan_fused(x, v, f, U, W, dt, dt_scale=1.0, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, steps=1):
    """
    Fused Omelyan optimized 2nd-order integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.omelyan_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                     U.contiguous(), W.contiguous(), V_w.contiguous() if V_w is not None else torch.empty(0, device=x.device),
                                     dt, dt_scale, plasticity, sing_thresh, sing_strength, steps)
    else:
        return None

def heun_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused Heun 2nd-order integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.heun_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                  U.contiguous(), W.contiguous(), dt, dt_scale, steps)
    else:
        return None

def rk4_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused RK4 4th-order integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.rk4_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                 U.contiguous(), W.contiguous(), dt, dt_scale, steps)
    else:
        return None

def dormand_prince_fused(x, v, f, U, W, dt, dt_scale=1.0, steps=1):
    """
    Fused Dormand-Prince 5th-order integration step. (Supports Recurrent Fusion if steps > 1)
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.dormand_prince_fused(x.contiguous(), v.contiguous(), f.contiguous() if f is not None else torch.zeros_like(x),
                                            U.contiguous(), W.contiguous(), dt, dt_scale, steps)
    else:
        return None

def parallel_scan_fused(a, x, plasticity=0.0):
    """
    Fused Associative Scan with Active Inference.
    """
    if CUDA_AVAILABLE and a.is_cuda:
         return gfn_cuda.parallel_scan_fused(a.contiguous(), x.contiguous(), plasticity)
    else:
         # Standard PyTorch implementation for Associative Scan
         # y_t = a_t * y_{t-1} + x_t
         from torch.distributions.utils import broadcast_all
         a, x = broadcast_all(a, x)
         y = torch.zeros_like(x)
         prev = torch.zeros_like(x[:, 0])
         for t in range(x.size(1)):
             y[:, t] = a[:, t] * prev + x[:, t]
             prev = y[:, t]
         return y

def manifold_step_fused(x, v, force, U, W, dt, dt_scale=1.0, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, return_christoffel=False):
    """
    Fused Manifold Step (Time-Distributed, Layer-Atomic).
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.manifold_step_fused(
            x.contiguous(), v.contiguous(), force.contiguous(),
            U.contiguous(), W.contiguous(),
            dt, dt_scale, plasticity, sing_thresh, sing_strength, return_christoffel
        )
    return None

def head_mixing_fused(x_heads, v_heads, W_x, W_v):
    """
    Inter-head mixing via linear projections.
    
    Args:
        x_heads: [H, B, D/H] position states per head
        v_heads: [H, B, D/H] velocity states per head
        W_x: [D, D] mixing weight for positions
        W_v: [D, D] mixing weight for velocities
    
    Returns:
        x_out: [B, D] mixed positions
        v_out: [B, D] mixed velocities
    """
    if CUDA_AVAILABLE and x_heads.is_cuda:
        return gfn_cuda.head_mixing_fused(
            x_heads.contiguous(), v_heads.contiguous(),
            W_x.contiguous(), W_v.contiguous()
        )
    
    # Fallback PyTorch
    H, B, D_H = x_heads.shape
    D = H * D_H
    x_concat = x_heads.transpose(0, 1).reshape(B, D)  # [B, D]
    v_concat = v_heads.transpose(0, 1).reshape(B, D)
    x_out = torch.matmul(x_concat, W_x.t())
    v_out = torch.matmul(v_concat, W_v.t())
    return x_out, v_out

def dynamic_gating_fused(x, W1, b1, W2, b2):
    """
    Dynamic time-step gating via curvature estimation.
    
    Args:
        x: [B, D] input state
        W1: [D/4, D] first layer weight
        b1: [D/4] first layer bias
        W2: [1, D/4] second layer weight  
        b2: [1] second layer bias
    
    Returns:
        dt_scale: [B, 1] gating factor in [0, 1]
    """
    if CUDA_AVAILABLE and x.is_cuda:
        return gfn_cuda.dynamic_gating_fused(
            x.contiguous(),
            W1.contiguous(), b1.contiguous(),
            W2.contiguous(), b2.contiguous()
        )
    
    # Fallback PyTorch
    hidden = torch.tanh(torch.matmul(x, W1.t()) + b1)
    out = torch.sigmoid(torch.matmul(hidden, W2.t()) + b2)
    return out

def recurrent_manifold_fused(x_state, v_state, forces, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads=1,
                            plasticity=0.0, sing_thresh=1.0, sing_strength=1.0,
                            mix_x=None, mix_v=None):
    if not CUDA_AVAILABLE or not x_state.is_cuda: return None
    if not isinstance(dt_scales, torch.Tensor):
        dt_scales = torch.tensor([float(dt_scales)]*num_heads, device=x_state.device, dtype=torch.float32)
    if not isinstance(forget_rates, torch.Tensor):
        forget_rates = torch.tensor([float(forget_rates)]*num_heads, device=x_state.device, dtype=torch.float32)
        
    return gfn_cuda.recurrent_manifold_fused(
        x_state.contiguous(), v_state.contiguous(), 
        forces.contiguous(), U_stack.contiguous(), W_stack.contiguous(), 
        dt, dt_scales, forget_rates, num_heads,
        plasticity, sing_thresh, sing_strength,
        mix_x if mix_x is not None else torch.empty(0, x_state.options()),
        mix_v if mix_v is not None else torch.empty(0, x_state.options())
    )

def recurrent_manifold_backward(grad_x_seq, grad_x_final, grad_v_final, x_final, v_final, forces, U_stack, W_stack,
                               dt, dt_scales, forget_rates, num_heads=1,
                               plasticity=0.0, sing_thresh=1.0, sing_strength=1.0,
                               mix_x=None, mix_v=None):
    if not CUDA_AVAILABLE or not x_final.is_cuda: return None
    if not isinstance(dt_scales, torch.Tensor):
        dt_scales = torch.tensor([float(dt_scales)]*num_heads, device=x_final.device, dtype=torch.float32)
    if not isinstance(forget_rates, torch.Tensor):
        forget_rates = torch.tensor([float(forget_rates)]*num_heads, device=x_final.device, dtype=torch.float32)
        
    return gfn_cuda.recurrent_manifold_backward(
        grad_x_seq, grad_x_final, grad_v_final, x_final, v_final, 
        forces.contiguous(), U_stack.contiguous(), W_stack.contiguous(), 
        dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength,
        mix_x if mix_x is not None else torch.empty(0, x_final.options()),
        mix_v if mix_v is not None else torch.empty(0, x_final.options())
    )
