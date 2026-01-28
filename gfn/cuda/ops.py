
import torch
import torch.nn as nn
import os
import sys
import importlib.util
import importlib.machinery
from pathlib import Path

# CUDA op loading and Python fallbacks

CUDA_AVAILABLE = False
gfn_cuda = None

def get_cuda_path():
    return os.path.dirname(os.path.abspath(__file__))

# Specialized attempt to load/import the gfn_cuda module
_CUDA_LOG_ONCE = False

def _log_cuda_status():
    global _CUDA_LOG_ONCE
    if not _CUDA_LOG_ONCE:
        if CUDA_AVAILABLE:
            print(f"[GFN] CUDA enabled: {torch.cuda.get_device_name(0)}")
        else:
            print("[GFN] CUDA disabled: using Python fallbacks")
        _CUDA_LOG_ONCE = True

def _prepare_dll_paths():
    try:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass
    for ver in ["v12.9", "v12.4", "v12.3", "v11.8"]:
        p = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass

cuda_dir = Path(__file__).resolve().parent
project_root = cuda_dir.parent.parent
if str(cuda_dir) not in sys.path:
    sys.path.insert(0, str(cuda_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def _load_local_gfn_cuda():
    global gfn_cuda, CUDA_AVAILABLE
    _prepare_dll_paths()
    candidates = list(cuda_dir.glob("gfn_cuda*.pyd")) + list(project_root.glob("gfn_cuda*.pyd"))
    for path in candidates:
        try:
            loader = importlib.machinery.ExtensionFileLoader("gfn_cuda", str(path))
            spec = importlib.util.spec_from_file_location("gfn_cuda", str(path), loader=loader)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules["gfn_cuda"] = module
            gfn_cuda = module
            CUDA_AVAILABLE = True
            return True
        except Exception:
            continue
    return False

try:
    _prepare_dll_paths()
    import gfn_cuda
    CUDA_AVAILABLE = True
except ImportError:
    try:
        _prepare_dll_paths()
        from . import gfn_cuda
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = _load_local_gfn_cuda()

# Initialize status on first import if possible, or wait for ops
if CUDA_AVAILABLE:
    _log_cuda_status()

def christoffel_fused(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
    """
    Christoffel projection with optional plasticity and periodic features.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        from .autograd import christoffel_fused_autograd
        return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
    
    # Python fallback (vectorized)
    # print("[GFN:WARN] Fallback to Python implementation for Christoffel Fused")
    # h = U^T v
    h = torch.matmul(v, U) # [B, R]
    energy = torch.sum(h*h, dim=-1, keepdim=True)
    S = 1.0 / (1.0 + torch.sqrt(energy) + 1e-6)
    
    M = 1.0
    if plasticity != 0.0:
        E = torch.sum(v*v, dim=-1, keepdim=True) / v.shape[-1]
        M *= (1.0 + plasticity * torch.tanh(E))
    
    if x is not None and V_w is not None:
        # Periodic singularity logic in Python fallback
        if topology == 1: pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
        else: pot = torch.sum(x * V_w, dim=-1, keepdim=True)
        gate = torch.sigmoid(pot)
        soft_m = torch.sigmoid(10.0 * (gate - sing_thresh))
        M = M * (1.0 + (sing_strength - 1.0) * soft_m)

    # Python fallback uses matrix formulation for both topologies
    gamma = torch.matmul(h*h, W.t()) * S * M
    return 20.0 * torch.tanh(gamma / 20.0)

def reactive_christoffel(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
    """
    Geometry dispatcher: tries reactive kernel, falls back to christoffel_fused.
    """
    if CUDA_AVAILABLE and v.is_cuda and v.dim() == 2:
        from .autograd import reactive_christoffel_autograd
        res = reactive_christoffel_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        if res is not None: return res
    
    return christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)

def leapfrog_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, Wf=None, bf=None, plasticity=0.0, R=2.0, r=1.0):
    """
    Fused symplectic integrator step with Python fallback.
    """
    if CUDA_AVAILABLE and x.is_cuda and x.dim() == 2:
        from .autograd import leapfrog_fused_autograd
        # Ensure dt_scale is a Tensor for C++ binding
        if not isinstance(dt_scale, torch.Tensor):
            dt_scale = torch.tensor(float(dt_scale), device=x.device, dtype=x.dtype)
            
        res = leapfrog_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology=topology, Wf=Wf, bf=bf, plasticity=plasticity, R=R, r=r)
        if res is not None: return res
    
    # Python fallback (vectorized)
    eff_dt = dt * (float(dt_scale) if not isinstance(dt_scale, torch.Tensor) else dt_scale)
    h = 0.5 * eff_dt
    curr_x, curr_v = x, v
    
    for _ in range(steps):
        # 1. Friction coefficient
        mu = torch.zeros_like(v)
        if Wf is not None and bf is not None:
             feat = curr_x
             if topology == 1: # Torus
                  feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
             gate = torch.matmul(feat, Wf.t()) + bf
             mu = torch.sigmoid(gate) * 5.0 # Max 5.0 dampen
             
        # 2. Kick 1 (Half Step) with implicit friction
        gamma = christoffel_fused(curr_v, U, W, curr_x, None, plasticity, 1.0, 1.0, topology, R, r)
        force_val = f if f is not None else 0.0
        # Implicit update: v_next = (v_prev + h*(F - gamma)) / (1 + h*mu)
        curr_v = (curr_v + h * (force_val - gamma)) / (1.0 + h * mu)
        
        # 3. Drift (Full Step)
        curr_x = curr_x + eff_dt * curr_v
        if topology == 1:
             from gfn.geometry.boundaries import apply_boundary_python
             curr_x = apply_boundary_python(curr_x, 1)
             
        # 4. Kick 2 (Half Step) with implicit friction at new position
        if Wf is not None and bf is not None:
             feat = curr_x
             if topology == 1:
                  feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
             gate2 = torch.matmul(feat, Wf.t()) + bf
             mu = torch.sigmoid(gate2) * 5.0

        gamma2 = christoffel_fused(curr_v, U, W, curr_x, None, plasticity, 1.0, 1.0, topology, R, r)
        curr_v = (curr_v + h * (force_val - gamma2)) / (1.0 + h * mu)
        
    return curr_x, curr_v

def euler_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if CUDA_AVAILABLE and x.is_cuda:
        from .autograd import euler_fused_autograd
        return euler_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)
    return None

def heun_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if CUDA_AVAILABLE and x.is_cuda:
        from .autograd import heun_fused_autograd
        return heun_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)
    return None

def rk4_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if CUDA_AVAILABLE and x.is_cuda:
        from .autograd import rk4_fused_autograd
        return rk4_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)
    return None

def verlet_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if CUDA_AVAILABLE and x.is_cuda:
        from .autograd import verlet_fused_autograd
        return verlet_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)
    return None

def yoshida_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    # Yoshida 4th Order Symplectic
    if CUDA_AVAILABLE and x.is_cuda:
        pass
    return None

def forest_ruth_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if CUDA_AVAILABLE and x.is_cuda:
        pass
    return None

def dormand_prince_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if CUDA_AVAILABLE and x.is_cuda:
        pass
    return None

def recurrent_manifold_fused(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads, 
                             plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, 
                             mix_x=None, mix_v=None, Wf=None, Wi=None, bf=None, 
                             Wp=None, bp=None, # NEW ARGS
                             topology=0, R=2.0, r=1.0):
    """
    Fused recurrent manifold step for sequence training.
    """
    if CUDA_AVAILABLE and x.is_cuda: # Enabled for Torus (Fixed kernels)
        from .autograd import recurrent_manifold_fused_autograd
        return recurrent_manifold_fused_autograd(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x, mix_v, Wf, Wi, bf, Wp, bp, topology, R, r)
    
    return None # Use Python Sequence Loop (Autograd managed)
