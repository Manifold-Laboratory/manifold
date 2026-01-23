
"""
Leapfrog (Kick-Drift-Kick) Symplectic Integrator.
"""
import torch
import torch.nn as nn

class LeapfrogIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0):
        if force is None:
            force = torch.zeros_like(x)
            
        # Try CUDA kernel (Inference Only - requires float dt_scale)
        is_scalar_scale = isinstance(dt_scale, float) or (isinstance(dt_scale, torch.Tensor) and dt_scale.numel() == 1)
        
        if hasattr(self.christoffel, 'U') and hasattr(self.christoffel, 'W') and x.is_cuda:
             try:
                from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                     dt_val = dt_scale 
                     return leapfrog_fused(x, v, force, self.christoffel.U, self.christoffel.W, self.dt, dt_val)
             except ImportError:
                pass
             except Exception:
                pass

        effective_dt = self.dt * dt_scale
        
        # Fallback to PyTorch
        # 1. Kick (half step velocity)
        gamma = self.christoffel(v, x)
        v_half = v + 0.5 * effective_dt * (force - gamma)
        
        # 2. Drift (full step position)
        x_new = x + effective_dt * v_half
        
        # 3. Kick (half step velocity at new pos)
        gamma_half = self.christoffel(v_half, x_new)
        v_new = v_half + 0.5 * effective_dt * (force - gamma_half)
        
        return x_new, v_new
