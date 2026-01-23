
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
            
        # Try Professional Fused CUDA Kernel
        if x.is_cuda:
            try:
                from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    # Logic matrices
                    U = getattr(self.christoffel, 'U', None)
                    W = getattr(self.christoffel, 'W', None)
                    
                    if U is not None and W is not None:
                        return leapfrog_fused(x, v, force, U, W, self.dt, dt_scale)
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
