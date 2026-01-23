
"""
Leapfrog (Kick-Drift-Kick) Symplectic Integrator.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

class LeapfrogIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False):
        if force is None:
            force = torch.zeros_like(x)
            
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                # Logic matrices
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                
                if U is not None and W is not None:
                    return leapfrog_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps)
            except Exception:
                pass

        curr_x, curr_v = x, v
        for _ in range(steps):
            effective_dt = self.dt * dt_scale
            
            # Fallback to PyTorch
            # 1. Kick (half step velocity)
            gamma = self.christoffel(curr_v, curr_x)
            v_half = curr_v + 0.5 * effective_dt * (force - gamma)
            
            # 2. Drift (full step position)
            curr_x = curr_x + effective_dt * v_half
            
            # 3. Kick (half step velocity at new pos)
            gamma_half = self.christoffel(v_half, curr_x)
            curr_v = v_half + 0.5 * effective_dt * (force - gamma_half)
        
        return curr_x, curr_v
