
"""
Symplectic Integrator (Velocity Verlet).
Kept separate for historical continuity and as a standard baseline.
"""
import torch
import torch.nn as nn

class SymplecticIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False):
        # Try Professional Fused CUDA Kernel
        if x.is_cuda and not collect_christ:
            try:
                from gfn.cuda.ops import verlet_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    U = getattr(self.christoffel, 'U', None)
                    W = getattr(self.christoffel, 'W', None)
                    if U is not None and W is not None:
                        return verlet_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps)
            except Exception:
                pass

        for _ in range(steps):
            dt = self.dt * dt_scale
            
            # Compute acceleration at current state
            gamma = self.christoffel(v, x)
            
            if force is None:
                a = -gamma
            else:
                a = -gamma + force
                
            # Velocity Verlet Step 1: v(t+0.5*dt)
            v_half = v + 0.5 * dt * a
            
            # Step 2: x(t+dt)
            x = x + dt * v_half
            
            # Re-compute acceleration at x_next
            gamma_next = self.christoffel(v_half, x)
            if force is None:
                a_next = -gamma_next
            else:
                a_next = -gamma_next + force
                
            v = v_half + 0.5 * dt * a_next
        
        return x, v
