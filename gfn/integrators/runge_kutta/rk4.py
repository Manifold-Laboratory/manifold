
"""
RK4 Integrator (Classic Runge-Kutta 4th Order).
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import rk4_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

class RK4Integrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    return rk4_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps)
            except Exception:
                pass

        curr_x, curr_v = x, v
        for _ in range(steps):
            dt = self.dt * dt_scale
            
            def dynamics(current_x, current_v):
                acc = -self.christoffel(current_v, current_x)
                if force is not None:
                    acc = acc + force
                return acc
                
            # k1
            dx1 = curr_v
            dv1 = dynamics(curr_x, curr_v)
            
            # k2
            v2 = curr_v + 0.5 * dt * dv1
            x2 = curr_x + 0.5 * dt * dx1
            dx2 = v2
            dv2 = dynamics(x2, v2)
            
            # k3
            v3 = curr_v + 0.5 * dt * dv2
            x3 = curr_x + 0.5 * dt * dx2
            dx3 = v3
            dv3 = dynamics(x3, v3)
            
            # k4
            v4 = curr_v + dt * dv3
            x4 = curr_x + dt * dx3
            dx4 = v4
            dv4 = dynamics(x4, v4)
            
            # Update
            curr_x = curr_x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
            curr_v = curr_v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        return curr_x, curr_v
