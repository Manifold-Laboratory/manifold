
"""
Heun's Method (Improved Euler / RK2).
2nd order accuracy with only 2 evaluations per step.
Great balance between accuracy and speed.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import heun_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

class HeunIntegrator(nn.Module):
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
                    return heun_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps)
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
            
            # Predictor step (Euler)
            v_pred = curr_v + dt * dv1
            x_pred = curr_x + dt * dx1
            
            # k2 (using predicted velocity AND position)
            dx2 = v_pred
            dv2 = dynamics(x_pred, v_pred)
            
            # Corrector step
            curr_x = curr_x + (dt / 2.0) * (dx1 + dx2)
            curr_v = curr_v + (dt / 2.0) * (dv1 + dv2)
        
        return curr_x, curr_v
