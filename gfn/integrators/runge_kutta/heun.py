
"""
Heun's Method (Improved Euler / RK2).
2nd order accuracy with only 2 evaluations per step.
Great balance between accuracy and speed.
"""
import torch
import torch.nn as nn

class HeunIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        def dynamics(current_x, current_v):
            acc = -self.christoffel(current_v, current_x)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        dx1 = v
        dv1 = dynamics(x, v)
        
        # Predictor step (Euler)
        v_pred = v + dt * dv1
        x_pred = x + dt * dx1
        
        # k2 (using predicted velocity AND position)
        dx2 = v_pred
        dv2 = dynamics(x_pred, v_pred)
        
        # Corrector step
        x_next = x + (dt / 2.0) * (dx1 + dx2)
        v_next = v + (dt / 2.0) * (dv1 + dv2)
        
        return x_next, v_next
