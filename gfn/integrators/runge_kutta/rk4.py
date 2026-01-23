
"""
RK4 Integrator (Classic Runge-Kutta 4th Order).
"""
import torch
import torch.nn as nn

class RK4Integrator(nn.Module):
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
        
        # k2
        v2 = v + 0.5 * dt * dv1
        x2 = x + 0.5 * dt * dx1
        dx2 = v2
        dv2 = dynamics(x2, v2)
        
        # k3
        v3 = v + 0.5 * dt * dv2
        x3 = x + 0.5 * dt * dx2
        dx3 = v3
        dv3 = dynamics(x3, v3)
        
        # k4
        v4 = v + dt * dv3
        x4 = x + dt * dx3
        dx4 = v4
        dv4 = dynamics(x4, v4)
        
        # Update
        x_next = x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        v_next = v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        return x_next, v_next
