
"""
Euler Integrator (1st Order).
The simplest explicit integrator. 
Useful as a baseline to demonstrate the instability of low-order non-symplectic methods.
"""
import torch
import torch.nn as nn

class EulerIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        acc = -self.christoffel(v, x)
        if force is not None:
            acc = acc + force
            
        x_next = x + dt * v
        v_next = v + dt * acc
        
        return x_next, v_next
