
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

    def forward(self, x, v, force=None, dt_scale=1.0):
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
        x_next = x + dt * v_half
        
        # Re-compute acceleration at x_next
        gamma_next = self.christoffel(v_half, x_next)
        if force is None:
            a_next = -gamma_next
        else:
            a_next = -gamma_next + force
            
        v_next = v_half + 0.5 * dt * a_next
        
        return x_next, v_next
