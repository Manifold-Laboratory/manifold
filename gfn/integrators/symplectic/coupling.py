
"""
Coupling Flow Integrator ("The Perfect Integrator").
Based on Normalizing Flows (NICE/RealNVP).
Uses separate coupling layers for Position and Velocity to guarantee exactly geometric volume preservation.
Jacobian Determinant is strictly 1.0.

Structure:
    v' = v + F(x)  (Shear transformation on v)
    x' = x + G(v') (Shear transformation on x)
    
This requires F(x) to be INDEPENDENT of v.
Standard Christoffel symbols \Gamma(v, x) are quadratic in v.
To enforce "Perfect" symplectic behavior (separable Hamiltonian logic),
we approximate the force F(x) by evaluating \Gamma at v=0 (or a learned proxy).
"""
import torch
import torch.nn as nn

class CouplingFlowIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
        # "Drift" Network (The Warper)
        # In standard physics, x' = x + v*dt. This is linear drift.
        # In a coupling flow, we can use x' = x + G(v).
        # We learn a small residual MLP to warp space-time based on velocity.
        # This makes the "mass" effective dynamic (Special Relativity vibe).
        # Automatic dimension discovery
        if hasattr(christoffel, 'dim'):
            self.dim = christoffel.dim
        else:
            self.dim = 16 # Fallback
             
        self.drift_net = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.dim)
        )
        # Init as near-zero to start with standard kinematics
        nn.init.zeros_(self.drift_net[2].weight)
        nn.init.zeros_(self.drift_net[2].bias)

    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        if force is None:
            force = torch.zeros_like(x)
            
        # We implement a Symmetric Splitting (Strang Splitting) for higher order accuracy
        # Kick - Drift - Kick (Verlet/Leapfrog structure) but with generalized networks.
        # This ensures time-reversibility and higher stability.
        
        # 1. KICK (Half Step)
        # v_half = v + 0.5 * dt * F(x)
        # Note: We must avoid v-dependence in F(x) to keep Jacobian=1
        v_dummy = torch.zeros_like(x)
        acc_1 = -self.christoffel(v_dummy, x) + force
        v_half = v + 0.5 * dt * acc_1
        
        # 2. DRIFT (Full Step)
        # x_new = x + dt * G(v_half)
        # G(v) = v + drift_net(v)
        # This allows the integrator to learn "Warp Drive" (non-linear displacement)
        warp = self.drift_net(v_half)
        x_new = x + dt * (v_half + warp)
        
        # 3. KICK (Half Step)
        # v_new = v_half + 0.5 * dt * F(x_new)
        acc_2 = -self.christoffel(v_dummy, x_new) + force # Force assumed constant or position dependent
        v_new = v_half + 0.5 * dt * acc_2
        
        # Because each step is a shear transformation ( triangular Jacobian with 1s on diagonal),
        # the total determinant is 1.0 * 1.0 * 1.0 = 1.0.
        # Exact Volume Preservation confirmed.
        
        return x_new, v_new
