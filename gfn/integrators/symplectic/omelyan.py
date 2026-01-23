
"""
Omelyan PEFRL (Position Extended Forest-Ruth Like) Integrator.
4th Order Symplectic Integrator optimized for Hamiltonian systems.
Has approx 100x better error constants than standard Forest-Ruth/Yoshida for some potentials.
"""
import torch
import torch.nn as nn

class OmelyanIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
        # Omelyan PEFRL coefficients
        # xi = 0.1786178958448091
        # lambda = -0.2123418310626054
        # chi = -0.06626458266981849
        
        self.xi = 0.1786178958448091
        self.lam = -0.2123418310626054
        self.chi = -0.06626458266981849
        
        # Derived weights for the steps
        self.c1 = self.xi
        self.c2 = self.chi
        self.c3 = 1.0 - 2.0*(self.chi + self.xi)
        self.c4 = self.chi
        self.c5 = self.xi
        
        self.d1 = (1.0 - 2.0*self.lam) / 2.0
        self.d2 = self.lam
        self.d3 = self.lam
        self.d4 = (1.0 - 2.0*self.lam) / 2.0

    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        if force is None:
             force = torch.zeros_like(x)
        
        def acceleration(tx, tv):
             a = -self.christoffel(tv, tx)
             if force is not None:
                 a = a + force
             return a
        
        # Omelyan PEFRL Structure:
        # x1 = x + xi*dt*v
        # v1 = v + (1-2*lam)/2 * dt * a(x1)
        # x2 = x1 + chi*dt*v1
        # v2 = v1 + lam*dt * a(x2)
        # x3 = x2 + (1-2(chi+xi))*dt*v2
        # v3 = v2 + lam*dt * a(x3)
        # x4 = x3 + chi*dt*v3
        # v4 = v3 + (1-2*lam)/2 * dt * a(x4)
        # x5 = x4 + xi*dt*v4
        
        # Step 1
        x1 = x + self.c1 * dt * v
        v1 = v + self.d1 * dt * acceleration(x1, v)
        
        # Step 2
        x2 = x1 + self.c2 * dt * v1
        v2 = v1 + self.d2 * dt * acceleration(x2, v1)
        
        # Step 3
        x3 = x2 + self.c3 * dt * v2
        v3 = v2 + self.d3 * dt * acceleration(x3, v2)
        
        # Step 4
        x4 = x3 + self.c4 * dt * v3
        v4 = v3 + self.d4 * dt * acceleration(x4, v3)
        
        # Step 5 (Final Drift)
        x_final = x4 + self.c5 * dt * v4
        
        return x_final, v4
