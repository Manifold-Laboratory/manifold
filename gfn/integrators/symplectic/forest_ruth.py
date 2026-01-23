
"""
Forest-Ruth 4th Order Symplectic Integrator.
Standard symplectic integrator, often considered an improved alternative to Yoshida 
for certain Hamiltonian systems due to different coefficient properties.

coeffs from: Forest, E., & Ruth, R. D. (1990). Fourth-order symplectic integration.
"""
import torch
import torch.nn as nn

class ForestRuthIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
        # Forest-Ruth coefficients
        theta = 1.0 / (2.0 - 2.0**(1.0/3.0))
        
        self.c1 = theta / 2.0
        self.c2 = (1.0 - theta) / 2.0
        self.c3 = (1.0 - theta) / 2.0
        self.c4 = theta / 2.0
        
        self.d1 = theta
        self.d2 = 1.0 - 2.0*theta
        self.d3 = theta

    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        if force is None:
             force = torch.zeros_like(x)
        
        def acceleration(tx, tv):
             a = -self.christoffel(tv, tx)
             if force is not None:
                 a = a + force
             return a
             
        # Step 1
        x1 = x + self.c1 * dt * v
        v1 = v + self.d1 * dt * acceleration(x1, v) 
        
        # Step 2
        x2 = x1 + self.c2 * dt * v1
        v2 = v1 + self.d2 * dt * acceleration(x2, v1)
        
        # Step 3
        x3 = x2 + self.c3 * dt * v2
        v3 = v2 + self.d3 * dt * acceleration(x3, v2)
        
        # Step 4 (Final Drift)
        x_final = x3 + self.c4 * dt * v3
        
        return x_final, v3
