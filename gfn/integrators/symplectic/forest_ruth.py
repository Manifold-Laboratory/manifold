
"""
Forest-Ruth 4th Order Symplectic Integrator.
Standard symplectic integrator, often considered an improved alternative to Yoshida 
for certain Hamiltonian systems due to different coefficient properties.

coeffs from: Forest, E., & Ruth, R. D. (1990). Fourth-order symplectic integration.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import forest_ruth_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

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

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    V_w = getattr(self.christoffel, 'V_w', torch.empty(0, device=x.device))
                    plasticity = getattr(self.christoffel, 'plasticity', 0.0)
                    sing_thresh = getattr(self.christoffel, 'sing_thresh', 1.0)
                    sing_strength = getattr(self.christoffel, 'sing_strength', 1.0)
                    return forest_ruth_fused(x, v, force, U, W, self.dt, dt_scale, V_w, plasticity, sing_thresh, sing_strength, steps=steps)
            except Exception:
                pass

        dt = self.dt * dt_scale
        
        if force is None:
             force = torch.zeros_like(x)
        
        curr_x, curr_v = x, v
        for _ in range(steps):
            def acceleration(tx, tv):
                 a = -self.christoffel(tv, tx)
                 if force is not None:
                     a = a + force
                 return a
                 
            # Step 1
            x1 = curr_x + self.c1 * dt * curr_v
            v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v) 
            
            # Step 2
            x2 = x1 + self.c2 * dt * v1
            v2 = v1 + self.d2 * dt * acceleration(x2, v1)
            
            # Step 3
            x3 = x2 + self.c3 * dt * v2
            v3 = v2 + self.d3 * dt * acceleration(x3, v2)
            
            # Step 4 (Final Drift)
            curr_x = x3 + self.c4 * dt * v3
            curr_v = v3
        
        return curr_x, curr_v
