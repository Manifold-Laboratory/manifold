
"""
Yoshida 4th Order Symplectic Integrator.
"""
import torch
import torch.nn as nn

class YoshidaIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
        # Yoshida Coefficients (4th order)
        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w0 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
        
        self.c1 = w1 / 2.0
        self.c2 = (w0 + w1) / 2.0
        self.c3 = self.c2
        self.c4 = self.c1
        
        self.d1 = w1
        self.d2 = w0
        self.d3 = w1

    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        if force is None:
             force = torch.zeros_like(x)
             
        # Try Professional Fused CUDA Kernel
        if x.is_cuda:
            try:
                from gfn.cuda.ops import yoshida_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    # Retrieve physics params from christoffel if it's a Reactive/Hyper manifold
                    plasticity = getattr(self.christoffel, 'plasticity', 0.0)
                    sing_thresh = getattr(self.christoffel, 'sing_thresh', 10.0)
                    sing_strength = getattr(self.christoffel, 'sing_strength', 20.0)
                    V_w = getattr(self.christoffel, 'V_w', None)
                    
                    # Logic matrices
                    U = getattr(self.christoffel, 'U', None)
                    W = getattr(self.christoffel, 'W', None)
                    
                    if U is not None and W is not None:
                        res = yoshida_fused(x, v, force, U, W, self.dt, dt_scale, 
                                           V_w=V_w, plasticity=plasticity, 
                                           sing_thresh=sing_thresh, sing_strength=sing_strength)
                        if res is not None:
                            return res
            except Exception:
                pass
        
        # Python Implementation
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
