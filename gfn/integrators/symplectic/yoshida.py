
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
             
        # Try Fused CUDA Kernel (Supports Training via Autograd now!)
        if hasattr(self.christoffel, 'U') and hasattr(self.christoffel, 'W') and x.is_cuda:
            try:
                from gfn.cuda.ops import yoshida_fused, CUDA_AVAILABLE
                # Physics Hyperparameters for Active Inference (Defaults)
                plasticity = 0.0 # Default off unless specified
                sing_thresh = 10.0
                sing_strength = 20.0
                
                # Check if christoffel has these attributes active
                if hasattr(self.christoffel, 'plasticity_mask'):
                    # This is just a heuristic, ideally we pass the full config
                    pass
                    
                # We need U, V_w (W) to be available. 
                # Assuming LowRankChristoffel stores W.
                U = self.christoffel.U
                W = self.christoffel.W
                
                if CUDA_AVAILABLE:
                    return yoshida_fused(x, v, force, U, W, self.dt, dt_scale)
            except ImportError:
                pass
            except Exception as e:
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
