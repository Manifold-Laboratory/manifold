
"""
Neural Symplectic Integrator.
Uses a learned controller to dynamically predict the optimal time-step (dt)
for each integration step, allowing the system to verify its own orbit
and correct drift by slowing down or speeding up.

Idea: x_{t+1} = x_t + v_t * NeuralNet(x_t, v_t)
"""
import torch
import torch.nn as nn

class NeuralIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01, dim=None):
        super().__init__()
        self.christoffel = christoffel
        self.base_dt = dt
        
        # Controller Network
        # Input: [x, v, force] -> 3 * dim
        # We need to handle force being None or present.
        if dim is not None:
            self.dim = dim
        elif hasattr(christoffel, 'dim'):
            self.dim = christoffel.dim
        else:
            self.dim = 16 # Fallback
        
        self.controller = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim), # Input: [x, v, f]
            nn.GELU(), # Better gradients than Tanh
            nn.Linear(self.dim, 1),
            nn.Softplus() # Strictly positive dt
        )
        
        # Initialize to output ~1.0 (neutral scaling)
        # Softplus(0.55) ~ 1.0
        nn.init.constant_(self.controller[2].bias, 0.55)
        nn.init.xavier_uniform_(self.controller[0].weight, gain=0.1) # low gain to start neutral

    def forward(self, x, v, force=None, dt_scale=1.0):
        # 0. Handle Force
        if force is None:
            f_in = torch.zeros_like(x)
        else:
            f_in = force

        # 1. Predict Dynamic DT
        # We concatenate x, v, AND force. 
        # "If I'm moving fast (v) towards a wall (x) with a push (f), I need a tiny step."
        state = torch.cat([x, v, f_in], dim=-1)
        
        # Learned scale: Range (0, Infinity), centered around 1.0
        # We add a small epsilon to prevent time stopping completely (Zeno's paradox)
        learned_scale = self.controller(state) + 0.1 
        
        # dynamic_dt = base_dt * external_wormhole * learned_internal_judgment
        dynamics_dt = self.base_dt * dt_scale * learned_scale
        
        # 2. Symplectic Step (Leapfrog-like using dynamic dt)
        
        # Half-step Kick
        acc = -self.christoffel(v, x)
        if force is not None:
            acc = acc + force
            
        v_half = v + 0.5 * dynamics_dt * acc
        
        # Full-step Drift
        x_next = x + dynamics_dt * v_half
        
        # Half-step Kick (at new pos)
        acc_next = -self.christoffel(v_half, x_next)
        if force is not None:
            acc_next = acc_next + force
            
        v_next = v_half + 0.5 * dynamics_dt * acc_next
        
        return x_next, v_next
