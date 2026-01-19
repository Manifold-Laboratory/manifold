"""
GFN Adjoint State Method
========================

Implements Neural ODE-style backpropagation for O(1) memory training.
Instead of storing intermediate states, we solve the adjoint ODE backward.

Requires: pip install torchdiffeq
"""

import torch
import torch.nn as nn

# Try to import torchdiffeq, but provide fallback
try:
    from torchdiffeq import odeint_adjoint, odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not installed. Adjoint method unavailable.")
    print("Install with: pip install torchdiffeq")


class GeodesicODEFunc(nn.Module):
    """
    ODE function for geodesic flow with external force.
    
    State: [x, v, f] concatenated along last dimension
    Dynamics:
        dx/dt = v
        dv/dt = f - Γ(v, v)
        df/dt = 0  (Force is constant during integration step)
    """
    
    def __init__(self, christoffel_net):
        super().__init__()
        self.christoffel = christoffel_net
        self.dim = None
    
    def forward(self, t, state):
        """
        Compute derivatives for state = [x, v, f].
        """
        if self.dim is None:
            self.dim = state.shape[-1] // 3
        
        dim = self.dim
        x = state[..., :dim]
        v = state[..., dim:2*dim]
        f = state[..., 2*dim:]
        
        # dx/dt = v
        dx_dt = v
        
        # dv/dt = f - Γ(v, v)
        dv_dt = f - self.christoffel(v)
        
        # df/dt = 0
        df_dt = torch.zeros_like(f)
        
        return torch.cat([dx_dt, dv_dt, df_dt], dim=-1)


class AdjointMLayer(nn.Module):
    """
    Manifold Layer using Adjoint State Method.
    
    Uses Neural ODE integration with adjoint backpropagation
    for O(1) memory complexity regardless of integration steps.
    """
    
    def __init__(self, dim, rank=16, integration_time=1.0, n_steps=10):
        super().__init__()
        from .geometry import LowRankChristoffel
        
        self.christoffel = LowRankChristoffel(dim, rank)
        self.ode_func = GeodesicODEFunc(self.christoffel)
        
        self.integration_time = integration_time
        # ... (rest of init)

class AdjointManifold(nn.Module):
    """
    Full MANIFOLD model using Adjoint State Method for O(1) memory.
    
    This version uses continuous ODE integration instead of
    discrete layer-by-layer updates, providing:
    - O(1) memory for backpropagation
    - Smoother gradients
    - Better numerical stability for deep networks
    """
    
    def __init__(self, vocab_size, dim=256, depth=4, rank=32, heads=1, integration_time=1.0):
        super().__init__()
        
        if heads > 1:
            raise NotImplementedError("AdjointManifold currently only supports heads=1. "
                                      "For Multi-Head Geodesic Flows, use standard Manifold (use_adjoint=False).")
                                      
        self.dim = dim
        self.depth = depth
        
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Use adjoint layers
        self.layers = nn.ModuleList([
            AdjointMLayer(dim, rank=rank, integration_time=integration_time/depth, n_steps=5)
            for _ in range(depth)
        ])
        
        # Pre-LN (consistent with V2)
        # AdjointGLayer dynamics are continuous, but we can apply LN between layers
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        
        self.readout_norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, vocab_size)
        
        # Improved Initialization (Consistent with GFN V2)
        self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)
        
        # Init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, state=None):
        batch_size, seq_len = input_ids.shape
        
        if state is None:
            x = self.x0.expand(batch_size, -1)
            v = self.v0.expand(batch_size, -1)
        else:
            x, v = state
        
        all_forces = self.embedding(input_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
        else:
            mask = torch.ones(batch_size, seq_len, 1, device=input_ids.device)
        
        logits_list = []
        
        for t in range(seq_len):
            force = all_forces[:, t] * mask[:, t]
            
            for i, layer in enumerate(self.layers):
                # Pre-LN
                x = self.norms[i](x)
                v = self.norms[i](v)
                
                x, v = layer(x, v, force)
            
            out = self.readout_norm(x)
            logit = self.readout(out)
            logits_list.append(logit.unsqueeze(1))
        
        logits = torch.cat(logits_list, dim=1)
        return logits, (x, v)

    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressive generation with sampling.
        """
        self.eval()
        device = prompt_ids.device
        
        # Process prompt
        logits, state = self(prompt_ids)
        
        # Start generation
        generated = prompt_ids.tolist()[0]
        
        def sample_next(logits, temp=1.0, k=None, p=None):
            # Last timestep logits
            next_logit = logits[:, -1, :] / temp
            
            # Top-K
            if k is not None:
                v, _ = torch.topk(next_logit, k)
                next_logit[next_logit < v[:, [-1]]] = -float('Inf')
            
            # Top-P (Nucleus)
            if p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logit[indices_to_remove] = -float('Inf')
            
            # Sample
            if k is None and p is None:
                return torch.argmax(next_logit, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logit, dim=-1)
                return torch.multinomial(probs, num_samples=1)

        # Initial sample
        curr_token = sample_next(logits, temperature, top_k, top_p)
        generated.append(curr_token.item())
        
        for _ in range(max_new_tokens - 1):
            logits, state = self(curr_token, state=state)
            curr_token = sample_next(logits, temperature, top_k, top_p)
            generated.append(curr_token.item())
        
        return generated
