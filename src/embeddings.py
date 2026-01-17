"""
Implicit Neural Embeddings (INFs)
=================================

Replaces discrete lookup tables with continuous functions defined on a manifold.
Based on SIREN (Sinusoidal Representation Networks) for high-frequency detail.

Theory:
Instead of storing a vector E[i] for every token i, we store a low-rank coordinate c[i]
and learn a continuous function f(c) -> R^D.

    Embedding(i) = f( Coord(i) )

This allows:
1. Infinite Vocabulary (via hashing or continuous inputs)
2. Smooth Interpolation (Metric Topology between tokens)
3. Massive Parameter Reduction (O(1) vs O(V))
"""

import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    Linear Layer with Sinusoidal Activation (SIREN).
    High-frequency periodic activation allows fitting complex signals/embeddings.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer needs range to cover multiple periods [-1, 1] -> [-omega, omega]
                bound = 1 / self.linear.weight.size(1)
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Subsequent layers need specific initialization for gradient flow consistency
                bound = np.sqrt(6 / self.linear.weight.size(1)) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ImplicitEmbedding(nn.Module):
    """
    Implicit Neural Field Embedding.
    
    Maps Token IDs -> Learnable Coordinates -> Vector Space via SIREN.
    
    Args:
        vocab_size (int): Number of tokens (for coordinate table size).
        emb_dim (int): Output embedding dimension.
        coord_dim (int): Dimension of the underlying coordinate space (default: 16).
    """
    def __init__(self, vocab_size, emb_dim, coord_dim=16, hidden_dim=64, layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.coord_dim = coord_dim
        
        # 1. Coordinate Map (Low-Rank)
        # Much smaller than a full embedding table.
        # e.g. 10k tokens * 16 dim = 160k params (vs 10k * 256 = 2.5M params)
        self.coords = nn.Embedding(vocab_size, coord_dim)
        
        # Init coordinates uniformly to spread them out
        nn.init.uniform_(self.coords.weight, -1.0, 1.0)
        
        # 2. Continuous Function f(c) -> v
        # SIREN MLP
        net = []
        
        # Input Layer
        net.append(SineLayer(coord_dim, hidden_dim, is_first=True, omega_0=30.0))
        
        # Hidden Layers
        for _ in range(layers):
            net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=30.0))
            
        # Output Linear Projection (to match emb_dim magnitude correctly)
        # We don't use Sine on output to allow unbounded range if needed,
        # but typically embeddings are loose.
        self.net = nn.Sequential(*net)
        
        # Final projection to exact dimension
        self.out_proj = nn.Linear(hidden_dim, emb_dim)
        
        # Init final linear to be reasonable magnitude
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / 30.0
            self.out_proj.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.out_proj.bias)
            
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len] (Indices)
        Returns:
            embeddings: [batch, seq_len, emb_dim]
        """
        # 1. Lookup Coordinates
        # c: [batch, seq, coord_dim]
        c = self.coords(input_ids)
        
        # 2. Evaluate Field
        # x: [batch, seq, hidden]
        x = self.net(c)
        
        # 3. Project
        out = self.out_proj(x)
        
        return out

class FunctionalEmbedding(nn.Module):
    """
    Pure Functional Embedding (Zero-Lookup).
    Maps Index -> Sinusoidal Coordinate -> SIREN -> Vector.
    
    O(1) Memory: Parameters do NOT scale with Vocab Size.
    """
    def __init__(self, vocab_size, emb_dim, coord_dim=16, hidden_dim=64, layers=2):
        super().__init__()
        self.coord_dim = coord_dim
        # Ensure even coord_dim for sin/cos split
        if coord_dim % 2 != 0: self.coord_dim += 1
            
        # SIREN Network
        net = []
        net.append(SineLayer(self.coord_dim, hidden_dim, is_first=True, omega_0=30.0))
        for _ in range(layers):
            net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=30.0))
            
        self.net = nn.Sequential(*net)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)
        
        # Init projection
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / 30.0
            self.out_proj.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.out_proj.bias)
            
        # Register frequencies as buffer (fixed)
        # Log-space frequencies for multi-scale resolution of the ID
        freqs = torch.exp(torch.arange(0, self.coord_dim, 2).float() * -(np.log(10000.0) / self.coord_dim))
        self.register_buffer('freqs', freqs)
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len]
        """
        B, L = input_ids.shape
        
        # 1. Functional Coordinate Generation (Positional Encoding logic applied to ID)
        # inputs: [B, L, 1]
        x = input_ids.unsqueeze(-1).float()
        
        # project: [B, L, coord_dim//2]
        args = x * self.freqs
        
        # coords: [B, L, coord_dim] via sin/cos
        # cat last dim
        coords = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # 2. Evaluate Field
        x_out = self.net(coords)
        return self.out_proj(x_out)
