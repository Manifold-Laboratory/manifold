import torch
import torch.nn as nn

class LowRankChristoffel(nn.Module):
    r"""
    Computes the Christoffel symbols \Gamma^k_{ij} using a low-rank decomposition.
    To ensure symmetry in lower indices (torsion-free), we use a symmetric decomposition:
    \Gamma^k_{ij} = \sum_{r=1}^R \lambda_{kr} * (U_{ir} * U_{jr})
    
    Args:
        dim (int): Dimension of the manifold (hidden size).
        rank (int): Rank of the decomposition.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.config = physics_config or {}
        self.clamp_val = self.config.get('stability', {}).get('curvature_clamp', 5.0)
        
        # Factors to reconstruct Gamma
        # U: [dim, rank] - represents the "basis" for the input indices i, j
        # W: [dim, rank] - represents the "basis" for the output index k (or weighting)
        # Init very small to start with FLAT manifold (Euclidean geometry)
        # This helps in preserving long-term dependencies (linear dynamics)
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.001)
        self.W = nn.Parameter(torch.randn(dim, rank) * 0.001)
        
        # Friction coefficient for Conformal Symplectic System
        self.friction = self.config.get('stability', {}).get('friction', 0.05)
        
        # Position Gate V: dim -> 1 (Scalar gravity well strength)
        # We start with near-zero weights so initially there are no gravity wells.
        # Position Gate V: dim -> 1 (Scalar gravity well strength)
        # We start with near-zero weights so initially there are no gravity wells.
        self.V = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.V.weight)
        
        # Adaptive Curvature Gate (The Valve): dim -> dim
        # Learns when to apply curvature vs coasting
        # Init to open (0 bias = 0.5 sigmoid) or slightly closed?
        # Let's start neutral.
        self.gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0) # Start OPEN (sigmoid(2) ~ 0.88)
        
        # === Dynamic Friction (The Forget Gate) ===
        # Replaces static friction.
        # Logic: If context switch needed -> High Friction -> Dissipate Energy -> Forget.
        # If long-term dependency -> Low Friction -> Conserve Energy -> Remember.
        # F_damp = - sigma(Gate(x)) * v
        self.forget_gate = nn.Linear(dim, dim)
        # Init to low friction (preserve memory by default = 0.05 equivalent)
        nn.init.normal_(self.forget_gate.weight, std=0.01)
        nn.init.constant_(self.forget_gate.bias, -3.0) # sigmoid(-3) approx 0.047 (close to static 0.05)
        
    def forward(self, v, x=None):
        """
        Compute Generalized Force: Γ(v, v) + Friction(x)*v
        
        Output represents the effective "Resistance" to motion.
        Acc = F_ext - Output
        """
        # Try Fused CUDA Kernel (Now supports Training via Autograd!)
        if x is None and v.is_cuda:
            try:
                from gfn.cuda.ops import christoffel_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    return christoffel_fused(v, self.U, self.W)
            except Exception:
                pass
        
        # PyTorch Implementation
        # v: [batch, dim]
        proj = torch.matmul(v, self.U) # [batch, rank]
        
        # Norm-based saturation (geometrically consistent)
        norm = torch.norm(proj, dim=-1, keepdim=True)
        scale = 1.0 / (1.0 + norm)  # Soft saturation based on total magnitude
        sq = (proj * proj) * scale
        
        gamma = torch.matmul(sq, self.W.t()) # [batch, dim]
        
        # Dynamic Curvature Modulation (Gravity Wells)
        if x is not None:
            modulation = torch.sigmoid(self.V(x)) # Range (0, 1)
            gamma = gamma * (1.0 + modulation)
            
        # Stability: Tight clamp
        gamma = torch.clamp(gamma, -self.clamp_val, self.clamp_val)
        
        # Adaptive Gating for Curvature
        if x is not None:
            gate = torch.sigmoid(self.gate_proj(x)) # [batch, dim]
            gamma = gamma * gate
            
        # === Apply Dynamic Friction (Forget Gate) ===
        # This adds the linear damping term: + mu(x) * v
        # Since Acc = -Output, this becomes Acc = -Gamma - mu*v (Correct Damped Harmonic Oscillator)
        if x is not None:
            # Friction coefficient per dimension: [0, 1]
            friction = torch.sigmoid(self.forget_gate(x))
            damping_force = friction * v
            
            # Combine geometric curvature with thermodynamic friction
            # Total Resistance = Gamma(v^2) + Damping(v)
            return gamma + damping_force
            
        return gamma



class HyperChristoffel(LowRankChristoffel):
    """
    Hyper-Christoffel: Context-Dependent Geometry.
    
    Architecture:
    Gamma(v, v | x) = W(x) * (U(x)^T v)^2
    
    Efficient Implementation (Gated Modulation):
    U(x) = U_static * diag(Gate_u(x))
    W(x) = W_static * diag(Gate_w(x))
    
    Where Gate(x) outputs a [rank] vector in [0, 2], scaling the importance 
    of each geometric basis vector dynamically.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__(dim, rank, physics_config)
        
        # HyperNetworks: State x -> Modulation Gates [rank]
        # Light-weight: just a linear projection + activation
        self.gate_u = nn.Linear(dim, rank)
        self.gate_w = nn.Linear(dim, rank)
        
        # Initialize gates to be near identity (output ~1.0)
        # Sigmoid(0) = 0.5 -> * 2 = 1.0
        nn.init.zeros_(self.gate_u.weight)
        nn.init.zeros_(self.gate_u.bias)
        nn.init.zeros_(self.gate_w.weight)
        nn.init.zeros_(self.gate_w.bias)
        
    def forward(self, v, x=None):
        if x is None:
            # Fallback to static if no context provided (e.g. init or blind mode)
            return super().forward(v, None)
            
        # 1. Compute Context Gates
        # Range: [0, 2] - allowing to silence (0) or amplify (2) specific basis vectors
        g_u = torch.sigmoid(self.gate_u(x)) * 2.0 # [batch, rank]
        g_w = torch.sigmoid(self.gate_w(x)) * 2.0 # [batch, rank]
        
        # 2. Modulate Static Basis
        # U: [dim, rank]
        # g_u: [batch, rank]
        # Effective U: U * g_u (broadcast) -> effectively specific U for each batch item!
        # U_dynamic = U (1, dim, rank) * g_u (batch, 1, rank)
        
        # PyTorch optimization: Don't materialize full U_dynamic [batch, dim, rank] (too big)
        # Instead, modulate projection:
        # proj = v @ U -> [batch, rank]
        # proj_dynamic = proj * g_u
        
        # Weights U, W are [dim, rank]
        # v: [batch, dim]
        
        # a) Project momentum onto static basis
        proj_static = torch.matmul(v, self.U) # [batch, rank]
        
        # b) Modulate projection by Context (Hyper-U)
        proj_dynamic = proj_static * g_u # [batch, rank]
        
        # c) Soft-Saturation (to prevent energy explosion)
        # Instead of pure quadratic sq_dynamic = proj_dynamic * proj_dynamic
        sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic))
        
        # d) Modulate Reconstruction by Context (Hyper-W)
        sq_modulated = sq_dynamic * g_w # [batch, rank]
        
        # e) Reconstruct force
        # out = sq_modulated @ W.T
        out = torch.matmul(sq_modulated, self.W.t()) # [batch, dim]
        
        # 3. Apply inherited Active Inference (Plasticity/Singularities)
        # Note: HyperChristoffel currently inherits from LowRankChristoffel directly.
        # Active inference features from ReactiveChristoffel are not automatically included unless explicitly mixed in.
        
        return torch.clamp(out, -self.clamp_val, self.clamp_val)

class ReactiveChristoffel(LowRankChristoffel):
    """
    Active Inference: Geometry that reacts to the agent's state.
    
    Features:
    1. Reactive Curvature (Plasticity): Metric deforms based on kinetic energy.
       High energy (confusion/exploration) -> Higher curvature (more braking).
       
    2. Logical Singularities: If 'V(x)' (potential) exceeds a threshold, 
       we trigger a 'Black Hole' (infinite curvature) to trap the thought 
       in a semantic certainty.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__(dim, rank, physics_config=physics_config)
        self.config = physics_config or {}
        self.active_cfg = self.config.get('active_inference', {})
        
        self.plasticity = self.active_cfg.get('reactive_curvature', {}).get('plasticity', 0.0)
        self.singularity_threshold = self.active_cfg.get('singularities', {}).get('threshold', 0.8)
        self.black_hole_strength = self.active_cfg.get('singularities', {}).get('strength', 10.0)

    def forward(self, v, x=None):
        # Try Fused CUDA Kernel (Supports Training via Autograd!)
        if v.is_cuda:
            try:
                from gfn.cuda.ops import christoffel_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    # Pass Active Parameters
                    # x and V.weight are needed for Singularities
                    V_w = self.V.weight if (x is not None and self.active_cfg.get('singularities', {}).get('enabled', False)) else None
                    pos_x = x if (V_w is not None) else None
                    
                    return christoffel_fused(
                        v, self.U, self.W, 
                        x=pos_x, V_w=V_w, 
                        plasticity=self.plasticity if self.active_cfg.get('reactive_curvature', {}).get('enabled', False) else 0.0,
                        sing_thresh=self.singularity_threshold,
                        sing_strength=self.black_hole_strength
                    )
            except Exception:
                pass

        # Base curvature (static memory or PyTorch fallback)
        gamma = super().forward(v, x)
        
        if not self.active_cfg.get('enabled', False):
            return gamma
            
        # 1. Reactive Curvature (Plasticity)
        if self.active_cfg.get('reactive_curvature', {}).get('enabled', False):
            # Energy = Kinetic Energy of thoughts (~ v^2)
            # Use tanh to bound the reaction
            energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
            # If energy is high, increase curvature (slow down/turn harder)
            # Gamma_new = Gamma * (1 + alpha * energy)
            gamma = gamma * (1.0 + self.plasticity * energy)
            
        # 2. Logical Singularities (Black Holes)
        if self.active_cfg.get('singularities', {}).get('enabled', False) and x is not None:
            # Check Semantic Potential V(x)
            # We use the existing self.V gate from LowRankChristoffel
            potential = torch.sigmoid(self.V(x)) # [batch, 1]
            
            # If we are very sure (High Potential), trigger Singularity
            # This creates a stiff attractor
            is_singularity = (potential > self.singularity_threshold).float()
            
            # Apply Black Hole Gravity: Gamma * Strength
            # But only where potential is high
            singularity_mult = 1.0 + is_singularity * (self.black_hole_strength - 1.0)
            gamma = gamma * singularity_mult
            
        return gamma



# --- Analytic Manifolds (MoM Components) ---

class EuclideanChristoffel(nn.Module):
    """
    Flat Geometry. Gamma = 0.
    Standard Deep Learning / ResNet behavior.
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        
    def forward(self, v, x=None):
        return torch.zeros_like(v)

class HyperbolicChristoffel(nn.Module):
    """
    Hyperbolic Geometry (Poincaré Ball Model).
    Constant Negative Curvature.
    
    Structure:
    Tree-like embeddings, ideal for Hierarchies and Syntax.
    
    Geodesic Accel: a = -Gamma(v,v)
    Approximation near origin or exact formula?
    Uses Conformal Factor lambda = 2 / (1 - |x|^2)
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        self.curvature = -1.0
        
    def forward(self, v, x):
        if x is None: return torch.zeros_like(v)
        
        # Conformal factor lambda(x) approx
        # For numeric stability with unconstrained x, we treat x as being in tangent space 
        # mapped to manifold, or we assume x is typically small.
        # Strict Poincaré requires |x| < 1.
        # We implementation a Soft-Poincaré:
        # Scale curvature effect by distance from origin.
        
        # Formula: a = 2 (<x,v>v - |v|^2 x) / (1 - |x|^2)  (roughly)
        # We simplify to: Gamma ~ - ( <x,v>v - |v|^2 x )
        # Negative curvature pushes paths APART (diverge).
        
        x_sq = torch.sum(x*x, dim=-1, keepdim=True)
        v_sq = torch.sum(v*v, dim=-1, keepdim=True)
        xv = torch.sum(x*v, dim=-1, keepdim=True)
        
        # Divergent force:
        gamma = 2 * xv * v - v_sq * x
        
        # Scale by 1/(1-x^2)? No, dangerous if x not bounded.
        # Let's just use the directionality for now as a "Hyperbolic Bias".
        return gamma * 0.1 # Small scale factor for stability

class SphericalChristoffel(nn.Module):
    """
    Spherical Geometry (Stereographic Projection).
    Constant Positive Curvature.
    
    Structure:
    Cyclic embeddings, valid for Rotations and Patterns.
    
    Positive curvature pulls paths TOGETHER (converge).
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        self.curvature = 1.0
        
    def forward(self, v, x):
        if x is None: return torch.zeros_like(v)
        
        x_sq = torch.sum(x*x, dim=-1, keepdim=True)
        v_sq = torch.sum(v*v, dim=-1, keepdim=True)
        xv = torch.sum(x*v, dim=-1, keepdim=True)
        
        # Convergent force (Sign flip vs Hyperbolic):
        # Gamma ~ ( <x,v>v - |v|^2 x )
        gamma = -(2 * xv * v - v_sq * x)
        
        return gamma * 0.1
