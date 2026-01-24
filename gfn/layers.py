import torch
import torch.nn as nn
from .geometry import LowRankChristoffel, ReactiveChristoffel, HyperChristoffel, EuclideanChristoffel, HyperbolicChristoffel, SphericalChristoffel
from .integrators import (
    SymplecticIntegrator, RK4Integrator, HeunIntegrator, LeapfrogIntegrator, 
    YoshidaIntegrator, DormandPrinceIntegrator, EulerIntegrator,
    ForestRuthIntegrator, OmelyanIntegrator, CouplingFlowIntegrator, NeuralIntegrator
)
from .scan import parallel_scan

class RiemannianGating(nn.Module):
    """
    Computes a scalar curvature-based gating mechanism.
    If curvature is high, dt should be small (complex region).
    If curvature is low (flat), dt can be large (skip connection behavior).
    """
    def __init__(self, dim):
        super().__init__()
        self.curvature_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid() # Range [0, 1]
        )
        
    def forward(self, x):
        """
        Returns a scaling factor for dt.
        """
        # Try CUDA path
        try:
            from .cuda.ops import dynamic_gating_fused, CUDA_AVAILABLE
            if CUDA_AVAILABLE and x.is_cuda:
                W1 = self.curvature_net[0].weight  # [dim/4, dim]
                b1 = self.curvature_net[0].bias    # [dim/4]
                W2 = self.curvature_net[2].weight  # [1, dim/4]
                b2 = self.curvature_net[2].bias    # [1]
                return dynamic_gating_fused(x, W1, b1, W2, b2)
        except Exception:
            pass
        
        # Fallback PyTorch
        return self.curvature_net(x)


class MLayer(nn.Module):
    """
    Manifold Layer (M-Layer):
    Takes current state (x, v) and input token force F.
    Evolves state via Geodesic Flow on multiple independent manifold subspaces.
    
    Architecture:
        1. Pre-LayerNorm (x, v)
        2. Split into K heads (Multi-Head Geodesic Flow)
        3. Parallel Geodesic Integration per head
        4. Concatenate & Mix
    
    Available integrators:
        - 'heun': Heun's method (RK2) - Fast & stable [DEFAULT]
        - 'rk4': Runge-Kutta 4 - High accuracy
        - 'rk45': Dormand-Prince (Golden Integration) - Adaptive
        - 'symplectic': Velocity Verlet - Energy preserving
        - 'leapfrog': Störmer-Verlet - Best symplectic
    """
    def __init__(self, dim, heads=4, rank=16, base_dt=0.1, integrator_type='heun', physics_config=None, layer_idx=0, total_depth=6):
        super().__init__()
        assert dim % heads == 0, f"Dim {dim} must be divisible by heads {heads}"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.physics_config = physics_config or {}
        self.base_dt = self.physics_config.get('stability', {}).get('base_dt', base_dt)
        
        # DeepNet-style depth scaling for gradient stability
        self.layer_idx = layer_idx
        self.total_depth = total_depth
        self.depth_scale = 1.0 / (total_depth ** 0.5)  # 1/√depth
        
        # 1. Pre-LayerNorm for stability (Standard in modern Transformers)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # 2. Independent or Symmetric Geodesic Dynamics per Head
        # Each head learns its own manifold geometry (Christoffel symbols)
        # Mixture of Manifolds (MoM) support
        mixture_cfg = self.physics_config.get('mixture', {})
        mixture_enabled = mixture_cfg.get('enabled', False)
        
        head_rank = max(4, rank // heads)
        sym_cfg = self.physics_config.get('symmetries', {})
        isomeric_groups = sym_cfg.get('isomeric_groups', None) # e.g. [[0, 1], [2, 3]]
        
        self.christoffels = nn.ModuleList()
        christoffel_map = {}
        
        if isomeric_groups:
             # Logic for symmetries override MoM individual allocation for grouped heads
             # We assume MoM is per-group if symmetries are on.
             pass

        # Manifold Factory
        def create_manifold(head_idx):
            if not mixture_enabled:
                 # Standard Behavior
                 hyper = self.physics_config.get('hyper_curvature', {}).get('enabled', False)
                 if hyper:
                     return HyperChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
                 else:
                     return ReactiveChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
            
            # Mixture allocation
            # components: {'euclidean': [0], 'hyperbolic': [1], 'spherical': [2], 'learnable': [3]}
            comps = mixture_cfg.get('components', {})
            
            # Check explicit assignment
            for type_name, indices in comps.items():
                if head_idx in indices:
                    if type_name == 'euclidean':
                        return EuclideanChristoffel(self.head_dim, physics_config=self.physics_config)
                    elif type_name == 'hyperbolic':
                        return HyperbolicChristoffel(self.head_dim, physics_config=self.physics_config)
                    elif type_name == 'spherical':
                        return SphericalChristoffel(self.head_dim, physics_config=self.physics_config)
                    elif type_name == 'learnable' or type_name == 'hyper':
                         return HyperChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
            
            # Default fallback for unassigned heads in MoM mode: Learnable (Hyper)
            return HyperChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)

        # Fill Map
        for i in range(heads):
             # Handle symmetries if present
             if isomeric_groups:
                 found_group = False
                 for group in isomeric_groups:
                     if i in group:
                         if group[0] in christoffel_map:
                             # Already created for leader
                             christoffel_map[i] = christoffel_map[group[0]]
                         else:
                             # Create for leader
                             instance = create_manifold(i)
                             christoffel_map[i] = instance
                             # Assign to others for consistency
                             for member in group:
                                 christoffel_map[member] = instance
                         found_group = True
                         break
                 if found_group: continue
             
             # Independent
             christoffel_map[i] = create_manifold(i)
        
        # Add to ModuleList in order
        for i in range(heads):
            self.christoffels.append(christoffel_map[i])
            
        # 🚀 PERFORMANCE: Stack Christoffel Parameters for Vectorized Head Processing
        # This allows us to call a single bmm instead of looping over heads.
        self.register_buffer('headless_mode', torch.tensor(False)) 
        
        # We REMOVED self.U_stack / self.W_stack parameters as they were redundant copies.
        # model.py generates U_stack dynamically from self.christoffels for the fused kernel.

        
        # Integrators per head and Time Scaling
        # Check if "Autonomous Geometric Attention" (Dynamic Time) is enabled
        self.use_dynamic_time = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False)
        
        if self.use_dynamic_time:
            # Auto-Wormholes: Model predicts dt per head/step
            # Disabled for now as TimeDilationHead was removed
            # range_min, range_max = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('range', [0.1, 5.0])
            self.time_heads = None 
            self.gatings = None
            print("Warning: dynamic_time enabled in config but code support was removed. Using scalar dt.")
        else:
            # Gating per head (Legacy Static Wormholes)
            self.gatings = nn.ModuleList([
                RiemannianGating(self.head_dim) for _ in range(heads)
            ])
            
            # Static Wormholes (Multi-Scale Initialization)
            scale_vals = []
            for i in range(heads):
                 # Head 0: dt scale = 1.0 (Fast)
                # Head k: dt scale = 1.5^k (Slow)
                scale_init = 1.5 ** i
                val = torch.tensor(scale_init).log() # Initial bias
                scale_vals.append(val)
                
            self.dt_params = nn.Parameter(torch.tensor(scale_vals))
            self.time_heads = None

        # Friction coefficient for Conformal Symplectic System
        # Integrated formally into the integrator dynamics
        self.friction = self.physics_config.get('stability', {}).get('friction', 0.05)


        
        self.integrators = nn.ModuleList()
        for i in range(heads):
            # Integrator setup
             if integrator_type == 'rk4':
                integ = RK4Integrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'rk45':
                # Golden Integration
                integ = DormandPrinceIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'heun':
                integ = HeunIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'euler':
                 integ = EulerIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'leapfrog':
                integ = LeapfrogIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'yoshida':
                 integ = YoshidaIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'forest_ruth':
                 integ = ForestRuthIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'omelyan':
                 integ = OmelyanIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'coupling':
                 integ = CouplingFlowIntegrator(self.christoffels[i], dt=0.1)
             elif integrator_type == 'neural':
                 # Neural integrator needs dim to build its controller
                 integ = NeuralIntegrator(self.christoffels[i], dt=0.1, dim=self.head_dim)
             else:
                integ = SymplecticIntegrator(self.christoffels[i], dt=0.1)
             self.integrators.append(integ)
            
        # Output projection for mixing heads
        if heads > 1:
            self.out_proj_x = nn.Linear(dim, dim)
            self.out_proj_v = nn.Linear(dim, dim)
            
            # Init as almost identity to start with stable independent dynamics?
            # Or standard init?
            # Let's use standard init but small to preserve flow structure
            nn.init.xavier_uniform_(self.out_proj_x.weight)
            # Full scale for faster learning on discrete logic
            nn.init.zeros_(self.out_proj_x.bias)
            
            nn.init.xavier_uniform_(self.out_proj_v.weight)
            nn.init.zeros_(self.out_proj_v.bias)
            
        # Recursive Geodesics: "Copilot" Mixer
        # Projects previous layer's context (e.g. curvature/gate) into this layer's force
        self.use_recursive = self.physics_config.get('active_inference', {}).get('recursive_geodesics', {}).get('enabled', False)
        if self.use_recursive:
            self.context_proj = nn.Linear(heads, dim) # context is [batch, heads] (gates)
            nn.init.zeros_(self.context_proj.weight) # Start with no influence
            
    def forward(self, x, v, force=None, context=None, collect_christ=False):
        """
        Vectorized Forward: Processes ALL heads in parallel via Tensor Batching.
        Professional Speed Path.
        """
        batch = x.shape[0]
        
        # We bypass LayerNorm on Hamiltonian states x and v to preserve history
        x_norm = self.norm_x(x)
        v_norm = v # Unit-norm is handled per head
        
        # 2. Reshape into Head Batches [Heads, Batch, HeadDim]
        # This is the "Professional" way to avoid Python loops.
        x_heads = x_norm.view(batch, self.heads, self.head_dim).transpose(0, 1)
        v_heads = v_norm.view(batch, self.heads, self.head_dim).transpose(0, 1)
        
        if force is not None:
             # Apply recursive context before reshaping
             if self.use_recursive and context is not None:
                 force = force + self.context_proj(context)
             f_heads = force.view(batch, self.heads, self.head_dim).transpose(0, 1)
        else:
             f_heads = torch.zeros_like(x_heads)
             
        # 3. Vectorized Gating [Heads, Batch, 1]
        # Currently gatings are separate modules, we can vectorize them by 
        # combining their weights if they are all Linear.
        # For v2.6.4, we use a slightly faster list comprehension but target vectorized linear soon.
        gates = torch.stack([self.gatings[i](x_heads[i]) for i in range(self.heads)], dim=0)
        
        dt_base = torch.nn.functional.softplus(self.dt_params).view(self.heads, 1, 1)
        scale = (dt_base * gates) / self.base_dt # [Heads, Batch, 1]
        
        # 4. Batched Geodesic Step
        # We call the integrator with the FULL stack of heads
        # Each integrator must support [H, B, d] inputs.
        
        # Update U_stack and W_stack if they were modified (parameter tie)
        # In professional mode, we use the stacked versions directly.
        
        # Calculate Γ(v, v) for ALL heads in one go
        # Since integrators in ModuleList are individual, we still loop them, 
        # BUT the heavy math (Christoffel) is now vectorized inside them.
        
        x_outs = []
        v_outs = []
        christoffel_outputs = []
        
        for i in range(self.heads):
            # LEVEL 8 PHYSICAL ALIGNMENT: Strict Unit-Norm Velocity
            # We must force |v|=1.0 per head to match the Fused CUDA kernel
            vh_in = v_heads[i]
            v_norm_val = torch.norm(vh_in, dim=-1, keepdim=True) + 1e-6
            vh_unit = vh_in / v_norm_val
            
            # Step for head i
            xh, vh = self.integrators[i](x_heads[i], vh_unit, force=f_heads[i], dt_scale=scale[i])
            
            # Re-normalize after step for absolute stability (Matches CUDA s_v_h /= v_norm)
            vh_norm_out = torch.norm(vh, dim=-1, keepdim=True) + 1e-6
            vh = vh / vh_norm_out
            
            x_outs.append(xh)
            v_outs.append(vh)
            
            if collect_christ or self.training:
                 with torch.no_grad():
                     christoffel_outputs.append(self.christoffels[i](v_heads[i], x_heads[i]))

        # 5. Concatenate and Mix
        # Try CUDA path first if available and not collecting christoffel
        if self.heads > 1 and not collect_christ:
            try:
                from .cuda.ops import head_mixing_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE and x.is_cuda:

                    # Stack outputs: [H, B, D/H]
                    x_stacked = torch.stack(x_outs, dim=0)
                    v_stacked = torch.stack(v_outs, dim=0)
                    
                    # Dispatch to CUDA
                    x_next, v_next = head_mixing_fused(
                        x_stacked, v_stacked,
                        self.out_proj_x.weight, self.out_proj_v.weight
                    )
                    context_next = gates.squeeze(-1).transpose(0, 1)
                    return x_next, v_next, context_next, christoffel_outputs
            except Exception:
                # Fallback to PyTorch if CUDA fails
                pass
        
        # Fallback: PyTorch mixing
        x_cat = torch.stack(x_outs, dim=1).view(batch, -1)
        v_cat = torch.stack(v_outs, dim=1).view(batch, -1)
        
        if self.heads > 1:
            x_next = self.out_proj_x(x_cat)
            v_next = self.out_proj_v(v_cat)
            context_next = gates.squeeze(-1).transpose(0, 1) # [Batch, Heads]
        else:
            x_next, v_next = x_cat, v_cat
            context_next = gates.squeeze(-1).transpose(0, 1)
            
        return x_next, v_next, context_next, christoffel_outputs



class ParallelMLayer(nn.Module):
    """
    Parallel Manifold Layer (M-Layer) using Associative Scan.
    
    linearizes the Geodesic Flow to enable O(log N) parallel training.
    
        dv/dt = F - \\Gamma(v, v)   [Non-linear]
        
        Is approximated as a Linear Time-Varying (LTV) system during scan:
        dv/dt = F - D(F) * v       [Linearized]
        
        Where D(F) is a predicted damping/rotation factor based on input force.
        
    Dynamics:
        v_t = A_t * v_{t-1} + B_t
        x_t = x_{t-1} + v_t * dt
        
    Args:
        dim: Hidden dimension
        heads: Number of heads
    """
    def __init__(self, dim, heads=4, physics_config=None, **kwargs):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.norm_x = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # Parallel Geometry Predictors
        # Instead of implicit Christoffels, we predict linearization params A, B directly
        
        # Predict A_t (Decay/Rotation) from input Force
        # A_t = 1 - dt * D, where D > 0
        self.to_A = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid() # Output range [0, 1] acts as "retain gate" (A) directly
        )
        
        # Predict B_t (Input modulation) from input Force
        self.to_B = nn.Linear(dim, dim)
        
        self.to_dt = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softplus()
        )
        
        # Parallel Multi-Scale Initialization
        # We want different channels to have different base time-scales 
        # to effectively create "Wormholes" in the parallel scan.
        # Channels 0..HeadDim: Fast
        # Channels ...: Slow
        scale_vec = []
        for i in range(heads):
            # Scale for this head
            s = 1.5 ** i
            # Append s repeated head_dim times
            scale_vec.extend([s] * (dim // heads))
        
        # Register as buffer (fixed base scales, learnable modulation via to_dt)
        self.register_buffer('base_dt_scales', torch.tensor(scale_vec, dtype=torch.float32))
        
        self.base_dt = 0.1
        
        # Output projection
        if heads > 1:
            self.out_proj = nn.Linear(dim * 2, dim * 2)
            
    def forward(self, x, v, force, collect_christ=False):
        """
        Args:
            x: [Batch, Seq, Dim]
            v: [Batch, Seq, Dim]
            force: [Batch, Seq, Dim] (All timesteps at once!)
            
        Returns:
            x_seq, v_seq: [Batch, Seq, Dim]
        """
        B, L, D = force.shape
        
        # 1. Pre-norm
        if x is not None:
             x_norm = self.norm_x(x)
        else:
             force = self.norm_x(force)
        
        # Parallel Scan Logic:
        # We model the dynamics as a Linear Time-Varying (LTV) system for O(log N) parallelization.
        # v_t = A_t * v_{t-1} + B_t
        
        # Compute linearization parameters for ALL timesteps in parallel
        # Force acts as the input signal "u_t"
        
        # A_t [B, L, D] = Decay factor (0 = forget/stop, 1 = persist/fly)
        A = self.to_A(force) 
        
        # dt [B, L, D]
        # Modulate learned dt by the multi-scale base factors
        dt = self.to_dt(force) * self.base_dt * self.base_dt_scales.view(1, 1, -1)
        
        # B_t [B, L, D] = Effective input
        B_val = self.to_B(force) * dt
        
        # 2. Run Parallel Scan for Velocity
        # v_t = A_t * v_{t-1} + B_t
        v_seq = parallel_scan(A, B_val)
        
        # 3. Integrate Position
        # x_t = x_{t-1} + v_t * dt
        # This is another scan! 
        # x_t = 1 * x_{t-1} + (v_t * dt)
        x_update = v_seq * dt
        # Position scan: x_t = x_{t-1} + v_t * dt
        A_pos = torch.ones_like(v_seq)  # Identity for position accumulation
        x_seq = parallel_scan(A_pos, x_update)
        
        # In Parallel mode, we don't return individual head curvatures currently 
        # (needs complex extraction from the scan parameters)
        return x_seq, v_seq, None, []


class FractalMLayer(nn.Module):
    """
    Fractal Manifold Layer: Implements multiscale "Recursive Tunneling".
    
    If local curvature R is high, the particle "tunnels" into a 
    high-resolution sub-manifold to resolve semantic complexity.
    """
    def __init__(self, dim, heads=8, rank=16, base_dt=0.1, integrator_type='symplectic', physics_config=None, layer_idx=0, total_depth=6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.rank = rank
        self.physics_config = physics_config or {}
        
        # DeepNet-style depth scaling for gradient stability
        self.layer_idx = layer_idx
        self.total_depth = total_depth
        self.depth_scale = 1.0 / (total_depth ** 0.5)  # 1/√depth
        
        # Macro-manifold: Standard MLayer evolution
        self.macro_manifold = MLayer(
            dim, heads=heads, rank=rank, 
            base_dt=base_dt, integrator_type=integrator_type, 
            physics_config=self.physics_config
        )
        
        # Sub-manifold: Dedicated to resolving high-curvature details
        # Smaller rank but higher resolution (smaller dt)
        micro_cfg = self.physics_config.copy()
        # Disable fractal recursion in the sub-manifold to avoid infinite loops
        if 'fractal' not in micro_cfg: micro_cfg['fractal'] = {}
        micro_cfg['fractal']['enabled'] = False 
        
        self.micro_manifold = MLayer(
            dim, heads=heads, rank=max(8, rank//2), 
            base_dt=base_dt * 0.5, integrator_type=integrator_type, 
            physics_config=micro_cfg
        )
        
        fract_cfg = self.physics_config.get('fractal', {})
        self.threshold = fract_cfg.get('threshold', 0.5)
        self.alpha_scale = fract_cfg.get('alpha', 0.2)
        
    def forward(self, x, v, force=None, context=None, collect_christ=False):
        # 1. Macro-evolution (Standard flow)
        x_m, v_m, ctx_m, christoffels = self.macro_manifold(x, v, force, context, collect_christ=collect_christ)
        
        if not self.physics_config.get('fractal', {}).get('enabled', False):
            return x_m, v_m, ctx_m, christoffels
            
        # 2. Estimate average Curvature R from Christoffel magnitudes
        # Gamma has shape [batch, head_dim]
        # We stack and take the norm to estimate local complexity
        stacked_gamma = torch.stack(christoffels, dim=1) # [batch, heads, head_dim]
        curvature_r = torch.norm(stacked_gamma, dim=-1).mean(dim=-1, keepdim=True) # [batch, 1]
        
        # 3. Tunneling condition (Smooth sigmoid gate)
        # alpha is 0 if curvature is low (flat), rises to 1 when r > threshold
        # GRADIENT FIX: Increase baseline to 0.3 for stronger micro_manifold signal
        tunnel_gate = 0.3 + 0.7 * torch.sigmoid((curvature_r - self.threshold) * 5.0)
        
        # 4. Micro-evolution (Zooming in)
        # We use the macro-updated state as input to the sub-manifold
        # to refine the results in complex semantic regions.
        x_f, v_f, _, _ = self.micro_manifold(x_m, v_m, force, context, collect_christ=collect_christ)
        
        # 5. Recursive Blending
        # The micro-manifold provides a perturbative correction to the macro-flow
        x_final = x_m + tunnel_gate * (x_f - x_m) * self.alpha_scale
        v_final = v_m + tunnel_gate * (v_f - v_m) * self.alpha_scale
        
        return x_final, v_final, ctx_m, christoffels

