import torch
import torch.nn as nn
from .geometry import LowRankChristoffel, SymplecticIntegrator, RK4Integrator, HeunIntegrator, LeapfrogIntegrator, DormandPrinceIntegrator, ReactiveChristoffel, HyperChristoffel, EuclideanChristoffel, HyperbolicChristoffel, SphericalChristoffel
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
        # Scalar curvature estimate "R"
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
             elif integrator_type == 'leapfrog':
                integ = LeapfrogIntegrator(self.christoffels[i], dt=0.1)
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
            
    def forward(self, x, v, force=None, context=None):
        """
        Args:
            x: Position [batch, dim]
            v: Velocity [batch, dim]
            force: External force [batch, dim]
            context: Context from previous layer [batch, context_dim]
        Returns:
            x_next, v_next, context_next, christoffel_outputs
        """
        # 1. Pre-LayerNorm (Residual approach)
        x_norm = self.norm_x(x)
        
        # Momentum-Preserving Stability:
        # LayerNorm destroys velocity magnitude. We use soft-clamping to prevent 
        # integration divergence while preserving relative momentum.
        v_mag = torch.norm(v, dim=-1, keepdim=True)
        v_norm = v * torch.min(torch.ones_like(v_mag), 5.0 / (v_mag + 1e-6))
        
        # Apply Recursive Context
        if self.use_recursive and context is not None:
             # Context (previous gates) acts as a "correction force"
             # "Turn here because the last layer struggled"
             correction = self.context_proj(context)
             if force is None:
                 force = correction
             else:
                 force = force + correction
        
        # 2. Split into heads
        # [batch, dim] -> list of [batch, head_dim]
        # We chunk the normalized x, but keep v untethered
        x_heads = x_norm.chunk(self.heads, dim=-1)
        v_heads = v_norm.chunk(self.heads, dim=-1)
        
        if force is not None:
            f_heads = force.chunk(self.heads, dim=-1)
        else:
            f_heads = [None] * self.heads
            
        # 3. Process each head independently
        x_outs = []
        v_outs = []
        gate_outputs = [] # Collect gates for next layer's context
        christoffel_outputs = [] # Collect Γ(v) for Noether/Hamiltonian loss

        
        for i in range(self.heads):
            # Dynamic time-step selection
            # Dynamic time-step selection
            if self.use_dynamic_time and self.time_heads is not None:
                # Auto-Wormholes: Predict optimal dt for this thought
                # scale is roughly [0.1, 5.0]
                dt_scale = self.time_heads[i](x_heads[i], v_heads[i], f_heads[i])
                
                # We still might use gating/softplus?
                # The head outputs the final scale directly.
                scale = dt_scale 
                gate_outputs.append(dt_scale) # Use dt prediction as context
            else:
                # Legacy Static Wormholes
                gate = self.gatings[i](x_heads[i])
                
                # Integrate with Learnable DT
                # scale = gate * softplus(dt_param) to ensure positive time
                dt_effective = nn.functional.softplus(self.dt_params[i]) * gate
                
                # Pass effective dt via dt_scale (assuming integrator uses dt * scale)
                # Since integrator has base dt=0.1, we scale relative to that.
                scale = dt_effective / 0.1
                gate_outputs.append(gate)
            
            # Collect geometry metadata for the loss function
            # We re-evaluate here to avoid modifying integrator return types
            # CRITICAL: no_grad() prevents metric networks from being trained by task loss
            # This avoids catastrophic interference between task learning and geometry
            with torch.no_grad():
                gamma = self.christoffels[i](v_heads[i], x_heads[i])
            christoffel_outputs.append(gamma)
            
            # Pure integration - integrator returns absolute next state
            x_h, v_h = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale)
            
            # Velocity normalization (preserves direction/memory, controls magnitude)
            # This prevents vanishing memory while maintaining stability
            v_h = v_h / (torch.norm(v_h, dim=-1, keepdim=True) + 1e-6)
            
            # Store absolute states (NOT deltas)
            x_outs.append(x_h)
            v_outs.append(v_h)
            
        # 4. Concatenate absolute states
        x_cat = torch.cat(x_outs, dim=-1)
        v_cat = torch.cat(v_outs, dim=-1)
        
        # 5. Output Projection (Mixing) - optional for multi-head
        if self.heads > 1:
            x_next = self.out_proj_x(x_cat)
            v_next = self.out_proj_v(v_cat)
        else:
            x_next, v_next = x_cat, v_cat
            
        # Prepare context for next layer
        if self.heads > 1:
             context_next = torch.cat(gate_outputs, dim=-1)
        else:
             context_next = gate_outputs[0]
             
        # Return absolute states (Pure Hamiltonian Evolution)
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
            
    def forward(self, x, v, force):
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
        
    def forward(self, x, v, force=None, context=None):
        # 1. Macro-evolution (Standard flow)
        x_m, v_m, ctx_m, christoffels = self.macro_manifold(x, v, force, context)
        
        if not self.physics_config.get('fractal', {}).get('enabled', False):
            return x_m, v_m, ctx_m, christoffels
            
        # 2. Estimate average Curvature R from Christoffel magnitudes
        # Gamma has shape [batch, head_dim]
        # We stack and take the norm to estimate local complexity
        stacked_gamma = torch.stack(christoffels, dim=1) # [batch, heads, head_dim]
        curvature_r = torch.norm(stacked_gamma, dim=-1).mean(dim=-1, keepdim=True) # [batch, 1]
        
        # 3. Tunneling condition (Smooth sigmoid gate)
        # alpha is 0 if curvature is low (flat), rises to 1 when r > threshold
        tunnel_gate = torch.sigmoid((curvature_r - self.threshold) * 5.0)
        
        # 4. Micro-evolution (Zooming in)
        # We use the macro-updated state as input to the sub-manifold
        # to refine the results in complex semantic regions.
        x_f, v_f, _, _ = self.micro_manifold(x_m, v_m, force, context)
        
        # 5. Recursive Blending
        # The micro-manifold provides a perturbative correction to the macro-flow
        x_final = x_m + tunnel_gate * (x_f - x_m) * self.alpha_scale
        v_final = v_m + tunnel_gate * (v_f - v_m) * self.alpha_scale
        
        return x_final, v_final, ctx_m, christoffels

