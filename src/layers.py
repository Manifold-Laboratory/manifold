import torch
import torch.nn as nn
from .geometry import LowRankChristoffel, SymplecticIntegrator, RK4Integrator, HeunIntegrator, LeapfrogIntegrator, DormandPrinceIntegrator, ReactiveChristoffel, TimeDilationHead
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
    def __init__(self, dim, heads=4, rank=16, base_dt=0.1, integrator_type='heun', physics_config=None):
        super().__init__()
        assert dim % heads == 0, f"Dim {dim} must be divisible by heads {heads}"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.base_dt = base_dt
        self.physics_config = physics_config or {}
        
        # 1. Pre-LayerNorm for stability (Standard in modern Transformers)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # 2. Independent Geodesic Dynamics per Head
        # Each head learns its own manifold geometry (Christoffel symbols)
        # We share the rank budget across heads to keep params similar
        head_rank = max(4, rank // heads)
        
        # Use ReactiveChristoffel if Active Inference is enabled
        # It defaults to static behavior if disabled, so safe to use always?
        # Yes, we implemented it to check its own config.
        self.christoffels = nn.ModuleList([
            ReactiveChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
            for _ in range(heads)
        ])
        
        # Gating per head
        self.gatings = nn.ModuleList([
            RiemannianGating(self.head_dim) for _ in range(heads)
        ])
        
        # Integrators per head and Time Scaling
        # Check if "Autonomous Geometric Attention" (Dynamic Time) is enabled
        self.use_dynamic_time = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False)
        
        if self.use_dynamic_time:
            # Auto-Wormholes: Model predicts dt per head/step
            range_min, range_max = self.physics_config['active_inference']['dynamic_time'].get('range', [0.1, 5.0])
            self.time_heads = nn.ModuleList([
                TimeDilationHead(self.head_dim, range_min, range_max)
                for _ in range(heads)
            ])
            # We don't use dt_params in this mode
        else:
            # Static Wormholes (Multi-Scale Initialization)
            scale_vals = []
            for i in range(heads):
                 # Head 0: dt scale = 1.0 (Fast)
                # Head k: dt scale = 1.5^k (Slow)
                scale_init = 1.5 ** i
                val = torch.tensor(scale_init).log() # Initial bias
                scale_vals.append(val)
                
            self.dt_params = nn.Parameter(torch.tensor(scale_vals))
        
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
            nn.init.eye_(self.out_proj_x.weight)
            nn.init.zeros_(self.out_proj_x.bias)
            nn.init.eye_(self.out_proj_v.weight)
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
            x_next, v_next, context_next
        """
        # 1. Pre-LayerNorm
        x_norm = self.norm_x(x)
        v_norm = self.norm_v(v)
        
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
        
        for i in range(self.heads):
            # Dynamic time-step selection
            if self.use_dynamic_time:
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
            
            x_h, v_h = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale)
            
            x_outs.append(x_h)
            v_outs.append(v_h)
            
        # 4. Concatenate
        x_cat = torch.cat(x_outs, dim=-1)
        v_cat = torch.cat(v_outs, dim=-1)
        
        # 5. Output Projection (Mixing)
        if self.heads > 1:
            x_geo = self.out_proj_x(x_cat)
            v_geo = self.out_proj_v(v_cat)
        else:
            x_geo, v_geo = x_cat, v_cat
            
        # Prepare context for next layer
        # Concatenate gates [batch, heads]
        if self.heads > 1:
             context_next = torch.cat(gate_outputs, dim=-1)
        else:
             context_next = gate_outputs[0]
             
        return x_geo, v_geo, context_next


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
        
        # 1. Pre-norm (Adapting for Stacked SSM behavior)
        if x is not None:
             x_norm = self.norm_x(x)
             # If x is provided, we might want to use it, but for Parallel Scan 
             # the 'force' argument carries the sequence input.
        else:
             # In stacked mode, 'force' is the input from the previous layer
             force = self.norm_x(force)
        # Wait, in parallel scan training, we compute the whole sequence of states from the sequence of forces.
        # We don't take x_t as input for the layer, we take the *previous layer's output sequence*.
        # But for the FIRST layer, x is fixed (embedded).
        # Actually, standard RNN/Transformer layer takes "hidden states" sequence.
        # Here we take the sequence of "Force" (inputs) and evolve internal state.
        
        # For M-Layer:
        # Input: "Force" sequence (function of PREVIOUS layer outputs or Embeddings)
        # Internal State: (x, v)
        # Output: Updated (x, v) sequence
        
        # Compute linearization parameters for ALL timesteps in parallel
        # Force acts as the input signal "u_t"
        
        # A_t [B, L, D] = Decay factor (0 = forget/stop, 1 = persist/fly)
        A = self.to_A(force) 
        
        # dt [B, L, D]
        # Modulate learned dt by the multi-scale base factors (Wormholes)
        dt = self.to_dt(force) * self.base_dt * self.base_dt_scales.view(1, 1, -1)
        
        # Apply dt to A? 
        # In discrete form v = (1 - D*dt)v + F*dt
        # Let's say our network predicts the 'effective' A directly for stability.
        
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
        # Parallel cumsum is a special case of scan where A=1
        x_seq = torch.cumsum(x_update, dim=1) 
        
        # Add initial conditions if provided (usually 0 or learned)
        # Assuming starting from 0 for the sequence relative block
        
        return x_seq, v_seq
