import torch
import torch.nn as nn
from ..geometry import (
    LowRankChristoffel, ReactiveChristoffel, HyperChristoffel, 
    EuclideanChristoffel, HyperbolicChristoffel, SphericalChristoffel,
    ToroidalChristoffel
)
from ..integrators import (
    SymplecticIntegrator, RK4Integrator, HeunIntegrator, LeapfrogIntegrator, 
    YoshidaIntegrator, DormandPrinceIntegrator, EulerIntegrator,
    ForestRuthIntegrator, OmelyanIntegrator, CouplingFlowIntegrator, NeuralIntegrator
)
from .gating import RiemannianGating

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
            topo_type = self.physics_config.get('topology', {}).get('type', 'euclidean').lower()
            is_torus = (topo_type == 'torus')

            if not mixture_enabled:
                 # Standard Behavior
                 hyper = self.physics_config.get('hyper_curvature', {}).get('enabled', False)
                 if is_torus:
                      return ToroidalChristoffel(self.head_dim, physics_config=self.physics_config)
                 elif hyper:
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
                    elif type_name == 'toroidal' or type_name == 'torus':
                         return ToroidalChristoffel(self.head_dim, physics_config=self.physics_config)
            
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
            
        # Performance Optimization: Stack Christoffel Parameters for Vectorized Head Processing
        # This allows us to call a single bmm instead of looping over heads.
        self.register_buffer('headless_mode', torch.tensor(False)) 
        
        # We REMOVED self.U_stack / self.W_stack parameters as they were redundant copies.
        # model.py generates U_stack dynamically from self.christoffels for the fused kernel.

        
        # Integrators per head and Time Scaling
        # Check if "Autonomous Geometric Attention" (Dynamic Time) is enabled
        self.use_dynamic_time = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False)
        
        # LEVEL 13: TOPOLOGY PARSING
        topo_type = self.physics_config.get('topology', {}).get('type', 'euclidean').lower()
        self.topology_id = 1 if topo_type == 'torus' else 0

        if self.use_dynamic_time:
            # Silenced: Handled by Model Manifest
            pass
        
        # Gating per head (Riemannian Wormholes)
        self.gatings = nn.ModuleList([
            RiemannianGating(self.head_dim, topology=self.topology_id) for _ in range(heads)
        ])
        
        # Static/Initial Wormhole Tuning
        scale_vals = []
        for i in range(heads):
            # Level 33: Alignment with base_dt
            # We want softplus(val) \approx base_dt / 0.9 initially
            target_dt = self.base_dt / 0.9
            val_init = torch.tensor(target_dt).exp().sub(1.0).log()
            val = val_init + i * 0.1 # Small spread
            scale_vals.append(val)
            
        self.dt_params = nn.Parameter(torch.tensor(scale_vals))
        self.time_heads = None

        self.friction_gates = nn.ModuleList()
        # Stacked weights for Fused Kernel efficiency [Heads, Out, In]
        # We will manually stack these into W_forget_stack, W_input_stack etc.
        
        for i in range(heads):
            # Input: x (head_dim) + Force (head_dim) -> Output: Friction (head_dim)
            # For Torus, x is mapped to 2*D features
            # LINK TO CHRISTOFFEL GATES IF AVAILABLE
            if hasattr(self.christoffels[i], 'forget_gate'):
                 gate = self.christoffels[i].forget_gate
            else:
                 gate_in_dim = (3 if self.topology_id == 1 else 2) * self.head_dim
                 gate = nn.Linear(gate_in_dim, self.head_dim) 
                 nn.init.orthogonal_(gate.weight, gain=0.5)
                 nn.init.constant_(gate.bias, 0.0) 
            
            self.friction_gates.append(gate)

        # Pre-compute stacked weights for the kernel
        # We'll do this dynamically in forward or cache it.
        
        self.integrators = nn.ModuleList()
        for i in range(heads):
            # Integrator setup
            if integrator_type == 'rk4':
                integ = RK4Integrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'rk45':
                # Golden Integration
                integ = DormandPrinceIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'heun':
                integ = HeunIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'euler':
                integ = EulerIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'leapfrog':
                integ = LeapfrogIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'yoshida':
                integ = YoshidaIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'forest_ruth':
                integ = ForestRuthIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'omelyan':
                integ = OmelyanIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'coupling':
                integ = CouplingFlowIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'neural':
                # Neural integrator needs dim to build its controller
                integ = NeuralIntegrator(self.christoffels[i], dt=self.base_dt, dim=self.head_dim)
            else:
                integ = SymplecticIntegrator(self.christoffels[i], dt=self.base_dt)
            self.integrators.append(integ)
            
        # Output projection for mixing heads
        if heads > 1:
            # LEVEL 28: PERIODIC MIXING
            # Input to mixing is now [sin(x), cos(x), v]
            self.out_proj_x = nn.Linear(3 * dim if self.topology_id == 1 else dim, dim)
            self.out_proj_v = nn.Linear(dim, dim)
            
            # Post-Mixing Normalization
            self.mixed_norm_x = nn.RMSNorm(dim)
            self.mixed_norm_v = nn.RMSNorm(dim)
            
            nn.init.xavier_uniform_(self.out_proj_x.weight)
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
        
        # LEVEL 17: TOPOLOGICAL INTEGRITY - REMOVE LAYERNORM
        # Hidden states (x, v) are passed raw. LayerNorm violates Periodicity.
        # x_heads = x.view(batch, self.heads, self.head_dim).transpose(0, 1)
        # v_heads = v.view(batch, self.heads, self.head_dim).transpose(0, 1)
        
        # We manually split heads
        x_heads = x.view(batch, self.heads, self.head_dim).permute(1, 0, 2).contiguous() # [H, B, D]
        v_heads = v.view(batch, self.heads, self.head_dim).permute(1, 0, 2).contiguous()
        
        if force is not None:
             if self.use_recursive and context is not None:
                 force = force + self.context_proj(context)
             f_heads = force.view(batch, self.heads, self.head_dim).permute(1, 0, 2).contiguous()
        else:
             f_heads = torch.zeros_like(x_heads)
             
        # 3. Vectorized Gating [Heads, Batch, 1]
        dt_base = torch.nn.functional.softplus(self.dt_params).view(self.heads, 1, 1)
        
        # Only apply dynamic gating if enabled in config
        use_gating = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False)
        
        if use_gating:
            gates = torch.stack([self.gatings[i](x_heads[i]) for i in range(self.heads)], dim=0)
            scale = dt_base * gates # [Heads, Batch, 1]
        else:
            # Define dummy gates for context passing (All 1.0 = Fully Open)
            gates = torch.ones(self.heads, batch, 1, device=x.device, dtype=x.dtype)
            scale = dt_base # Static scaling matching CUDA kernel default
        
        # 4. CLUTCH PARAMETER STACKING
        # We need to construct W_forget, W_input for the kernel.
        # The kernel expects:
        # W_forget: [L*H, D, D] acting on x
        # W_input: [L*H, D, D] acting on Force
        # For efficiency, we just cat the weights from our Linear modules.
        # Linear(2D -> D) weights are [D, 2D]. We split them.
        
        W_f_list = []
        W_i_list = []
        b_f_list = []
        
        for i in range(self.heads):
            # weight: [Out, In]
            # Handle separate gates or combined gate from Christoffel
            head_geo = self.christoffels[i]
            
            if hasattr(head_geo, 'forget_gate') and hasattr(head_geo, 'input_gate'):
                # Professional Path: Separate State and Force Gates
                W_f_list.append(head_geo.forget_gate.weight)
                W_i_list.append(head_geo.input_gate.weight)
                b_f_list.append(head_geo.forget_gate.bias)
            else:
                # Legacy/Combined Path
                w = self.friction_gates[i].weight
                b = self.friction_gates[i].bias
                d = self.head_dim
                
                if self.topology_id == 1:
                    # W_forget = [D, 2D] (sin, cos)
                    # W_input = [D, D] (force)
                    W_f_list.append(w[:, :2*d])
                    W_i_list.append(w[:, 2*d:])
                else:
                    W_f_list.append(w[:, :d])
                    W_i_list.append(w[:, d:])
                b_f_list.append(b)
        
        # Note: W_forget_stack will have shape [H, D, 2D] if torus
        W_forget_stack = torch.stack(W_f_list, dim=0).contiguous()
        W_input_stack = torch.stack(W_i_list, dim=0).contiguous() 
        b_forget_stack = torch.stack(b_f_list, dim=0).contiguous()

        # 5. Batched Geodesic Step via Fused Kernel
        x_outs = []
        v_outs = []
        christoffel_outputs = []
        
        # Recurrent Manifold Fused (Supports multi-head internally now)
        # We can pass the whole stack to one kernel call if we restructure.
        # Ideally, we call `recurrent_manifold_fused_autograd` ONCE for all heads.
        # The current python wrapper expects [B, T, D] but our heads are independent.
        # We can simulate a "Batch = Batch * Heads" call if weights are shared, but they aren't.
        # However, the Kernel v5.0 supports `U_stack` [H, D, R].
        # It processes heads in parallel blocks.
        
        # Let's perform the call PER HEAD using the legacy interface for safety now,
        # but passing the clutch weights.
        
        for i in range(self.heads):
            # Step for head i
            # The integrator now needs to accept Clutch weights if it supports them.
            # Currently `self.integrators[i]` is a wrapper. We need to bypass it 
            # or update it if we want to use the full Clutch power.
            # But `recurrent_manifold_fused` is the one that has the kernel logic.
            # We are inside MLayer which assumes a single step (seq_len=1 usually? No, Transfomer is T-major).
            
            # Wait, `MLayer` is usually called per-step in a loop in `model.py` OR 
            # it processes the whole sequence? 
            # In `model.py`, `Manifold.forward` calls `self.layers[i](x, v, f)`.
            # If input is [B, T, D], MLayer processes the sequence.
            # Let's assume sequence for maximum efficiency.
            
            # But the Integrator class interface is `step(x, v, f, dt)`.
            # To use `recurrent_manifold_fused`, we need the sequence input.
            # If inputs are [B, D] (one step), we assume T=1.
            
            # We will pass the specific Clutch weights for this head to the integrator.
            # Note: Integrator wrappers in `integrators.py` might need updating.
            # FOR NOW: We rely on the fact that `recurrent_manifold_fused` is called if available.
            # We need to pass `W_forget_stack=...` to the integrator.
            
            # Note: We attach the clutch weights to `self.christoffels[i]` or pass via kwargs.
            # Most clean way: Update `Integrator.step` to accept `clutch_params`.
            
            # For this Level 13 Fix, we assume we are using `recurrent_manifold_fused` directly
            # or we patch the integrator call.
            
            extra_kwargs = {
                'W_forget_stack': W_forget_stack[i:i+1], # [1, D, D]
                'W_input_stack': W_input_stack[i:i+1],
                'b_forget_stack': b_forget_stack[i:i+1],
                'topology': self.topology_id,
                'collect_christ': collect_christ
            }
            
            xh, vh = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale[i], **extra_kwargs)
            
            x_outs.append(xh)
            v_outs.append(vh)

        # 6. Concatenate and Mix (Standard)
        if self.heads > 1 and not collect_christ:
            try:
                from gfn.cuda.ops import head_mixing_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE and x.is_cuda:
                    x_stacked = torch.stack(x_outs, dim=0)
                    v_stacked = torch.stack(v_outs, dim=0)
                    x_next, v_next = head_mixing_fused(x_stacked, v_stacked, self.out_proj_x.weight, self.out_proj_v.weight)
                    context_next = gates.squeeze(-1).transpose(0, 1)
                    return x_next, v_next, context_next, christoffel_outputs
            except: pass
        
        # 6. Concatenate and Mix
        x_cat = torch.stack(x_outs, dim=1).view(batch, -1)
        v_cat = torch.stack(v_outs, dim=1).view(batch, -1)
        
        if self.heads > 1:
            if self.topology_id == 1:
                 # PERIODIC MIXING: Mixer sees [sin(x), cos(x), v]
                 v_mix = torch.tanh(v_cat / 100.0)
                 mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
                 x_next = self.out_proj_x(mixer_in_x)
            else:
                 x_next = self.out_proj_x(x_cat)
            
            v_next = self.out_proj_v(v_cat)
            
            # Normalize to prevent magnitude creep (Bypass for Torus to preserve phase)
            # Normalize to prevent magnitude creep (Bypass for Torus to preserve phase)
            if self.topology_id != 1:
                x_next = self.mixed_norm_x(x_next)
            else:
                # LEVEL 35: TORUS WRAPPING
                # Project back to [0, 2pi] to maintain precision
                PI = 3.14159265359
                TWO_PI = 2.0 * PI
                x_next = torch.remainder(x_next, TWO_PI)
                
            v_next = self.mixed_norm_v(v_next)
        else:
            x_next, v_next = x_cat, v_cat
            
        # Velocity Saturation (Relativistic Bounding)
        # LEVEL 34: INCREASED TO 100.0 TO REMOVE "THE BRAKE"
        v_next = 100.0 * torch.tanh(v_next / 100.0)
            
        context_next = gates.squeeze(-1).transpose(0, 1)
        return x_next, v_next, context_next, christoffel_outputs
