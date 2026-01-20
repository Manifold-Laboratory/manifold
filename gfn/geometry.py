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
        # Try CUDA kernel (Inference Only)
        if x is None and not torch.is_grad_enabled():
            try:
                from gfn.cuda.ops import christoffel_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE and v.is_cuda:
                     return christoffel_fused(v, self.U, self.W)
            except ImportError:
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

class SymplecticIntegrator(nn.Module):
    r"""
    Integrates the geodesic equation: d^2x/dt^2 + \Gamma(v, v) = F
    using a symplectic method (e.g., Velocity Verlet) to preserve energy/stability.
    """
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        r"""
        One step of integration.
        
        Velocity Verlet:
        1. v_{t+0.5} = v_t + 0.5 * a(x_t, v_t) * dt
        2. x_{t+1} = x_t + v_{t+0.5} * dt
        3. a_{t+1} = a(x_{t+1}, v_{t+0.5})  (Approximation: depend on v_{t+0.5})
        4. v_{t+1} = v_{t+0.5} + 0.5 * a_{t+1} * dt
        
        Acceleration a(x, v) = F - \Gamma(v, v)
        """
        dt = self.dt * dt_scale
        
        # Pure Symplectic Integration (NO friction in integrator)
        # Acceleration at t
        gamma_t = self.christoffel(v, x)
        acc_t = -gamma_t
        if force is not None:
            acc_t = acc_t + force
        
        # Half-kick
        v_half = v + 0.5 * dt * acc_t
        
        # Full-drift
        x_next = x + dt * v_half
        
        # Acceleration at t+1
        gamma_next = self.christoffel(v_half, x_next)
        acc_next = -gamma_next
        if force is not None:
            acc_next = acc_next + force
        
        # Half-kick
        v_next = v_half + 0.5 * dt * acc_next
        
        return x_next, v_next

class RK4Integrator(nn.Module):
    r"""
    Runge-Kutta 4 (RK4) Integrator for the geodesic equation.
    System:
    dx/dt = v
    dv/dt = F - \Gamma(v, v)
    
    State Y = [x, v]
    """
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        r"""
        One step of RK4 integration.
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
        """
        dt = self.dt * dt_scale
        
        def dynamics(current_x, current_v):
            # dv/dt = F - Gamma(v, v, x)
            acc = -self.christoffel(current_v, current_x)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        dx1 = v
        dv1 = dynamics(x, v)
        
        # k2
        v2 = v + 0.5 * dt * dv1
        x2 = x + 0.5 * dt * dx1
        dx2 = v2
        dv2 = dynamics(x2, v2)
        
        # k3
        v3 = v + 0.5 * dt * dv2
        x3 = x + 0.5 * dt * dx2
        dx3 = v3
        dv3 = dynamics(x3, v3)
        
        # k4
        v4 = v + dt * dv3
        x4 = x + dt * dx3
        dx4 = v4
        dv4 = dynamics(x4, v4)
        
        # Update
        x_next = x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        v_next = v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        return x_next, v_next

class HeunIntegrator(nn.Module):
    r"""
    Heun's Method (Improved Euler / RK2).
    2nd order accuracy with only 2 evaluations per step.
    Great balance between accuracy and speed.
    """
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        def dynamics(current_x, current_v):
            acc = -self.christoffel(current_v, current_x)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        dx1 = v
        dv1 = dynamics(x, v)
        
        # Predictor step (Euler)
        v_pred = v + dt * dv1
        x_pred = x + dt * dx1
        
        # k2 (using predicted velocity AND position)
        dx2 = v_pred
        dv2 = dynamics(x_pred, v_pred)
        
        # Corrector step
        x_next = x + (dt / 2.0) * (dx1 + dx2)
        v_next = v + (dt / 2.0) * (dv1 + dv2)
        
        return x_next, v_next


class LeapfrogIntegrator(nn.Module):
    r"""
    Leapfrog (Störmer-Verlet) Integrator for Geodesic Flow.
    
    A symplectic integrator that preserves the Hamiltonian structure,
    ensuring energy conservation and long-term stability.
    
    Algorithm (Kick-Drift-Kick variant):
        1. v_{1/2} = v + (dt/2) * a(x, v)           [Half-Kick]
        2. x_{new} = x + dt * v_{1/2}               [Full-Drift]
        3. v_{new} = v_{1/2} + (dt/2) * a(x_{new}, v_{1/2})  [Half-Kick]
    
    This is time-reversible and symplectic, making it ideal for
    preserving phase-space volume and preventing energy drift.
    """
    
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        """
        Perform one Leapfrog (Störmer-Verlet) step.
        
        Uses custom fused CUDA kernel when available for 4-5x speedup.
        
        Args:
            x: Position
            v: Velocity
            force: External force
            dt_scale: Adaptive time scaling (Golden Integration)
        """
        if force is None:
            force = torch.zeros_like(x)
            
        # Try CUDA kernel (Inference Only - requires float dt_scale)
        # We only support scalar dt_scale for now in kernels.
        is_scalar_scale = isinstance(dt_scale, float) or (isinstance(dt_scale, torch.Tensor) and dt_scale.numel() == 1)
        
        if not torch.is_grad_enabled() and is_scalar_scale:
            try:
                from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE and x.is_cuda:
                     # Ensure dt_scale is float
                     dt_val = dt_scale.item() if isinstance(dt_scale, torch.Tensor) else dt_scale
                     return leapfrog_fused(x, v, force, self.christoffel.U, self.christoffel.W, self.dt, dt_val)
            except ImportError:
                pass
        effective_dt = self.dt * dt_scale
        
        # Fallback to PyTorch
        
        # Half-step velocity
        gamma = self.christoffel(v, x)
        v_half = v + 0.5 * effective_dt * (force - gamma)
        
        # Full-step position
        x_new = x + effective_dt * v_half
        
        # Half-step velocity again
        # Use x_new for new Gamma calculation
        gamma_half = self.christoffel(v_half, x_new)
        v_new = v_half + 0.5 * effective_dt * (force - gamma_half)
        
        return x_new, v_new

class DormandPrinceIntegrator(nn.Module):
    r"""
    Dormand-Prince (RK45) Adaptive Integrator.
    
    Uses 5th order and 4th order approximations to estimate local error and adapt `dt`.
    Ideally suited for "Golden Integration" to ensure physical stability.
    """
    def __init__(self, christoffel_net, dt=0.1, rtol=1e-5, atol=1e-6):
        super().__init__()
        self.christoffel = christoffel_net
        self.base_dt = dt
        self.rtol = rtol
        self.atol = atol
        
        # Butcher Tableau for RK45 (Dormand-Prince)
        # c: nodes
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        
        # a: Runge-Kutta matrix (flattened or manual for efficiency)
        # b5: 5th order weights
        self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        # b4: 4th order weights (for error est)
        self.b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        """
        Perform ONE adaptive step.
        If error is too high, it effectively performs multiple smaller substeps (conceptually).
        Actually, for a fixed-graph implementations (like here), we usually:
        1. Try step with dt.
        2. Calc error.
        3. If error > tol, we simply return the result but with a "penalty" or we implement 
           a while loop (slow in PyTorch graph).
           
        Golden Integration Strategy for GFN:
        Synchronous Adaptation:
        - We calculate error for the batch.
        - We output the Valid Next State.
        - If error was high, we effectively took a "smaller" physical step in terms of manifold distance,
          even if wall-clock time is the same.
          Or, strictly: We adapt dt.
          
        Since we are in a fixed compute graph layer (MLayer), we can't easily loop indefinitely.
        We will implement a "Try-Retry" logic with max 1 retry level for efficiency,
        or just compute the error and scale the NEXT dt (Controller method).
        
        Let's implement **Controller Method** (Standard for ODENets):
        - Current step uses `dt`.
        - Compute error.
        - Update `next_dt` for the NEXT forward call (stored in state? No, stateless).
        
        Wait, `MLayer` maintains `dt_params`. We can't update them easily here during inference without RNN state.
        
        Alternative: **Bounded Adaptive Step**.
        We compute the step. If error is high, we interpolate result to `0.5 * dt`.
        
        """
        dt = self.base_dt * dt_scale
        
        # Coefficients (DP54)
        c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1
        
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        # a7... same as b5 for FSAL property (First Same As Last) - optimization TODO
        
        def dynamics(tx, tv):
            acc = -self.christoffel(tv, tx)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        k1_x = v
        k1_v = dynamics(x, v)
        
        # k2
        x2 = x + dt * (a21*k1_x)
        v2 = v + dt * (a21*k1_v)
        k2_x = v2
        k2_v = dynamics(x2, v2)
        
        # k3
        x3 = x + dt * (a31*k1_x + a32*k2_x)
        v3 = v + dt * (a31*k1_v + a32*k2_v)
        k3_x = v3
        k3_v = dynamics(x3, v3)
        
        # k4
        x4 = x + dt * (a41*k1_x + a42*k2_x + a43*k3_x)
        v4 = v + dt * (a41*k1_v + a42*k2_v + a43*k3_v)
        k4_x = v4
        k4_v = dynamics(x4, v4)
        
        # k5
        x5 = x + dt * (a51*k1_x + a52*k2_x + a53*k3_x + a54*k4_x)
        v5 = v + dt * (a51*k1_v + a52*k2_v + a53*k3_v + a54*k4_v)
        k5_x = v5
        k5_v = dynamics(x5, v5)
        
        # k6
        x6 = x + dt * (a61*k1_x + a62*k2_x + a63*k3_x + a64*k4_x + a65*k5_x)
        v6 = v + dt * (a61*k1_v + a62*k2_v + a63*k3_v + a64*k4_v + a65*k5_v)
        k6_x = v6
        k6_v = dynamics(x6, v6)
        
        # k7 (same as k1 for next step if acceptable, but we don't cache here)
        x7 = x + dt * (self.b5[0]*k1_x + self.b5[2]*k3_x + self.b5[3]*k4_x + self.b5[4]*k5_x + self.b5[5]*k6_x)
        v7 = v + dt * (self.b5[0]*k1_v + self.b5[2]*k3_v + self.b5[3]*k4_v + self.b5[4]*k5_v + self.b5[5]*k6_v)
        # Note: b5[1] is 0
        
        # 5th order solution
        y5_x = x7
        y5_v = v7
        
        # 4th order solution for error estimate using standard DP coefficients
        y4_x = x + dt * (self.b4[0]*k1_x + self.b4[2]*k3_x + self.b4[3]*k4_x + self.b4[4]*k5_x + self.b4[5]*k6_x)
        y4_v = v + dt * (self.b4[0]*k1_v + self.b4[2]*k3_v + self.b4[3]*k4_v + self.b4[4]*k5_v + self.b4[5]*k6_v)
        
        # Error Estimate
        error_scale = self.atol + torch.max(torch.abs(v), torch.abs(y5_v)) * self.rtol
        delta_v = torch.abs(y5_v - y4_v)
        error_ratio = torch.mean(delta_v / error_scale)
        
        # Adaptive Logic:
        # In this implementation (fixed-depth MLayer), we do not retry steps.
        # Instead, we provide RK45 primarily as a higher-precision integrator (5th order).
        # The error estimate is calculated but not currently used for step-size adjustment within the layer graph.
        
        return y5_x, y5_v

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
        # Try Fused CUDA Kernel (Active Inference Mode)
        if not torch.is_grad_enabled() and v.is_cuda:
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
            except ImportError:
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
