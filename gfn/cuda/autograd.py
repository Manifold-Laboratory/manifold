
import torch
from torch.autograd import Function
try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

# Import CUDA availability from ops.py
try:
    from .ops import CUDA_AVAILABLE, gfn_cuda
except ImportError:
    try:
        from gfn.cuda.ops import CUDA_AVAILABLE, gfn_cuda
    except ImportError:
        CUDA_AVAILABLE = False
        gfn_cuda = None

class ChristoffelFusedFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.topology = topology
        ctx.R, ctx.r = R, r
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        return gfn_cuda.christoffel_fused(v, U, W, x_in, V_w_in, plasticity, sing_thresh, sing_strength, topology, R, r)

    @staticmethod
    def backward(ctx, grad_gamma):
        v, U, W, x, V_w = ctx.saved_tensors
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        grads = gfn_cuda.christoffel_backward(grad_gamma.contiguous(), v, U, W, x_in, V_w_in, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology, ctx.R, ctx.r)
        gv, gU, gW, gx, gV = grads
        return gv, gU, gW, (gx if x is not None else None), (gV if V_w is not None else None), None, None, None, None, None, None

def christoffel_fused_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not v.is_cuda: return None
    return ChristoffelFusedFn.apply(v.contiguous(), U.contiguous(), W.contiguous(), x.contiguous() if x is not None else None, V_w.contiguous() if V_w is not None else None, plasticity, sing_thresh, sing_strength, topology, R, r)

class ReactiveChristoffelFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
        if v.dim() == 3: return None
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.topology = topology
        ctx.R, ctx.r = R, r
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        return gfn_cuda.reactive_christoffel_forward(v, U, W, x_in, V_w_in, plasticity, sing_thresh, sing_strength, topology, R, r)

    @staticmethod
    def backward(ctx, grad_gamma):
        # Fallback to christoffel_backward for now
        v, U, W, x, V_w = ctx.saved_tensors
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        grads = gfn_cuda.christoffel_backward(grad_gamma.contiguous(), v, U, W, x_in, V_w_in, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology, ctx.R, ctx.r)
        gv, gU, gW, gx, gV = grads
        return gv, gU, gW, (gx if x is not None else None), (gV if V_w is not None else None), None, None, None, None, None, None

def reactive_christoffel_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not v.is_cuda: return None
    return ReactiveChristoffelFn.apply(v.contiguous(), U.contiguous(), W.contiguous(), x.contiguous() if x is not None else None, V_w.contiguous() if V_w is not None else None, plasticity, sing_thresh, sing_strength, topology, R, r)

class LeapfrogFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scale, steps, topology=0, Wf=None, bf=None, plasticity=0.0, R=2.0, r=1.0):
        ctx.save_for_backward(x, v, f, U, W)
        ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology = dt, dt_scale, steps, topology
        ctx.R, ctx.r = R, r
        f_in = f if f is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        wf_in = Wf if Wf is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        bf_in = bf if bf is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        # Call fused kernel
        res = gfn_cuda.leapfrog_fused(x.contiguous(), v.contiguous(), f_in.contiguous(), U.contiguous(), W.contiguous(), wf_in.contiguous(), bf_in.contiguous(), dt, dt_scale, steps, topology, plasticity, R, r)
        return res[0], res[1]

    @staticmethod
    def backward(ctx, grad_xn, grad_vn):
        x, v, f, U, W = ctx.saved_tensors
        f_in = f if f is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        if isinstance(ctx.dt_scale, torch.Tensor):
            dt_scale = float(ctx.dt_scale.reshape(-1)[0].item()) if ctx.dt_scale.numel() > 0 else 1.0
        else:
            dt_scale = float(ctx.dt_scale)
        
        # Correct call to bound C++ function
        grads = gfn_cuda.leapfrog_backward(
            grad_xn.contiguous(), grad_vn.contiguous(), 
            x.contiguous(), v.contiguous(), f_in.contiguous(), 
            U.contiguous(), W.contiguous(),
            ctx.dt, dt_scale, ctx.steps, ctx.topology, ctx.R, ctx.r
        )
        gx, gv, gf, gU, gW = grads
        
        return gx, gv, (gf if f is not None else None), gU, gW, None, None, None, None, None, None, None, None, None

def leapfrog_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology=0, Wf=None, bf=None, plasticity=0.0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return LeapfrogFusedFn.apply(x, v, f, U, W, dt, dt_scale, steps, topology, Wf, bf, plasticity, R, r)

# --- New integrators ---

class EulerFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
        ctx.save_for_backward(x, v, f, U, W)
        ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology = dt, dt_scale, steps, topology
        ctx.R, ctx.r = R, r
        f_in = f if f is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        # Forward relies on CUDA kernel; ops fallback handles missing support
        return gfn_cuda.euler_fused(x.contiguous(), v.contiguous(), f_in.contiguous(), U.contiguous(), W.contiguous(), dt, dt_scale, steps, topology, R, r)

    @staticmethod
    def backward(ctx, grad_xn, grad_vn):
        x, v, f, U, W = ctx.saved_tensors
        dt, dt_scale, steps, topology = ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology
        R, r = ctx.R, ctx.r
        
        # Replay the forward to recover intermediates for gradients
        
        curr_x, curr_v = x.detach(), v.detach()
        f_in = f if f is not None else torch.zeros_like(x)
        h = dt * dt_scale
        
        # Store trajectory for reverse pass
        x_history = [curr_x.clone()]
        v_history = [curr_v.clone()]
        
        with torch.no_grad():
            for _ in range(steps):
                acc = gfn_cuda.christoffel_fused(curr_v, U, W, curr_x, None, 0.0, 1.0, 1.0, topology, R, r)
                if f is not None: acc = -acc + f_in
                else: acc = -acc
                curr_x = curr_x + h * curr_v
                curr_v = curr_v + h * acc
                x_history.append(curr_x.clone())
                v_history.append(curr_v.clone())
        
        # Adjoint-style backward pass
        gx, gv = grad_xn, grad_vn
        gf = torch.zeros_like(f_in) if f is not None else None
        gU = torch.zeros_like(U)
        gW = torch.zeros_like(W)
        
        for t in reversed(range(steps)):
            xt, vt = x_history[t], v_history[t]
            
            # v update gradient
            g_acc = h * gv
            
            # acc = f - gamma
            if gf is not None: gf += g_acc
            
            # gamma backward
            gamma_grads = gfn_cuda.christoffel_backward(-g_acc.contiguous(), vt, U, W, xt, None, 0.0, 1.0, 1.0, topology, R, r)
            
            gv = gv + gamma_grads[0]
            gU = gU + gamma_grads[1]
            gW = gW + gamma_grads[2]
            gx = gx + gamma_grads[3]
            
            # x update gradient
            gv = gv + h * gx
            
        return gx, gv, gf, gU, gW, None, None, None, None, None, None

def euler_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return EulerFusedFn.apply(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)

class HeunFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
        f_in = f if f is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, v, f_in, U, W)
        ctx.has_f = f is not None
        ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology = dt, dt_scale, steps, topology
        ctx.R, ctx.r = R, r
        return gfn_cuda.heun_fused(x.contiguous(), v.contiguous(), f_in.contiguous(), U.contiguous(), W.contiguous(), dt, dt_scale, steps, topology, R, r)
    
    @staticmethod
    def backward(ctx, gx, gv):
        x, v, f_in, U, W = ctx.saved_tensors
        dt, dt_scale, steps, topology = ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology
        R, r = ctx.R, ctx.r
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            U_req = U.detach().requires_grad_(True)
            W_req = W.detach().requires_grad_(True)
            f_req = f_in.detach().requires_grad_(True) if ctx.has_f else None
            xn, vn = _heun_recompute(x_req, v_req, f_req, U_req, W_req, dt, dt_scale, steps, topology, R, r)
            grads = torch.autograd.grad((xn, vn), (x_req, v_req, f_req, U_req, W_req), grad_outputs=(gx, gv), allow_unused=True)
        gx_out, gv_out, gf_out, gU_out, gW_out = grads
        if not ctx.has_f:
            gf_out = None
        return gx_out, gv_out, gf_out, gU_out, gW_out, None, None, None, None, None, None

def heun_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return HeunFusedFn.apply(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)

class RK4FusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
        f_in = f if f is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, v, f_in, U, W)
        ctx.has_f = f is not None
        ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology = dt, dt_scale, steps, topology
        ctx.R, ctx.r = R, r
        return gfn_cuda.rk4_fused(x.contiguous(), v.contiguous(), f_in.contiguous(), U.contiguous(), W.contiguous(), dt, dt_scale, steps, topology, R, r)
    
    @staticmethod
    def backward(ctx, gx, gv):
        x, v, f_in, U, W = ctx.saved_tensors
        dt, dt_scale, steps, topology = ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology
        R, r = ctx.R, ctx.r
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            U_req = U.detach().requires_grad_(True)
            W_req = W.detach().requires_grad_(True)
            f_req = f_in.detach().requires_grad_(True) if ctx.has_f else None
            xn, vn = _rk4_recompute(x_req, v_req, f_req, U_req, W_req, dt, dt_scale, steps, topology, R, r)
            grads = torch.autograd.grad((xn, vn), (x_req, v_req, f_req, U_req, W_req), grad_outputs=(gx, gv), allow_unused=True)
        gx_out, gv_out, gf_out, gU_out, gW_out = grads
        if not ctx.has_f:
            gf_out = None
        return gx_out, gv_out, gf_out, gU_out, gW_out, None, None, None, None, None, None

def rk4_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return RK4FusedFn.apply(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)

class VerletFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
        f_in = f if f is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, v, f_in, U, W)
        ctx.has_f = f is not None
        ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology = dt, dt_scale, steps, topology
        ctx.R, ctx.r = R, r
        return gfn_cuda.verlet_fused(x.contiguous(), v.contiguous(), f_in.contiguous(), U.contiguous(), W.contiguous(), dt, dt_scale, steps, topology, R, r)
    
    @staticmethod
    def backward(ctx, gx, gv):
        x, v, f_in, U, W = ctx.saved_tensors
        dt, dt_scale, steps, topology = ctx.dt, ctx.dt_scale, ctx.steps, ctx.topology
        R, r = ctx.R, ctx.r
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            U_req = U.detach().requires_grad_(True)
            W_req = W.detach().requires_grad_(True)
            f_req = f_in.detach().requires_grad_(True) if ctx.has_f else None
            xn, vn = _verlet_recompute(x_req, v_req, f_req, U_req, W_req, dt, dt_scale, steps, topology, R, r)
            grads = torch.autograd.grad((xn, vn), (x_req, v_req, f_req, U_req, W_req), grad_outputs=(gx, gv), allow_unused=True)
        gx_out, gv_out, gf_out, gU_out, gW_out = grads
        if not ctx.has_f:
            gf_out = None
        return gx_out, gv_out, gf_out, gU_out, gW_out, None, None, None, None, None, None

def _christoffel_eval(v, U, W, x, topology, R, r):
    return ChristoffelFusedFn.apply(v, U, W, x, None, 0.0, 1.0, 1.0, topology, R, r)

def _heun_recompute(x, v, f, U, W, dt, dt_scale, steps, topology, R, r):
    curr_x, curr_v = x, v
    for _ in range(steps):
        step_dt = dt * dt_scale
        k1_x = curr_v
        k1_v = -_christoffel_eval(curr_v, U, W, curr_x, topology, R, r)
        if f is not None:
            k1_v = k1_v + f
        v_pred = curr_v + step_dt * k1_v
        x_pred = apply_boundary_python(curr_x + step_dt * k1_x, topology)
        k2_x = v_pred
        k2_v = -_christoffel_eval(v_pred, U, W, x_pred, topology, R, r)
        if f is not None:
            k2_v = k2_v + f
        curr_x = curr_x + (step_dt * 0.5) * (k1_x + k2_x)
        curr_v = curr_v + (step_dt * 0.5) * (k1_v + k2_v)
        curr_x = apply_boundary_python(curr_x, topology)
    return curr_x, curr_v

def _rk4_recompute(x, v, f, U, W, dt, dt_scale, steps, topology, R, r):
    curr_x, curr_v = x, v
    for _ in range(steps):
        step_dt = dt * dt_scale
        dx1 = curr_v
        dv1 = -_christoffel_eval(curr_v, U, W, curr_x, topology, R, r)
        if f is not None:
            dv1 = dv1 + f
        v2 = curr_v + 0.5 * step_dt * dv1
        x2 = apply_boundary_python(curr_x + 0.5 * step_dt * dx1, topology)
        dx2 = v2
        dv2 = -_christoffel_eval(v2, U, W, x2, topology, R, r)
        if f is not None:
            dv2 = dv2 + f
        v3 = curr_v + 0.5 * step_dt * dv2
        x3 = apply_boundary_python(curr_x + 0.5 * step_dt * dx2, topology)
        dx3 = v3
        dv3 = -_christoffel_eval(v3, U, W, x3, topology, R, r)
        if f is not None:
            dv3 = dv3 + f
        v4 = curr_v + step_dt * dv3
        x4 = apply_boundary_python(curr_x + step_dt * dx3, topology)
        dx4 = v4
        dv4 = -_christoffel_eval(v4, U, W, x4, topology, R, r)
        if f is not None:
            dv4 = dv4 + f
        curr_x = curr_x + (step_dt / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
        curr_x = apply_boundary_python(curr_x, topology)
        curr_v = curr_v + (step_dt / 6.0) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
    return curr_x, curr_v

def _verlet_recompute(x, v, f, U, W, dt, dt_scale, steps, topology, R, r):
    curr_x, curr_v = x, v
    for _ in range(steps):
        step_dt = dt * dt_scale
        gamma = _christoffel_eval(curr_v, U, W, curr_x, topology, R, r)
        if f is None:
            a = -gamma
        else:
            a = -gamma + f
        v_half = curr_v + 0.5 * step_dt * a
        curr_x = curr_x + step_dt * v_half
        curr_x = apply_boundary_python(curr_x, topology)
        gamma_next = _christoffel_eval(v_half, U, W, curr_x, topology, R, r)
        if f is None:
            a_next = -gamma_next
        else:
            a_next = -gamma_next + f
        curr_v = v_half + 0.5 * step_dt * a_next
    return curr_x, curr_v

def verlet_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return VerletFusedFn.apply(x, v, f, U, W, dt, dt_scale, steps, topology, R, r)

class RecurrentManifoldFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, 
                mix_x=None, mix_v=None, W_forget_stack=None, W_input_stack=None, b_forget_stack=None, 
                W_potential_stack=None, b_potential_stack=None, 
                topology=0, R=2.0, r=1.0):
        # Placeholders for optional tensors
        mx = mix_x if mix_x is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        mv = mix_v if mix_v is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        wf = W_forget_stack if W_forget_stack is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        wi = W_input_stack if W_input_stack is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        bf = b_forget_stack if b_forget_stack is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        wp = W_potential_stack if W_potential_stack is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        bp = b_potential_stack if b_potential_stack is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            res = gfn_cuda.recurrent_manifold_fused(
                x_state=x.detach().clone(), v_state=v.detach().clone(), forces=f, U_stack=U, W_stack=W, 
                dt=dt, dt_scales=dt_scales, forget_rates=forget_rates, num_heads=num_heads, 
                plasticity=plasticity, sing_thresh=sing_thresh, sing_strength=sing_strength, 
                mix_x=mx, mix_v=mv, W_forget_stack=wf, W_input_stack=wi, b_forget_stack=bf, 
                W_potential_stack=wp, b_potential_stack=bp, 
                topology=topology, R=R, r=r
            )
            xf, vf, x_seq, v_seq, reg_loss = res
            
        ctx.save_for_backward(x, v, f, U, W, dt_scales, forget_rates, mix_x, mix_v, W_forget_stack, W_input_stack, b_forget_stack, W_potential_stack, b_potential_stack, x_seq, v_seq)
        ctx.dt, ctx.num_heads, ctx.topology = dt, num_heads, topology
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.R, ctx.r = R, r
        
        return xf, vf, x_seq, reg_loss

    @staticmethod
    def backward(ctx, grad_xf, grad_vf, grad_seq, grad_reg):
        x0, v0, f_seq, U, W, dt_scales, forget, mix_x, mix_v, wf, wi, bf, wp, bp, x_seq, v_seq = ctx.saved_tensors
        
        grads = gfn_cuda.recurrent_manifold_backward(
            grad_x_seq=grad_seq.contiguous() if grad_seq is not None else torch.zeros_like(x_seq),
            grad_x_final=grad_xf.contiguous() if grad_xf is not None else torch.zeros_like(x0),
            grad_v_final=grad_vf.contiguous() if grad_vf is not None else torch.zeros_like(v0),
            x_init=x0, v_init=v0, x_seq=x_seq, v_seq=v_seq, forces=f_seq, U_stack=U, W_stack=W,
            dt=ctx.dt, dt_scales=dt_scales, forget_rates=forget, num_heads=ctx.num_heads, 
            plasticity=ctx.plasticity, sing_thresh=ctx.sing_thresh, sing_strength=ctx.sing_strength,
            mix_x=mix_x if mix_x is not None else torch.empty(0, device=x0.device),
            mix_v=mix_v if mix_v is not None else torch.empty(0, device=x0.device),
            W_forget_stack=wf if wf is not None else torch.empty(0, device=x0.device),
            W_input_stack=wi if wi is not None else torch.empty(0, device=x0.device),
            b_forget_stack=bf if bf is not None else torch.empty(0, device=x0.device),
            W_potential_stack=wp if wp is not None else torch.empty(0, device=x0.device),
            b_potential_stack=bp if bp is not None else torch.empty(0, device=x0.device),
            topology=ctx.topology, R=ctx.R, r=ctx.r
        )
        
        gx0, gv0, gf, gU, gW, gmx, gmv, gfr, gwf, gwi, gbf, gwp, gbp = grads
        
        return (
            gx0, gv0, gf, gU, gW, 
            None, # dt
            None, # dt_scales
            gfr,
            None, # num_heads
            None, # plasticity
            None, # sing_thresh
            None, # sing_strength
            (gmx if mix_x is not None else None),
            (gmv if mix_v is not None else None),
            (gwf if wf is not None else None),
            (gwi if wi is not None else None),
            (gbf if bf is not None else None),
            (gwp if wp is not None else None),
            (gbp if bp is not None else None),
            None, # topology
            None, # R
            None  # r
        )

def recurrent_manifold_fused_autograd(x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, mix_x=None, mix_v=None, W_forget_stack=None, W_input_stack=None, b_forget_stack=None, W_potential_stack=None, b_potential_stack=None, topology=0, R=2.0, r=1.0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return RecurrentManifoldFusedFn.apply(x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x, mix_v, W_forget_stack, W_input_stack, b_forget_stack, W_potential_stack, b_potential_stack, topology, R, r)
