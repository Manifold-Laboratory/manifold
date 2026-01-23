
import torch
import torch.nn as nn
from torch.autograd import Function

# Try import extension
try:
    import gfn_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class ChristoffelFusedFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
        # Save context for backward
        # We need inputs to compute gradients
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        
        # Ensure optional tensors are compliant with C++ signature
        # We pass empty tensors if None, handled by C++ check
        v_in = v
        u_in = U
        w_in = W
        x_in = x if x is not None else torch.empty(0, device=v.device)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        # Call Forward Kernel
        gamma = gfn_cuda.christoffel_fused(
             v_in, u_in, w_in, x_in, V_w_in, 
             plasticity, sing_thresh, sing_strength
        )
        
        return gamma

    @staticmethod
    def backward(ctx, grad_gamma):
        # Retrieve saved tensors
        v, U, W, x, V_w = ctx.saved_tensors
        
        # Ensure contiguous memory (CUDA requirement)
        grad_gamma = grad_gamma.contiguous()
        
        # Prepare optional inputs
        x_in = x if x is not None else torch.empty(0, device=v.device)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        # Call Backward Kernel
        # Returns: grad_v, grad_U, grad_W, grad_x, grad_V_w
        grads = gfn_cuda.christoffel_backward(
            grad_gamma, v, U, W, x_in, V_w_in, 
            ctx.plasticity, ctx.sing_thresh, ctx.sing_strength
        )
        
        grad_v, grad_U, grad_W, grad_x, grad_V_w = grads
        
        # Determine if we return gradients for x and V_w
        # If input x was None, grad_x will be empty tensor or 0-like, we return None.
        gx_out = grad_x if x is not None else None
        gV_out = grad_V_w if V_w is not None else None
        
        # Return gradients matching forward signature:
        # v, U, W, x, V_w, plasticity, sing_thresh, sing_strength
        return grad_v, grad_U, grad_W, gx_out, gV_out, None, None, None

class YoshidaFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, V_w=None, dt=0.1, dt_scale=1.0, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
        ctx.save_for_backward(x, v, f, U, W, V_w)
        ctx.dt = dt
        ctx.dt_scale = dt_scale
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        
        # Handle optional V_w
        V_w_in = V_w if V_w is not None else torch.empty(0, device=x.device)
        if f is None: 
             f = torch.zeros_like(x)
        
        # Ensure dt_scale is contiguous if tensor
        dt_arg = dt_scale
        if isinstance(dt_scale, torch.Tensor):
            dt_arg = dt_scale.contiguous()
            if dt_arg.dtype != torch.float32:
                 dt_arg = dt_arg.float()

        return gfn_cuda.yoshida_fused(
            x, v, f, U, W, V_w_in, 
            dt, dt_arg, plasticity, sing_thresh, sing_strength
        )

    @staticmethod
    def backward(ctx, grad_x_new, grad_v_new):
        # Semi-Fused Backward (Recomputing steps in Python, calling CUDA kernels)
        x, v, f, U, W, V_w = ctx.saved_tensors
        dt = ctx.dt
        dt_scale = ctx.dt_scale
        plasticity = ctx.plasticity
        sing_thresh = ctx.sing_thresh
        sing_strength = ctx.sing_strength
        
        # Yoshida Params
        w0 = -1.7024143839193153
        w1 = 1.3512071919596578
        c1 = c4 = w1 / 2.0
        c2 = c3 = (w0 + w1) / 2.0
        d1 = d3 = w1
        d2 = w0
        eff_dt = dt * dt_scale
        
        V_safe = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        # Accumulator for dL/d(eff_dt)
        # Handle Batch Vectorization: [batch]
        g_eff_dt = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        # Recompute Forward States (Checkpointing to save memory)
        # S1
        x1 = x + c1 * eff_dt * v
        gamma1 = gfn_cuda.christoffel_fused(v, U, W, x1, V_safe, plasticity, sing_thresh, sing_strength)
        v1 = v + d1 * eff_dt * (f - gamma1)
        
        # S2
        x2 = x1 + c2 * eff_dt * v1
        gamma2 = gfn_cuda.christoffel_fused(v1, U, W, x2, V_safe, plasticity, sing_thresh, sing_strength)
        v2 = v1 + d2 * eff_dt * (f - gamma2)
        
        # S3
        x3 = x2 + c3 * eff_dt * v2
        gamma3 = gfn_cuda.christoffel_fused(v2, U, W, x3, V_safe, plasticity, sing_thresh, sing_strength)
        v3 = v2 + d3 * eff_dt * (f - gamma3)
        
        # === Backward Pass (Reverse Order) ===
        # Initial Gradients from Loss
        gx3 = grad_x_new
        # dL/dv3 = dL/dv_new + dL/dx_new * dx_new/dv3
        # x_new = x3 + c4 * eff_dt * v3
        # d(x_new)/d(eff_dt) = c4 * v3 (vector [batch, dim])
        # Contribution to dL/d(eff_dt) = Sum_dim(dL/dx_new * c4 * v3) -> [batch]
        gv3 = grad_v_new + grad_x_new * (c4 * eff_dt) 
        g_eff_dt += (grad_x_new * (c4 * v3)).sum(dim=1)
        
        # -- Step 3 --
        # v3 = v2 + d3*dt*(f - gamma3)
        # dL/d(eff_dt) via v3: dL/dv3 * d3 * (f - gamma3)
        g_eff_dt += (gv3 * d3 * (f - gamma3)).sum(dim=1)
        
        # dL/dgamma3 = -dL/dv3 * (d3 * dt)
        g_gamma3 = -gv3 * (d3 * eff_dt)
        
        cg_v2, cg_U3, cg_W3, cg_x3, cg_V_w3 = gfn_cuda.christoffel_backward(
            g_gamma3, v2, U, W, x3, V_safe, 
            plasticity, sing_thresh, sing_strength
        )
        
        # dL/dv2 = dL/dv3 + dL/dx3 * c3*dt + dL/dgamma3 * dgamma3/dv2
        #        = gv3 + gx3 * c3*eff_dt + cg_v2
        
        # dL/d(eff_dt) via x3 = x2 + c3*dt*v2
        # dL/dx3 * c3 * v2
        g_eff_dt += (gx3 * (c3 * v2)).sum(dim=1)
        
        gv2 = gv3 + gx3 * (c3 * eff_dt) + cg_v2
        gx2 = gx3 + cg_x3 
        
        gf = gv3 * (d3 * eff_dt)
        
        # -- Step 2 --
        # v2 = v1 + d2*dt*(f - gamma2)
        g_eff_dt += (gv2 * d2 * (f - gamma2)).sum(dim=1) # Accumulate eff_dt grad
        
        g_gamma2 = -gv2 * (d2 * eff_dt)
        cg_v1, cg_U2, cg_W2, cg_x2, cg_V_w2 = gfn_cuda.christoffel_backward(
            g_gamma2, v1, U, W, x2, V_safe, 
            plasticity, sing_thresh, sing_strength
        )
        
        # x2 = x1 + c2*dt*v1
        g_eff_dt += (gx2 * (c2 * v1)).sum(dim=1) # Accumulate eff_dt grad
        
        gv1 = gv2 + gx2 * (c2 * eff_dt) + cg_v1
        gx1 = gx2 + cg_x2
        gf += gv2 * (d2 * eff_dt)
        
        # -- Step 1 --
        # v1 = v + d1*dt*(f - gamma1)
        g_eff_dt += (gv1 * d1 * (f - gamma1)).sum(dim=1) # Accumulate eff_dt grad
        
        g_gamma1 = -gv1 * (d1 * eff_dt)
        cg_v, cg_U1, cg_W1, cg_x1, cg_V_w1 = gfn_cuda.christoffel_backward(
            g_gamma1, v, U, W, x1, V_safe, 
            plasticity, sing_thresh, sing_strength
        )
        
        # x1 = x + c1*dt*v
        g_eff_dt += (gx1 * (c1 * v)).sum(dim=1) # Accumulate eff_dt grad
        
        gv = gv1 + gx1 * (c1 * eff_dt) + cg_v
        gx = gx1 + cg_x1
        gf += gv1 * (d1 * eff_dt)
        
        # Accumulate Parameter Gradients
        gU = cg_U1 + cg_U2 + cg_U3
        gW = cg_W1 + cg_W2 + cg_W3
        
        gV_w = None
        if V_w is not None:
             gV_w = torch.zeros_like(V_w)
             if cg_V_w3.numel() > 0: gV_w += cg_V_w3
             if cg_V_w2.numel() > 0: gV_w += cg_V_w2
             if cg_V_w1.numel() > 0: gV_w += cg_V_w1
        
        # g_eff_dt is tensor [batch].
        # We need g_dt_scale = g_eff_dt * dt
        # If input dt_scale was tensor [batch], we return tensor [batch]
        # If input was scalar, we sum?
        # The backward logic should match input logic.
        g_dt_scale = g_eff_dt * dt
        
        # Check if input dt_scale was tensor logic (we don't have is_tensor flag easily, 
        # but YoshidaFusedFn is customized logic).
        # We return the tensor. Autograd engine handles reduction if input was scalar (via broadcasting logic).
        # Wait, if `dt_scale` was broadcasting, `g_dt_scale` should be summed by Autograd?
        # No, Functions must return exact shapes.
        # But `dt_scale` is saved in `ctx.dt_scale`.
        
        if isinstance(dt_scale, torch.Tensor):
            # Shape match logic
            if g_dt_scale.shape != dt_scale.shape:
                # If shapes differ, assume broadcasting happened (e.g. (1,) or scalar -> (batch,))
                # Reduce gradients to match input shape
                if dt_scale.numel() == 1:
                    g_dt_scale = g_dt_scale.sum().view(dt_scale.shape)
                else:
                    # Generic broadcasting reduction is complex, but for [1] vs [B], logic holds.
                    pass
        else:
            # Input was float. We return float (scalar tensor)
            g_dt_scale = g_dt_scale.sum()
            
        g_scale_tensor = g_dt_scale # scalar or tensor
        
        return gx, gv, gf, gU, gW, gV_w, None, g_scale_tensor, None, None, None

def christoffel_fused_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    """
    Autograd-compatible wrapper for Fused Christoffel Kernel.
    Ensures contiguous memory layout for CUDA kernels.
    """
    if not CUDA_AVAILABLE or not v.is_cuda:
        raise RuntimeError("GFN CUDA extension not available or tensor not on CUDA")
    
    # CRITICAL: Kernels assume contiguous memory (row-major).
    # MLayer passes sliced views (chunks), which are strided.
    v = v.contiguous()
    U = U.contiguous()
    W = W.contiguous()
    if x is not None: x = x.contiguous()
    if V_w is not None: V_w = V_w.contiguous()
        
    return ChristoffelFusedFn.apply(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength)

def yoshida_fused_autograd(x, v, f=None, U=None, W=None, V_w=None, dt=0.1, dt_scale=1.0, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    """
    Autograd-compatible wrapper for Fused Yoshida Integrator using Kernels.
    Ensures contiguous memory layout for CUDA kernels.
    """
    if U is None or W is None:
        raise ValueError("Yoshida requires Christoffel matrices U, W")
    if not CUDA_AVAILABLE or not x.is_cuda:
        raise RuntimeError("CUDA unavailable")
        
    if f is None:
        f = torch.zeros_like(x)
        
    # CRITICAL: Kernels assume contiguous memory (row-major).
    x = x.contiguous()
    v = v.contiguous()
    f = f.contiguous()
    U = U.contiguous()
    W = W.contiguous()
    if V_w is not None: V_w = V_w.contiguous()
        
    return YoshidaFusedFn.apply(x, v, f, U, W, V_w, dt, dt_scale, plasticity, sing_thresh, sing_strength)
