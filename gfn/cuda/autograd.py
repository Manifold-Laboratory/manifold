

import torch
import sys
from torch.autograd import Function

# Import CUDA_AVAILABLE from ops.py to ensure consistency
try:
    from .ops import CUDA_AVAILABLE, gfn_cuda
except ImportError:
    try:
        from gfn.cuda.ops import CUDA_AVAILABLE, gfn_cuda
    except ImportError:
        CUDA_AVAILABLE = False
        gfn_cuda = None

print(f"[DEBUG] gfn.cuda.autograd LOADED. CUDA_AVAILABLE={CUDA_AVAILABLE} from {__file__}")

class ChristoffelFusedFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        
        x_in = x if x is not None else torch.empty(0, device=v.device)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        return gfn_cuda.christoffel_fused(v, U, W, x_in, V_w_in, plasticity, sing_thresh, sing_strength)

    @staticmethod
    def backward(ctx, grad_gamma):
        v, U, W, x, V_w = ctx.saved_tensors
        x_in = x if x is not None else torch.empty(0, device=v.device)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        grads = gfn_cuda.christoffel_backward(
            grad_gamma.contiguous(), v, U, W, x_in, V_w_in, 
            ctx.plasticity, ctx.sing_thresh, ctx.sing_strength
        )
        
        gv, gU, gW, gx, gV = grads
        return gv, gU, gW, (gx if x is not None else None), (gV if V_w is not None else None), None, None, None

def christoffel_fused_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    if not CUDA_AVAILABLE or not v.is_cuda:
        # Fallback logic should be in ops.py
        return None
    return ChristoffelFusedFn.apply(v.contiguous(), U.contiguous(), W.contiguous(), 
                                   x.contiguous() if x is not None else None, 
                                   V_w.contiguous() if V_w is not None else None, 
                                   plasticity, sing_thresh, sing_strength)
class LowRankChristoffelFn(Function):
    @staticmethod
    def forward(ctx, v, U, W):
        ctx.save_for_backward(v, U, W)
        return gfn_cuda.lowrank_christoffel_forward(v, U, W)

    @staticmethod
    def backward(ctx, grad_gamma):
        v, U, W = ctx.saved_tensors
        grads = gfn_cuda.lowrank_christoffel_backward(grad_gamma.contiguous(), v, U, W)
        gv, gU, gW = grads
        return gv, gU, gW

def lowrank_christoffel_autograd(v, U, W):
    if not CUDA_AVAILABLE or not v.is_cuda:
        return None
    return LowRankChristoffelFn.apply(v.contiguous(), U.contiguous(), W.contiguous())

class ReactiveChristoffelFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        
        # Support 3D Batched Heads [H, B, d]
        v_in = v.reshape(-1, v.shape[-1]) if v.dim() == 3 else v
        x_in = x.reshape(-1, x.shape[-1]) if (x is not None and x.dim() == 3) else x
        x_in = x_in if x_in is not None else torch.empty(0, device=v.device)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        # U, W must be stacked correctly [H, d, R]
        # Kernel normally expects 2D U, W. If 3D, we must pass them carefully.
        # Currently, christoffel_fused kernel expects 2D U [D, R].
        # For professional speed, we assume U is already flattened or we loop if needed.
        # But wait, LowRankChristoffel now uses bmm for 3D.
        # Let's keep the CUDA kernel for 2D and use the optimized Python path for 3D
        # unless we want to rewrite the kernel.
        
        # For now, return None to trigger the optimized Python BMM fallback in ops.py for 3D
        if v.dim() == 3: return None
        
        return gfn_cuda.reactive_christoffel_forward(v_in, U, W, x_in, V_w_in, plasticity, sing_thresh, sing_strength)

    @staticmethod
    def backward(ctx, grad_gamma):
        v, U, W, x, V_w = ctx.saved_tensors
        x_in = x if x is not None else torch.empty(0, device=v.device)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device)
        
        grads = gfn_cuda.christoffel_backward(
            grad_gamma.contiguous(), v, U, W, x_in, V_w_in, 
            ctx.plasticity, ctx.sing_thresh, ctx.sing_strength
        )
        gv, gU, gW, gx, gV = grads
        return gv, gU, gW, (gx if x is not None else None), (gV if V_w is not None else None), None, None, None

def reactive_christoffel_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    if not CUDA_AVAILABLE or not v.is_cuda:
        return None
    return ReactiveChristoffelFn.apply(v.contiguous(), U.contiguous(), W.contiguous(), 
                                     x.contiguous() if x is not None else None, 
                                     V_w.contiguous() if V_w is not None else None, 
                                     plasticity, sing_thresh, sing_strength)

class LeapfrogFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scale, steps):
        """
        Supports [B, d] or [H, B, d] by flattening heads into batch dimension.
        Professional High-Throughput Path.
        """
        orig_shape = x.shape
        # Flatten [H, B, d] -> [H*B, d]
        x_flat = x.reshape(-1, x.shape[-1])
        v_flat = v.reshape(-1, v.shape[-1])
        f_flat = f.reshape(-1, f.shape[-1]) if f is not None else torch.zeros_like(x_flat)
        
        # If U/W are [H, d, R], we must tile them to [H*B, d, R] 
        # OR launch the kernel if it's head-aware.
        # CURRENT KERNEL LIMITATION: Expects shared U/W.
        # If heads have different U, we CANNOT flatten them into the same batch 
        # unless the kernel is rewritten. 
        # WORKAROUND: For training speed, we use the optimized Python BMM fallback 
        # if heads are independent, but keep the Fused Kernel for the sequence.
        
        if x.dim() == 3: return None # Triggers Python Vectorized Fallback (still fast)
        
        ctx.save_for_backward(x, v, f, U, W)
        ctx.dt, ctx.dt_scale, ctx.steps = dt, dt_scale, steps
        
        x_new, v_new = gfn_cuda.leapfrog_fused(x_flat.contiguous(), v_flat.contiguous(), 
                                              f_flat.contiguous(), U.contiguous(), W.contiguous(), 
                                              dt, dt_scale, steps)
        return x_new.view(orig_shape), v_new.view(orig_shape)
    
    @staticmethod
    def backward(ctx, grad_x_new, grad_v_new):
        x, v, f, U, W = ctx.saved_tensors
        f_in = f if f is not None else torch.empty(0, device=x.device)
        
        grads = gfn_cuda.leapfrog_backward(
            grad_x_new.contiguous(), grad_v_new.contiguous(),
            x, v, f_in, U, W,
            ctx.dt, ctx.dt_scale, ctx.steps
        )
        
        grad_x, grad_v, grad_f, grad_U, grad_W = grads
        return grad_x, grad_v, (grad_f if f is not None else None), grad_U, grad_W, None, None, None

def leapfrog_fused_autograd(x, v, f, U, W, dt, dt_scale, steps):
    if not CUDA_AVAILABLE or not x.is_cuda:
        return None
    return LeapfrogFusedFn.apply(x.contiguous(), v.contiguous(), 
                                 f.contiguous() if f is not None and f.numel() > 0 else None,
                                 U.contiguous(), W.contiguous(), 
                                 dt, dt_scale, steps)

class RecurrentManifoldFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x=None, mix_v=None):
        ctx.save_for_backward(x, v, f, U, W, dt_scales, forget_rates)
        ctx.dt, ctx.num_heads = dt, num_heads
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.mix_x, ctx.mix_v = mix_x, mix_v
        
        mix_x_in = mix_x if mix_x is not None else torch.empty(0, device=x.device)
        mix_v_in = mix_v if mix_v is not None else torch.empty(0, device=x.device)
        
        res = gfn_cuda.recurrent_manifold_fused(x.contiguous(), v.contiguous(), f.contiguous(), 
                                               U.contiguous(), W.contiguous(), dt, 
                                               dt_scales.contiguous(), forget_rates.contiguous(), num_heads,
                                               plasticity, sing_thresh, sing_strength,
                                               mix_x_in.contiguous(), mix_v_in.contiguous())
        x_state, v_state, x_out_seq, reg_loss = res
        return x_state, v_state, x_out_seq, reg_loss

    @staticmethod
    def backward(ctx, grad_x_final, grad_v_final, grad_x_seq, grad_reg_loss):
        x0, v0, f_seq, U_stack, W_stack, dt_scales, forget_rates = ctx.saved_tensors
        B, T, D = f_seq.shape
        H, dt = ctx.num_heads, ctx.dt
        pl, st, ss = ctx.plasticity, ctx.sing_thresh, ctx.sing_strength
        mix_x, mix_v = ctx.mix_x, ctx.mix_v
        
        mix_x_in = mix_x if mix_x is not None else torch.empty(0, device=x0.device)
        mix_v_in = mix_v if mix_v is not None else torch.empty(0, device=x0.device)
        gx_seq = grad_x_seq.contiguous() if grad_x_seq is not None else torch.empty(0, device=x0.device)
        gx_final = grad_x_final.contiguous() if grad_x_final is not None else torch.empty(0, device=x0.device)
        gv_final = grad_v_final.contiguous() if grad_v_final is not None else torch.empty(0, device=x0.device)
        
        with torch.no_grad():
             res = gfn_cuda.recurrent_manifold_fused(
                x0.contiguous(), v0.contiguous(), f_seq.contiguous(), 
                U_stack.contiguous(), W_stack.contiguous(), dt, dt_scales.contiguous(), forget_rates.contiguous(), H,
                pl, st, ss, mix_x_in.contiguous(), mix_v_in.contiguous())
             x_final, v_final, _, _ = res
        
        grads = gfn_cuda.recurrent_manifold_backward(
            gx_seq, gx_final, gv_final,
            x_final, v_final, 
            f_seq.contiguous(), U_stack.contiguous(), W_stack.contiguous(),
            dt, dt_scales.contiguous(), forget_rates.contiguous(), H,
            pl, st, ss, mix_x_in.contiguous(), mix_v_in.contiguous()
        )
        
        grad_x_init, grad_v_init, grad_f, grad_U, grad_W, grad_mix_x, grad_mix_v, grad_forget_rates = grads
        
        return grad_x_init, grad_v_init, grad_f, grad_U, grad_W, None, None, grad_forget_rates, None, None, None, None, (grad_mix_x if mix_x is not None else None), (grad_mix_v if mix_v is not None else None)

def recurrent_manifold_fused_autograd(x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, mix_x=None, mix_v=None):
    if not CUDA_AVAILABLE or not x.is_cuda:
        return None
    return RecurrentManifoldFusedFn.apply(x.contiguous(), v.contiguous(), f.contiguous(), U.contiguous(), W.contiguous(), dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x, mix_v)
