
import torch
from torch.autograd import Function

try:
    import gfn_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

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
