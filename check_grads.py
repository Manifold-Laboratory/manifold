import torch
from gfn.geometry import ReactiveChristoffel

def check_grads():
    print("Checking Gradients for Singularity Logic...")
    dim = 64
    geo = ReactiveChristoffel(dim, rank=16)
    geo.active_cfg = {'singularities': {'enabled': True, 'threshold': 0.6, 'strength': 5.0}, 'enabled': True}
    
    # Fake inputs
    x = torch.randn(1, dim, requires_grad=True)
    v = torch.randn(1, dim, requires_grad=True)
    
    # Forward
    gamma = geo(v, x)
    loss = gamma.sum()
    loss.backward()
    
    print(f"Gamma Norm: {gamma.norm().item():.4f}")
    if geo.V.weight.grad is None:
         print("!!! V.weight.grad is None !!!")
    else:
         grad_norm = geo.V.weight.grad.norm().item()
         print(f"V.weight Gradient Norm: {grad_norm:.6f}")
         if grad_norm == 0.0:
             print("!!! V.weight Gradient is ZERO (Dead Logic) !!!")
         else:
             print("Gradients are flowing.")

check_grads()
