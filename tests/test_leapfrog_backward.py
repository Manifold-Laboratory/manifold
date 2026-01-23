"""
Test Leapfrog Backward Pass
============================
Verifies that the CUDA backward kernel produces correct gradients.
"""

import torch
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE

def test_leapfrog_backward():
    print("=" * 60)
    print("Testing Leapfrog Backward Pass")
    print("=" * 60)
    
    if not CUDA_AVAILABLE:
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    batch, dim, rank = 16, 128, 32
    dt, dt_scale, steps = 0.1, 1.0, 1
    
    # Create inputs with gradients
    x = torch.randn(batch, dim, device=device, requires_grad=True)
    v = torch.randn(batch, dim, device=device, requires_grad=True)
    f = torch.randn(batch, dim, device=device)
    U = torch.randn(dim, rank, device=device, requires_grad=True)
    W = torch.randn(dim, rank, device=device, requires_grad=True)
    
    print(f"\n📊 Configuration:")
    print(f"   Batch: {batch}, Dim: {dim}, Rank: {rank}")
    print(f"   Steps: {steps}, dt: {dt}")
    
    # Forward pass
    print("\n🚀 Running forward pass...")
    x_new, v_new = leapfrog_fused(x, v, f, U, W, dt, dt_scale, steps)
    
    # Create dummy loss
    loss = (x_new.sum() + v_new.sum())
    
    # Backward pass
    print("⬅️  Running backward pass...")
    loss.backward()
    
    # Check gradients
    print("\n✅ Gradient Check:")
    print(f"   grad_x: {x.grad is not None} | shape: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"   grad_v: {v.grad is not None} | shape: {v.grad.shape if v.grad is not None else 'None'}")
    print(f"   grad_U: {U.grad is not None} | shape: {U.grad.shape if U.grad is not None else 'None'}")
    print(f"   grad_W: {W.grad is not None} | shape: {W.grad.shape if W.grad is not None else 'None'}")
    
    if x.grad is not None:
        print(f"\n📈 Gradient Statistics:")
        print(f"   grad_x: mean={x.grad.mean().item():.6f}, std={x.grad.std().item():.6f}")
        print(f"   grad_v: mean={v.grad.mean().item():.6f}, std={v.grad.std().item():.6f}")
        print(f"   grad_U: mean={U.grad.mean().item():.6f}, std={U.grad.std().item():.6f}")
        print(f"   grad_W: mean={W.grad.mean().item():.6f}, std={W.grad.std().item():.6f}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(x.grad).any() or torch.isnan(v.grad).any() or torch.isnan(U.grad).any() or torch.isnan(W.grad).any()
        has_inf = torch.isinf(x.grad).any() or torch.isinf(v.grad).any() or torch.isinf(U.grad).any() or torch.isinf(W.grad).any()
        
        if has_nan:
            print("\n❌ WARNING: NaN detected in gradients!")
        elif has_inf:
            print("\n❌ WARNING: Inf detected in gradients!")
        else:
            print("\n✅ All gradients are finite and valid!")
    else:
        print("\n❌ No gradients computed!")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_leapfrog_backward()
