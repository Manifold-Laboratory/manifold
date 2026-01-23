"""
Debug CUDA Usage During Training
=================================
This script will trace exactly which code paths are being used.
"""

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold

# Monkey-patch to trace calls
original_leapfrog_fused = None
original_christoffel_fused = None

def trace_leapfrog(*args, **kwargs):
    print(f"🔧 leapfrog_fused called with requires_grad: x={args[0].requires_grad if len(args) > 0 else 'N/A'}")
    return original_leapfrog_fused(*args, **kwargs)

def trace_christoffel(*args, **kwargs):
    print(f"🔧 christoffel_fused called with requires_grad: v={args[0].requires_grad if len(args) > 0 else 'N/A'}")
    return original_christoffel_fused(*args, **kwargs)

# Apply patches
from gfn.cuda import ops
original_leapfrog_fused = ops.leapfrog_fused
original_christoffel_fused = ops.christoffel_fused
ops.leapfrog_fused = trace_leapfrog
ops.christoffel_fused = trace_christoffel

# Test
device = torch.device('cuda')
model = Manifold(vocab_size=2, dim=128, depth=6, heads=4, integrator_type='leapfrog').to(device)
model.train()

print("\n" + "="*60)
print("Testing forward pass with gradients enabled")
print("="*60)

x = torch.randint(0, 2, (4, 20), device=device)
logits, (x_final, v_final), christoffels = model(x)

print(f"\n✅ Forward pass completed")
print(f"   Christoffels collected: {len(christoffels)}")
print(f"   Logits shape: {logits.shape}")

print("\n" + "="*60)
print("Testing backward pass")
print("="*60)

loss = logits.sum()
loss.backward()

print(f"\n✅ Backward pass completed")
