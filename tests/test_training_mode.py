#!/usr/bin/env python
"""Test training mode CUDA dispatch"""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold

print("Testing TRAINING mode CUDA dispatch...")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Manifold(
    vocab_size=10,
    dim=64,
    depth=2,
    heads=2,
    rank=16,
    integrator_type='heun',
    use_scan=False
).to(device)

model.train()  # TRAINING MODE
print(f"Model in training mode: {model.training}")

inputs = torch.randint(0, 10, (2, 5)).to(device)

print("\nForward pass with gradients...")
logits, (x_f, v_f), christ = model(inputs, collect_christ=False)

print(f"\n✓ Forward pass completed")
print(f"Output logits shape: {logits.shape}")
print(f"Requires grad: {logits.requires_grad}")
