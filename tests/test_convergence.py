"""
Quick convergence test to isolate the issue
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam

# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda')

# Test 1: Single head (known to work)
print("="*60)
print("Test 1: Single Head (should converge)")
print("="*60)

model1 = Manifold(vocab_size=2, dim=128, depth=6, heads=1, integrator_type='leapfrog').to(device)
optimizer1 = RiemannianAdam(model1.parameters(), lr=1e-3)
model1.train()

for step in range(100):
    x = torch.randint(0, 2, (128, 20), device=device)
    y = torch.cumsum(x, dim=1) % 2
    
    optimizer1.zero_grad()
    logits, _, _ = model1(x)
    
    coord_dim = 16
    mask = 2**torch.arange(coord_dim).to(device)
    target_bits = (y.unsqueeze(-1) & mask) > 0
    target_coords = target_bits.float() * 2 - 1
    
    loss = nn.MSELoss()(logits[:, :, 0], target_coords[:, :, 0])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)  # Less aggressive
    optimizer1.step()
    
    pred_bits = (logits[:, :, 0] > 0.0).long()
    acc = (pred_bits == y).float().mean().item()
    
    if step % 20 == 0:
        print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%")

print("\n" + "="*60)
print("Test 2: Multi-Head (testing new kernel)")
print("="*60)

model2 = Manifold(vocab_size=2, dim=128, depth=6, heads=4, integrator_type='leapfrog').to(device)
optimizer2 = RiemannianAdam(model2.parameters(), lr=1e-3)
model2.train()

for step in range(100):
    x = torch.randint(0, 2, (128, 20), device=device)
    y = torch.cumsum(x, dim=1) % 2
    
    optimizer2.zero_grad()
    logits, _, _ = model2(x)
    
    coord_dim = 16
    mask = 2**torch.arange(coord_dim).to(device)
    target_bits = (y.unsqueeze(-1) & mask) > 0
    target_coords = target_bits.float() * 2 - 1
    
    loss = nn.MSELoss()(logits[:, :, 0], target_coords[:, :, 0])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
    optimizer2.step()
    
    pred_bits = (logits[:, :, 0] > 0.0).long()
    acc = (pred_bits == y).float().mean().item()
    
    if step % 20 == 0:
        print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("If Test 1 converges but Test 2 doesn't: Multi-head kernel has a bug")
print("If both don't converge: Hyperparameter or task issue")
print("If both converge: Reproducibility issue")
