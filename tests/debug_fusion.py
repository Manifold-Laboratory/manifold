"""
Debug: Check if fusion kernel is being called
"""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold

# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda')

# Create model with heads=4
model = Manifold(vocab_size=2, dim=128, depth=6, heads=4, integrator_type='leapfrog').to(device)
model.train()

# Single forward pass
x = torch.randint(0, 2, (4, 20), device=device)

print("Testing forward pass with heads=4...")
print("If you see '[WARNING] Fusion failed', the kernel has a bug")
print("If you see nothing, check if fusion is even being attempted\n")

logits, (x_final, v_final), christoffels = model(x)

print(f"\nLogits shape: {logits.shape}")
print(f"Logits sample: {logits[0, 0, :]}")
print(f"Are all logits the same? {torch.allclose(logits[0, 0], logits[0, 1])}")

# Check if output makes any sense
if torch.isnan(logits).any():
    print("❌ NaN detected in output!")
elif torch.allclose(logits, torch.zeros_like(logits)):
    print("❌ All outputs are zero!")  
elif logits.std() < 0.01:
    print("❌ No variation in outputs - likely not learning")
else:
    print("✅ Outputs look reasonable")
