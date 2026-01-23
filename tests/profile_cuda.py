"""
Detailed Profiling of CUDA Kernel Usage
========================================
"""
import torch
import time
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GFN
from gfn.model import Manifold
from gfn.cuda.ops import CUDA_AVAILABLE

print("="*80)
print("CUDA Kernel Usage Profiler")
print("="*80)
print(f"\nCUDA Available: {CUDA_AVAILABLE}")
print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Create model
device = torch.device('cuda')
# Test with heads=1 to enable recurrent fusion kernel
model = Manifold(vocab_size=2, dim=128, depth=6, heads=1, integrator_type='leapfrog').to(device)

# Check christoffel type
layer0 = model.layers[0]
print(f"\nLayer 0 Christoffel type: {type(layer0.christoffels[0]).__name__}")
print(f"Has U attribute: {hasattr(layer0.christoffels[0], 'U')}")
print(f"Has W attribute: {hasattr(layer0.christoffels[0], 'W')}")

if hasattr(layer0.christoffels[0], 'U'):
    U = layer0.christoffels[0].U
    print(f"U shape: {U.shape}, requires_grad: {U.requires_grad}")

# Test single forward pass
print("\n" + "="*80)
print("Testing TRAINING Mode")
print("="*80)

model.train()
x_input = torch.randint(0, 2, (4, 20), device=device)

# Time the forward pass
print("\n⏱️  Timing forward pass...")
torch.cuda.synchronize()
t0 = time.time()

logits, (x_final, v_final), christoffels = model(x_input)

torch.cuda.synchronize()
t1 = time.time()

print(f"Forward pass time: {(t1-t0)*1000:.2f} ms")
print(f"Christoffels collected: {len(christoffels)}")
print(f"Logits shape: {logits.shape}")

# Test backward
print("\n⏱️  Timing backward pass...")
torch.cuda.synchronize()
t0 = time.time()

loss = logits.sum()
loss.backward()

torch.cuda.synchronize()
t1 = time.time()

print(f"Backward pass time: {(t1-t0)*1000:.2f} ms")

# Check if gradients exist
print(f"\nGradients computed:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"  ✅ {name}: grad shape {param.grad.shape}")
        break  # Just show one example

print("\n" + "="*80)
print("Testing INFERENCE Mode")  
print("="*80)

model.eval()
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    
    logits, (x_final, v_final), christoffels = model(x_input)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"Inference forward pass time: {(t1-t0)*1000:.2f} ms")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nIf training is >100ms per forward pass, CUDA kernels are NOT being used.")
print("If inference is <10ms, CUDA kernels ARE working for inference.")
