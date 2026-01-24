#!/usr/bin/env python
"""
Quick CUDA Diagnostic Test
===========================
Tests if CUDA kernels are actually being used.
"""

import torch
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("CUDA DIAGNOSTIC TEST")
print("="*60)

# Test 1: CUDA Available?
print("\n[1/5] Checking CUDA availability...")
print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")

# Test 2: Import CUDA ops
print("\n[2/5] Importing CUDA ops...")
try:
    from gfn.cuda.ops import CUDA_AVAILABLE, recurrent_manifold_fused
    print(f"  ✓ gfn.cuda.ops imported successfully")
    print(f"  CUDA_AVAILABLE = {CUDA_AVAILABLE}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Test recurrent_manifold_fused kernel
if CUDA_AVAILABLE and torch.cuda.is_available():
    print("\n[3/5] Testing recurrent_manifold_fused kernel...")
    try:
        batch, seq, dim, rank, heads = 2, 10, 64, 16, 2
        x = torch.randn(batch, dim).cuda()
        v = torch.randn(batch, dim).cuda()
        forces = torch.randn(batch, seq, dim).cuda()
        U = torch.randn(heads * 2, dim, rank).cuda()  # 2 layers * heads
        W = torch.randn(heads * 2, dim, rank).cuda()
        
        result = recurrent_manifold_fused(
            x, v, forces, U, W,
            dt=0.1, dt_scale=1.0, num_heads=heads,
            plasticity=0.0, sing_thresh=1.0, sing_strength=1.0
        )
        
        if result is not None:
            x_out, v_out, x_seq, reg_loss = result
            print(f"  ✓ Kernel SUCCESS")
            print(f"    Output shapes: x={x_out.shape}, v={v_out.shape}, x_seq={x_seq.shape}")
        else:
            print(f"  ✗ Kernel returned None")
    except Exception as e:
        print(f"  ✗ Kernel FAILED: {e}")
        import traceback
        traceback.print_exc()

# Test 4: Import Model
print("\n[4/5] Importing Manifold model...")
try:
    from gfn.model import Manifold
    print(f"  ✓ Manifold imported successfully")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test 5: Create small model and forward pass
print("\n[5/5] Testing model forward pass...")
try:
    from gfn.model import Manifold
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
    
    inputs = torch.randint(0, 10, (2, 5)).to(device)  # batch=2, seq=5
    
    model.eval()  # CRITICAL: Set to eval mode to use non-autograd path
    print(f"  Running forward pass (collect_christ=False to enable CUDA)...")
    with torch.no_grad():
        logits, (x_f, v_f), christ = model(inputs, collect_christ=False)
    
    print(f"  ✓ Forward pass SUCCESS")
    print(f"    Output logits shape: {logits.shape}")
    
except Exception as e:
    print(f"  ✗ Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nIf you see CUDA-related prints above (✓ CUDA AVAILABLE, etc.),")
print("then CUDA is working. If not, check the error messages.")
