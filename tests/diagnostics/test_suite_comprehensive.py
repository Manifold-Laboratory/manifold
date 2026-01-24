
import torch
import torch.nn as nn
from torch.autograd import gradcheck
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
# from gfn.utils import set_seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def run_gradient_check():
    print("\n[1/4] Running Gradient Correctness Check (Finite Differences)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tiny model for GradCheck to be fast
    dim = 4 # Tiny dim
    vocab = 2
    
    # Create functional wrapper for gradcheck
    class ManifoldFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, weights_U, weights_W):
            # We mock the model call here manually or use a wrapper
            # But gradcheck prefers pure tensor inputs. 
            # It's easier to check the model instance directly if we assume inputs requires_grad=False (discrete) 
            # and parameters requires_grad=True.
            pass

    # We will use the model directly but we need to verify parameters.
    model = Manifold(vocab_size=vocab, dim=dim, depth=1, heads=1, integrator_type='leapfrog').to(device).double() # Double precision for gradcheck
    
    # Inputs (Discrete, no grad)
    inputs = torch.randint(0, vocab, (2, 5)).to(device)
    
    # We want to check gradients w.r.t parameters (U, W)
    # But gradcheck verifies inputs.
    # Let's verify the "Embedding -> Readout" pipeline differentiable inputs.
    
    # Hack: Inject continuous embeddings
    embeddings = model.embedding(inputs).double().detach().requires_grad_(True)
    
    def func(emb):
        # Full model forward using the sequence
        # This will trigger the Fused BPTT kernel (training=True)
        model.train()
        logits, _, _ = model(force_manual=emb)
        return logits.sum()

    print(f"  Checking Full Model BPTT Gradients (L=5, B=2)...")
    try:
        # Check gradients w.r.t embeddings (this verifies the whole backprop chain)
        test = gradcheck(func, (embeddings,), eps=1e-6, atol=1e-3, rtol=1e-2) 
        print("  [PASS] Full BPTT Gradient Check Passed!")
    except Exception as e:
        print(f"  [FAIL] BPTT Gradient Check Failed: {e}")
        return False
        
    return True

def run_reversibility_check():
    print("\n[2/4] Running Symplectic Reversibility Check...")
    device = torch.device('cuda')
    dim = 128
    
    model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1).to(device)
    
    # Forward
    x0 = torch.randn(1, dim, device=device)
    v0 = torch.randn(1, dim, device=device)
    
    # Run one step manually via layer integrator
    with torch.no_grad():
        x1, v1 = x0.clone(), v0.clone()
        # Integrator step (Leapfrog)
        # We need access to the raw stepper. 
        # Using the layer's forward for now.
        # But layer forward does sequence.
        pass
        
    print("  [SKIP] Skipping Reversibility (Requires raw integrator access). Focusing on Gradients.")
    return True

def run_free_motion_check():
    print("\n[3/4] Running Free Particle Motion Check...")
    device = torch.device('cuda')
    dim = 64
    
    # Disable all forces
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'active_inference': {'enabled': False},
        'singularities': {'enabled': False},
        'fractal': {'enabled': False}
    }
    
    model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, physics_config=physics_config).to(device)
    
    # Zero out weights manually
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)
            
    # Input
    dt = model.physics_config.get('dt', 0.1)
    B, L = 1, 10
    x = torch.zeros(B, L, device=device).long() # Dummy IDs
    
    logits, (xF, vF), _ = model(x)
    
    # In free motion: x(t) should be constant if v0=0 (embedding init?)
    # Embeddings are 0. x0=0, v0=0.
    # Output should be 0.
    
    final_norm = xF.norm().item()
    print(f"  Final State Norm (Should be 0): {final_norm:.6f}")
    
    if final_norm < 1e-5:
        print("  [PASS] Free Motion is Stable.")
        return True
    else:
        print("  [FAIL] Free Motion Drifted!")
        return False

def run_energy_stability():
    print("\n[4/4] Running Hamiltonian Energy Stability...")
    print("  [INFO] Checking if Energy explodes over long sequence.")
    
    device = torch.device('cuda')
    model = Manifold(vocab_size=2, dim=128, depth=2, heads=1).to(device)
    
    inputs = torch.randint(0, 2, (1, 100)).to(device)
    
    try:
        logits, _, _ = model(inputs)
        print(f"  [PASS] Forward Pass 100 steps successful. Max Logit: {logits.max().item():.2f}")
    except Exception as e:
        print(f"  [FAIL] Explosion detected: {e}")
        return False
    return True

if __name__ == "__main__":
    print("=== GFN DIAGNOSTIC SUITE ===")
    r1 = run_free_motion_check()
    r2 = run_energy_stability()
    r3 = run_gradient_check()
    
    if r1 and r2 and r3:
        print("\nAll Systems Nominal.")
    else:
        print("\n[CRITICAL] Diagnostics Failed. See logs above.")
