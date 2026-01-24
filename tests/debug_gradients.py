
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import gfn
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[*] GFN Package Path: {gfn.__file__}")
from gfn.model import Manifold

def debug_grads():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Analysis Device: {device}")
    
    # Tiny Model
    dim = 64
    rank = 16
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'singularities': {'enabled': True, 'strength': 5.0}
    }
    
    model = Manifold(
        vocab_size=2, dim=dim, depth=1, heads=1, 
        integrator_type='leapfrog',
        physics_config=physics_config
    ).to(device)
    
    print(f"[*] Model x0 device: {model.x0.device}")
    
    # Reload autograd to ensure prints are active
    import gfn.cuda.autograd
    import importlib
    importlib.reload(gfn.cuda.autograd)
    print(f"[*] Autograd Module: {gfn.cuda.autograd.__file__}")
    
    print("[*] Model initialized. Checking param requires_grad...")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(f"WARNING: {name} does not require grad!")
            
    # Dummy Input
    inputs = torch.randint(0, 2, (16, 20)).to(device)
    
    # Forward
    model.zero_grad()
    logits, _, _ = model(inputs, collect_christ=False)
    
    # Dummy Loss
    loss = logits.mean()
    print(f"[*] Forward Loss: {loss.item()}")
    
    # Inspect grad_fn
    print(f"[*] Logits grad_fn: {logits.grad_fn}")
    current_fn = logits.grad_fn
    history = []
    while current_fn is not None and len(history) < 10:
        history.append(str(current_fn))
        if hasattr(current_fn, 'next_functions'):
            # Just take the first meaningful next function
            found = False
            for next_fn, _ in current_fn.next_functions:
                if next_fn is not None:
                    current_fn = next_fn
                    found = True
                    break
            if not found: break
        else:
            break
            
    print("[*] Graph History Head: ", history)
    
    # Backward
    loss.backward()
    
    # Inspect Gradients
    print("\n--- Gradient Inspection ---")
    has_grads = False
    for name, p in model.named_parameters():
        if 'layers' in name and ('U' in name or 'W' in name):
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                grad_max = p.grad.abs().max().item()
                grad_mean = p.grad.abs().mean().item()
                print(f"{name}: Norm={grad_norm:.9f}, Max={grad_max:.9f}, Mean={grad_mean:.9f}")
                if grad_norm > 0: has_grads = True
            else:
                print(f"{name}: GRAD IS NONE!")
                
    if not has_grads:
        print("\n[CRITICAL] No gradients found on U/W matrices! Kernel is calculating zeros.")
    else:
        print("\n[INFO] Gradients present. Checking magnitude...")

if __name__ == "__main__":
    debug_grads()
