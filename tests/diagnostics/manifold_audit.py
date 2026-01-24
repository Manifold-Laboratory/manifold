import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gfn.model import Manifold
import time

def print_result(name, passed, details=""):
    color = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
    print(f"{color} {name:<40} {details}")

def audit_gradient_energy():
    """Test 1: Do we have non-trivial gradients reaching the weights?"""
    print("\n--- TEST 1: GRADIENT ENERGY ---")
    model = Manifold(vocab_size=10, dim=128, depth=1, heads=1,rank=32,
                     physics_config={'readout': {'type': 'implicit'}})
    model.train()
    
    # Simple binary task
    coord_dim = 16
    x = torch.randint(1, 10, (1, 20))
    y = torch.randint(0, 2, (1, 20, coord_dim)).float()
    
    logits, _ , _ = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    
    max_grad_w = 0
    max_grad_u = 0
    for layer in model.layers:
        for head in layer.christoffels:
            max_grad_w = max(max_grad_w, head.W.grad.abs().max().item())
            max_grad_u = max(max_grad_u, head.U.grad.abs().max().item())
            
    print_result("Manifold W Gradient Energy", max_grad_w > 1e-6, f"Max: {max_grad_w:.2e}")
    print_result("Manifold U Gradient Energy", max_grad_u > 1e-6, f"Max: {max_grad_u:.2e}")
    
    readout_grad = model.readout.mlp[0].weight.grad.abs().max().item()
    print_result("Readout MLP Gradient Energy", readout_grad > 1e-6, f"Max: {readout_grad:.2e}")

def audit_state_persistence():
    """Test 2: Does the state x actually accumulate over time without LayerNorm?"""
    print("\n--- TEST 2: STATE PERSISTENCE (HISTORY) ---")
    dim = 64
    model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1, rank=32)
    model.eval()
    
    # Input sequence of 1s (Force impulses)
    seq_len = 20
    x_input = torch.ones(1, seq_len).long()
    
    t0_x = torch.zeros(1, dim)
    t0_v = torch.zeros(1, dim)
    
    x_history = []
    curr_x, curr_v = t0_x, t0_v
    
    with torch.no_grad():
        all_forces = model.embedding(x_input)
        for t in range(seq_len):
            # manual step via MLayer
            curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, force=all_forces[:, t])
            x_history.append(curr_x.clone())
            
    x_seq = torch.stack(x_history, dim=1) # [1, L, D]
    
    # Metric: Distance from origin. If history accumulates, dist increases.
    dist = torch.norm(x_seq[0, -1]).item()
    std_val = x_seq.std().item()
    
    print_result("Trajectory Integration", dist > 0.5, f"Final Norm: {dist:.2f}")
    print_result("State Dynamic Range", std_val > 0.05, f"Std: {std_val:.2f}")

def audit_mini_parity():
    """Test 3: Can we solve 4-bit parity in 50 steps?"""
    print("\n--- TEST 3: MINI-PARITY CONVERGENCE ---")
    dim = 64
    model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32,
                     physics_config={'readout': {'type': 'implicit'}})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    # Create fixed data: 4-bit strings
    # Bits are 0, 1. (Force for 0 is 0, Force for 1 is boosted impulse)
    inputs = torch.randint(0, 2, (32, 4))
    targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() # [32, 1]
    
    start_time = time.time()
    best_loss = 100
    for step in range(101):
        model.train()
        optimizer.zero_grad()
        
        # We only care about the final prediction for the parity of the sequence
        logits, (final_x, _), _ = model(inputs)
        # Select last logit [batch, 1]
        last_logits = logits[:, -1, 0:1] 
        
        loss = F.binary_cross_entropy_with_logits(last_logits, targets)
        loss.backward()
        optimizer.step()
        model.readout.update_step()
        
        acc = ((last_logits > 0) == targets).float().mean().item()
        best_loss = min(best_loss, loss.item())
        
        if acc == 1.0 and loss < 0.1:
            print_result("Parity Solver (L=4)", True, f"Converged at step {step} (Loss: {loss.item():.4f})")
            return
            
    print_result("Parity Solver (L=4)", False, f"Best Loss: {best_loss:.4f}, Accuracy: {acc:.2f}")

def audit_physical_limits():
    """Test 4: Check for NaNs and Singularities"""
    print("\n--- TEST 4: PHYSICAL INTEGRITY ---")
    model = Manifold(vocab_size=10, dim=128, depth=2, heads=4, rank=3) # Low rank to stress test
    x = torch.randint(0, 10, (128, 50))
    
    logits, (fx, fv), _ = model(x)
    
    has_nan = torch.isnan(logits).any().item() or torch.isnan(fx).any().item()
    max_val = fx.abs().max().item()
    
    print_result("NaN Stability", not has_nan, "No NaNs found" if not has_nan else "!!! NaNs DETECTED !!!")
    print_result("Clamping Effectiveness", max_val <= 101.0, f"Max State: {max_val:.2f}")

if __name__ == "__main__":
    print("====================================================")
    print("   GFN MANIFOLD MASTER AUDIT (v3.8 Diagnostic)     ")
    print("====================================================")
    
    try:
        audit_gradient_energy()
        audit_state_persistence()
        audit_physical_limits()
        audit_mini_parity()
    except Exception as e:
        print(f"\033[91m[CRITICAL ERROR]\033[0m Audit crashed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n====================================================")
    print("                AUDIT COMPLETE                      ")
    print("====================================================")
