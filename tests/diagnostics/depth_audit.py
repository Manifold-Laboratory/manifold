import torch
import torch.nn.functional as F
import numpy as np
from gfn.model import Manifold
import time

def print_metric(name, value, unit="", threshold=None):
    status = ""
    if threshold is not None:
        passed = value >= threshold if threshold > 0 else value <= abs(threshold)
        status = "\033[92m[GOOD]\033[0m" if passed else "\033[91m[WEAK]\033[0m"
    print(f"{status} {name:<40}: {value:.4f} {unit}")

def test_sequence_curriculum():
    """Test parity convergence across different sequence lengths."""
    print("\n--- TEST 1: SEQUENCE LENGTH CURRICULUM ---")
    lengths = [5, 10, 15, 20]
    dim = 128
    
    for L in lengths:
        model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32,
                         physics_config={'readout': {'type': 'implicit'}})
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        
        # Training loop
        converged = False
        best_acc = 0
        for step in range(200):
            model.train()
            optimizer.zero_grad()
            
            inputs = torch.randint(0, 2, (64, L))
            targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float()
            
            logits, _, _ = model(inputs)
            last_logits = logits[:, -1, 0:1]
            
            loss = F.binary_cross_entropy_with_logits(last_logits, targets)
            loss.backward()
            optimizer.step()
            model.readout.update_step()
            
            acc = ((last_logits > 0) == targets).float().mean().item()
            best_acc = max(best_acc, acc)
            
            if acc > 0.98:
                converged = True
                print(f"L={L:<2} | CONVERGED at step {step}")
                break
        
        if not converged:
            print(f"L={L:<2} | \033[91mFAILED\033[0m | Best Acc: {best_acc:.2%}")

def audit_gradient_decay():
    """Measure gradient magnitude at different sequence positions."""
    print("\n--- TEST 2: GRADIENT PATH DECAY (L=20) ---")
    L = 20
    dim = 128
    model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32)
    model.train()
    
    inputs = torch.randint(0, 2, (1, L))
    
    # Forward pass
    logits, (state_x, state_v), _ = model(inputs)
    
    # Measure gradient for EACH timestep force
    grads = []
    for t in range(L):
        model.zero_grad()
        # High loss at time T
        target = torch.tensor([[[1.0]]]) 
        loss = F.mse_loss(logits[:, t:t+1, 0:1], target)
        loss.backward(retain_graph=True)
        
        # Sum of gradients on W at this timestep? 
        # Actually better to look at dLoss/dForce_t
        # But we want to know if the optimizer can update the manifold.
        grad_norm = model.layers[0].christoffels[0].W.grad.abs().mean().item()
        grads.append(grad_norm)
        
    for i, g in enumerate(grads):
        print(f"Step {i+1:02} Gradient Energy: {g:.2e}")
    
    decay = grads[0] / (grads[-1] + 1e-9)
    print_metric("Gradient Decay (Early/Late)", decay, threshold=10.0)

def audit_state_saturation():
    """Check if state x hits clamping limits during long sequences."""
    print("\n--- TEST 3: STATE SATURATION AUDIT (L=50) ---")
    L = 50
    dim = 64
    model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32)
    model.eval()
    
    # Input sequence of all 1s
    inputs = torch.ones(1, L).long()
    
    with torch.no_grad():
        logits, (fx, fv), _ = model(inputs)
        # We need to see the intermediate x states
        # Manually step
        curr_x = torch.zeros(1, dim)
        curr_v = torch.zeros(1, dim)
        forces = model.embedding(inputs)
        
        max_norms = []
        for t in range(L):
            curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, forces[:, t])
            max_norms.append(curr_x.norm().item())
            
    print(f"Max Norm at t=1:  {max_norms[0]:.2f}")
    print(f"Max Norm at t=25: {max_norms[24]:.2f}")
    print(f"Max Norm at t=50: {max_norms[-1]:.2f}")
    
    saturation_risk = max_norms[-1] > 80.0
    print_result = "\033[91m[SATURATED]\033[0m" if saturation_risk else "\033[92m[SAFE]\033[0m"
    print(f"{print_result} Clamping Ceiling (100.0) Health")

if __name__ == "__main__":
    print("====================================================")
    print("   GFN DEPTH & SCALING AUDIT (Phase 10)            ")
    print("====================================================")
    
    try:
        audit_state_saturation()
        audit_gradient_decay()
        test_sequence_curriculum()
    except Exception as e:
        print(f"Audit crashed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n====================================================")
