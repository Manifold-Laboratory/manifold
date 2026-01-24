import torch
import torch.nn as nn
from gfn.model import Manifold
import matplotlib.pyplot as plt

def test_energy_conservation():
    """Test if velocity v and energy persist when F=0 (Hamiltonian check)."""
    print("\n--- TEST: Hamiltonian Energy Conservation ---")
    dim = 64
    # Create model with standard scale
    model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1, rank=32)
    model.eval()
    
    # 1. Give an initial Kick (Impulse)
    # x=0, v=0 -> F_impulse -> x_1, v_1
    force_impulse = torch.zeros(1, 20, dim)
    force_impulse[0, 0, 0] = 5.0 # Strong kick in first dimension
    
    with torch.no_grad():
        logits, (state_x, state_v), _ = model(force_manual=force_impulse)
        
        # Now track what happens in the NEXT 50 steps with ZERO force
        # If it's a pure Hamiltonian, velocity should be constant energy
        curr_x, curr_v = state_x, state_v
        v_norms = [curr_v.norm().item()]
        
        # Manual steps to track decay
        layer = model.layers[0]
        for _ in range(50):
            # Zero force step
            curr_x, curr_v, _, _ = layer(curr_x, curr_v, force=torch.zeros(1, dim))
            v_norms.append(curr_v.norm().item())
            
    # Metrics
    start_v = v_norms[0]
    end_v = v_norms[-1]
    retention = end_v / (start_v + 1e-9)
    
    print(f"Initial Velocity Norm: {start_v:.4f}")
    print(f"Final Velocity Norm:   {end_v:.4f}")
    print(f"Energy Retention:      {retention:.2%}")
    
    if retention < 0.5:
        print("\033[91m[LEAK]\033[0m Physical energy is dissipating (Friction too high?)")
    elif retention > 0.95:
        print("\033[92m[CONSERVED]\033[0m Hamiltonian state is stable.")
    else:
        print("\033[93m[DAMPED]\033[0m Significant damping detected.")

def test_state_reversibility():
    """Test if we can recover parity from the final state x."""
    print("\n--- TEST: Recurrent State Linearity ---")
    # For parity, if bit=1 flips state, two bits=1 should return to original state
    # (or continue in a cyclic way).
    pass

if __name__ == "__main__":
    test_energy_conservation()
