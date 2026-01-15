import torch
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import curiosity_loss

def test_curiosity_logic():
    print("Testing Entropy-Driven Curiosity Logic...")
    
    # 1. High Entropy Case: Diverse velocities
    v_high = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), # Diverse directions
        torch.tensor([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    ]
    loss_high = curiosity_loss(v_high, lambda_c=1.0)
    
    # 2. Low Entropy Case: Collapsed velocities
    v_low = [
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.001, 1.0], [1.0, 1.001]]), # Almost identical
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    ]
    loss_low = curiosity_loss(v_low, lambda_c=1.0)
    
    print(f"Loss High Entropy: {loss_high.item():.4f}")
    print(f"Loss Low Entropy:  {loss_low.item():.4f}")
    
    # Maximizing entropy means lower loss for high entropy
    # Since L = -S, if S_high > S_low, then -S_high < -S_low
    assert loss_high < loss_low, "High entropy should yield LOWER curiosity loss than low entropy"
    print(">> Entropy scaling: PASS")

def test_curiosity_gradients():
    print("\nTesting Curiosity Gradients...")
    
    # Initialize velocities as a leaf tensor
    params = torch.randn(4, 2, requires_grad=True)
    v = params * 0.1
    
    # Calculate curiosity loss
    loss = curiosity_loss([v], lambda_c=1.0)
    loss.backward()
    
    # Check if gradients exist in the leaf parameter
    assert params.grad is not None
    print(">> Gradient existence: PASS")

if __name__ == "__main__":
    try:
        test_curiosity_logic()
        test_curiosity_gradients()
        print("\nAll Curiosity tests: PASS")
    except Exception as e:
        print(f"\nCuriosity tests FAILED: {e}")
        sys.exit(1)
