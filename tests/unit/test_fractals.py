import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layers import FractalMLayer

def test_fractal_tunneling_activation():
    print("Testing Fractal Tunneling Activation...")
    
    dim = 64
    heads = 4
    rank = 16
    
    physics_config = {
        'fractal': {
            'enabled': True,
            'threshold': 0.1, # Low threshold to trigger easily
            'alpha': 1.0     # Full strength for testing
        }
    }
    
    layer = FractalMLayer(dim, heads=heads, rank=rank, physics_config=physics_config)
    
    # Mock inputs
    x = torch.randn(1, dim)
    v = torch.randn(1, dim)
    
    # 1. Setup Low Curvature: Set U, W to near zero
    with torch.no_grad():
        for christ in layer.macro_manifold.christoffels:
            christ.U.zero_()
            christ.W.zero_()
            
    x_low, v_low, _, christ_low = layer(x, v)
    
    # Without curvature, x_low should be exactly what macro_manifold produces (which is x + v*dt because Gamma=0)
    # And since tunnel_gate should be near 0 (sigmoid(-0.1 * 5)), the micro contribution is negligible.
    print(f">> Low Curvature Output: {x_low.sum().item():.4f}")
    
    # 2. Setup High Curvature: Set U, W to high values
    with torch.no_grad():
        for christ in layer.macro_manifold.christoffels:
            christ.U.fill_(1.0)
            christ.W.fill_(1.0)
            
    x_high, v_high, _, christ_high = layer(x, v)
    
    # Calculate curvature estimate manually to verify
    stacked_gamma = torch.stack(christ_high, dim=1)
    r = torch.norm(stacked_gamma, dim=-1).mean()
    print(f">> High Curvature Detected: R={r.item():.4f}")
    
    # Assert that results are different (tunneling happened)
    # In low curvature, x_low is roughly x + v*dt
    # In high curvature, x_high includes micro_manifold correction
    assert not torch.allclose(x_low, x_high), "Fractal layer should produce different results under high curvature"
    print(">> Tunneling Activation: PASS")

if __name__ == "__main__":
    try:
        test_fractal_tunneling_activation()
        print("\nFractal Manifold tests: PASS")
    except Exception as e:
        print(f"\nFractal tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
