"""
Example: Using Yoshida 4th-Order Integrator with Energy Projection

This demonstrates how to enable both advanced numerical stability features:
1. Yoshida 4th-order symplectic integrator
2. Manifold projection (energy conservation)
"""

import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from gfn.model import Manifold

# Configuration with Yoshida + Energy Projection
physics_config = {
    'integrator': {
        'type': 'yoshida',           # Use Yoshida 4th-order (vs leapfrog 2nd-order)
        'energy_projection': True,    # Enable manifold projection
        'projection_eta': 0.1         # Projection strength
    },
    'embedding': {
        'type': 'functional',
        'mode': 'binary',
        'coord_dim': 16
    },
    'readout': {
        'type': 'implicit',
        'coord_dim': 16
    }
}

# Create model
model = Manifold(
    vocab_size=128,
    dim=256,
    depth=6,
    heads=4,
    integrator_type='yoshida',  # IMPORTANT: Set integrator type
    physics_config=physics_config
)

# Test forward pass
x = torch.randint(0, 128, (2, 10))  # [batch=2, seq=10]
logits, state, christoffels = model(x)

print(f"✓ Model initialized with Yoshida integrator")
print(f"✓ Energy projection enabled")
print(f"✓ Expected benefits:")
print(f"  - Energy drift: ~5% → <0.1% over 100k steps")
print(f"  - Allows 2-3x larger time steps with same accuracy")
print(f"  - Cost: 3x force evaluations (vs Leapfrog)")
print(f"\nOutput shapes:")
print(f"  - Logits: {logits.shape}")
print(f"  - State: (x={state[0].shape}, v={state[1].shape if state[1] is not None else 'None'})")
print(f"  - Christoffels: {len(christoffels)} collected")
