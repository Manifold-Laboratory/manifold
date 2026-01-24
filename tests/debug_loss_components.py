
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from gfn.model import Manifold
from gfn.losses import geodesic_regularization

def debug_loss():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running on {device}")
    
    # Config from superiority benchmark
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {'enabled': True, 'plasticity': 0.1},
        'singularities': {'enabled': True, 'strength': 5.0}
    }
    
    model = Manifold(
        vocab_size=2, dim=128, depth=6, heads=1, 
        integrator_type='leapfrog',
        physics_config=physics_config
    ).to(device)
    
    # Create Dummy Batch
    B, L = 32, 20
    inputs = torch.randint(0, 2, (B, L)).to(device)
    
    # Forward Pass
    print("\n--- Forward Pass ---")
    logits, state, christoffels = model(inputs, collect_christ=False)
    
    print(f"Logits Shape: {logits.shape}")
    print(f"Christoffels (Reg Loss Container): {christoffels}")
    
    if len(christoffels) > 0:
        reg_val = christoffels[0]
        print(f"Raw Reg Loss Tensor: {reg_val.mean().item():.6f}")
        
    # Check what geodesic_regularization does
    loss_phy = geodesic_regularization(None, christoffels, lambda_g=0.001)
    print(f"Computed loss_phy (lambda=0.001): {loss_phy.item():.6f}")
    
    # Check MSE magnitude
    # Target: Random bits
    targets_int = torch.randint(0, 2, (B, L)).to(device)
    coord_dim = 16
    mask = 2**torch.arange(coord_dim).to(device)
    target_bits = (targets_int.unsqueeze(-1) & mask) > 0
    target_coords = target_bits.float() * 2 - 1
    
    # Align shapes
    pred = logits[:, :, 0] # As per benchmark
    train_step_mse = nn.MSELoss()(logits[:, :, 0], target_coords[:, :, 0]) # What benchmark does
    print(f"Benchmark MSE Loss: {train_step_mse.item():.6f}")
    
    total = train_step_mse + loss_phy
    print(f"Total Loss: {total.item():.6f}")
    
    # --- OVERFIT TEST ---
    print("\n--- Running Overfit Test (50 steps) ---")
    model.train()
    from gfn.optim import RiemannianAdam
    optimizer = RiemannianAdam(model.parameters(), lr=1e-2)
    
    for i in range(50):
        optimizer.zero_grad()
        logits, _, _ = model(inputs, collect_christ=False)
        pred = logits[:, :, 0]
        # Target needs to be broadcast or computed per step? 
        # We use fixed inputs/targets for overfit
        loss = nn.MSELoss()(pred, target_coords[:, :, 0])
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")

if __name__ == "__main__":
    debug_loss()
