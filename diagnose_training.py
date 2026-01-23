import torch
import torch.nn as nn
import torch.optim as optim
from gfn.model import Manifold
from gfn.optim import RiemannianAdam

def train_diag():
    print("Diagnosing Training Loop...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Model (v2.6.2 Golden Spec)
    model = Manifold(
        vocab_size=2, dim=128, depth=6, heads=4,
        integrator_type='yoshida',
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'implicit', 'coord_dim': 16},
            'active_inference': {'enabled': True, 'plasticity': 0.1},
            'singularities': {'enabled': True, 'strength': 5.0}, 
            'stability': {'base_dt': 0.05}
        }
    ).to(device)
    
    # Dummy Parity Data
    x = torch.randint(0, 2, (32, 20), device=device)
    # Target: 0 or 1.
    targets = torch.randint(0, 2, (32, 20), device=device) 
    # (Just random targets to check if loss decreases - don't need real task logic for convergence check)
    
    # Target Coords Generation (Match benchmark)
    mask = 2**torch.arange(16).to(device)
    target_bits = (targets.unsqueeze(-1) & mask) > 0 
    target_coords = target_bits.float() * 2 - 1
    
    criterion = nn.MSELoss()
    
    print("\n--- Test 1: AdamW (Standard) ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    for i in range(20):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits[:, :, :1], target_coords[:, :, :1])
        loss.backward()
        optimizer.step()
        
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"Step {i}: Loss={loss.item():.4f}, Grad={grad_norm:.4f}")
        
    print("\n--- Test 2: RiemannianAdam (Custom) ---")
    model = Manifold(
        vocab_size=2, dim=128, depth=6, heads=4,
        integrator_type='yoshida',
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'implicit', 'coord_dim': 16},
            'active_inference': {'enabled': True, 'plasticity': 0.1},
            'singularities': {'enabled': True, 'strength': 5.0}, 
            'stability': {'base_dt': 0.05}
        }
    ).to(device)
    
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    
    for i in range(20):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits[:, :, :1], target_coords[:, :, :1])
        loss.backward()
        optimizer.step()
        
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"Step {i}: Loss={loss.item():.4f}, Grad={grad_norm:.4f}")

if __name__ == "__main__":
    train_diag()
