import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

def debug_parity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Running Parity Test on {device}")
    
    dim = 64
    depth = 2
    heads = 4
    vocab = 2
    
    # 1. Manifold with Binary Functional Embedding and Binary Readout
    model = Manifold(
        vocab_size=vocab, dim=dim, depth=depth, heads=heads,
        use_scan=False,
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'binary'},
            'active_inference': {'enabled': True, 'reactive_curvature': {'enabled': True, 'plasticity': 0.05}},
            'hyper_curvature': {'enabled': True}
        }
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Simple training loop
    for step in range(151):
        # Generate batch
        x = torch.randint(0, 2, (128, 20), device=device)
        y = torch.cumsum(x, dim=1) % 2
        
        # Map targets to bits (coord 0 for parity)
        coord_dim = 16
        mask = 2**torch.arange(coord_dim).to(device)
        target_bits = (y.unsqueeze(-1) & mask) > 0
        target_bits = target_bits.float()

        logits, _, _ = model(x)
        loss = criterion(logits, target_bits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            # Decode: first bit is enough for parity check (ID 0 or 1)
            preds = (logits[:, :, 0] > 0.0).long()
            acc = (preds == y).float().mean()
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Acc: {acc.item()*100:.1f}%")

if __name__ == "__main__":
    debug_parity()
