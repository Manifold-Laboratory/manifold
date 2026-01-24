
import torch
import torch.nn as nn
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
import matplotlib.pyplot as plt

def test_gradient_flow():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 64
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {'enabled': True, 'plasticity': 0.1},
    }
    
    model = Manifold(vocab_size=2, dim=dim, depth=2, heads=1, physics_config=physics_config).to(device)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    
    # Parity Task L=10
    L = 10
    B = 32
    
    history = {"loss": [], "grad_norm": [], "x_norm": []}
    
    print(f"--- GFN Convergence Audit (100 steps) ---")
    for step in range(100):
        optimizer.zero_grad()
        x_task = torch.randint(0, 2, (B, L), device=device)
        y_task = torch.cumsum(x_task, dim=1) % 2
        
        logits, (x_final, v_final), _ = model(x_task)
        
        # Binary target mapping
        target_bit = y_task.float() # [B, L]
        loss = nn.BCEWithLogitsLoss()(logits[:, :, 0], target_bit)
        
        loss.backward()
        
        # Calculate gradient norm
        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += p.grad.norm().item()
        
        history["loss"].append(loss.item())
        history["grad_norm"].append(total_grad)
        history["x_norm"].append(x_final.norm().item() / B)
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Grad: {total_grad:.6f} | x: {history['x_norm'][-1]:.4f}")
            
        optimizer.step()
        if hasattr(model.readout, 'update_step'):
            model.readout.update_step()

    if history["loss"][-1] > 0.68:
        print("\n[VDICT] Model is STUCK in the 0.69 plateau.")
        if history["grad_norm"][-1] < 1e-3:
            print("[CAUSE] Vanishing Gradients detected.")
        else:
            print("[CAUSE] Gradients exist but model is not escaping the mean.")
    else:
        print("\n[VDICT] Model is converging.")

if __name__ == "__main__":
    test_gradient_flow()
