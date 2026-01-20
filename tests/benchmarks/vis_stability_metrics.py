import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam

def run_stability_test(steps=1000):
    print(f"🔬 Running Gradient Norm & Energy Stability Test ({steps} steps)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Model with explicit Hamiltonian requirements
    model = Manifold(
        vocab_size=1000,
        dim=256,
        depth=6,
        heads=4,
        integrator_type='leapfrog', # Symplectic
        physics_config={
           'embedding': {'type': 'functional'},
           'active_inference': {'enabled': False} 
        }
    ).to(device)
    
    # Optimizer: User requested RiemannianAdam (Correctly imported)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3, max_norm=10.0)
    criterion = nn.CrossEntropyLoss()
    
    # Import Hamiltonian Loss
    from gfn.losses import hamiltonian_loss
    
    # Metrics
    grad_norms = []
    energy_history = []
    loss_history = []
    
    model.train()
    
    # Fixed Dataset for Overfitting/Convergence Check
    # We want to show the loss GOES DOWN while Energy stays STABLE.
    # Generating random noise every step makes loss constant (unlearnable).
    fixed_inputs = torch.randint(0, 1000, (8, 32)).to(device)
    fixed_targets = torch.randint(0, 1000, (8, 32)).to(device)
    
    model.train()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Unroll sequence to capture velocity history for Hamiltonian Loss
        # Manifold forward typically returns only final state, so we check stability step-by-step
        velocities = []
        state = None
        outputs_list = []
        
        for t in range(fixed_inputs.size(1)):
            token = fixed_inputs[:, t:t+1]
            logit, state, _ = model(token, state=state)
            
            # state is (x, v)
            velocities.append(state[1])
            outputs_list.append(logit)
            
        logits = torch.cat(outputs_list, dim=1)
        
        # 1. Task Loss
        task_loss = criterion(logits.view(-1, 1000), fixed_targets.view(-1))
        
        # 2. Hamiltonian Loss (Energy Conservation)
        h_loss = hamiltonian_loss(velocities, lambda_h=0.01)
        
        total_loss = task_loss + h_loss
        loss_history.append(total_loss.item())
        
        total_loss.backward()
        
        # Global Gradient Norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Record Energy (Kinetic) of final state
        kinetic_energy = velocities[-1].pow(2).sum(dim=-1).mean().item() * 0.5
        energy_history.append(kinetic_energy)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={total_loss.item():.4f} (H={h_loss.item():.4f}) | GradNorm={total_norm:.4f} | Energy={kinetic_energy:.4f}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Gradient Norm
    axes[0].plot(grad_norms, color='#E76F51', linewidth=2)
    axes[0].set_title("Gradient Norm Stability", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Global L2 Norm")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Hamiltonian Energy
    axes[1].plot(energy_history, color='#2A9D8F', linewidth=2)
    axes[1].set_title("Hamiltonian Energy Conservation (Kinetic)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Energy (0.5 * v^2)")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Loss
    axes[2].plot(loss_history, color='#264653', linewidth=2)
    axes[2].set_title("Training Loss", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Cross Entropy")
    axes[2].set_xlabel("Training Steps")
    axes[2].grid(True, alpha=0.3)
    
    output_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "stability"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"stability_metrics_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n✅ Test Complete. Plot saved to: {plot_path}")
    print(f"Final Grad Norm: {grad_norms[-1]:.4f}")
    print(f"Energy Variance: {np.var(energy_history):.6f}")

if __name__ == "__main__":
    run_stability_test()
