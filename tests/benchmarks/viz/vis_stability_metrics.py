"""
Professional Stability Metrics Visualization
============================================
Monitoring gradient norms, energy conservation, and task convergence 
to verify the symplectic stability of the GFN manifold.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import hamiltonian_loss
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def run_stability_test(steps=500):
    logger = ResultsLogger("stability", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔬 Running Professional Stability Test ({steps} steps)...")
    
    # 1. Setup Model & Task
    dim = 256
    model = Manifold(vocab_size=1000, dim=dim, depth=6, heads=4, integrator_type='leapfrog').to(device)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3, max_norm=10.0)
    criterion = nn.CrossEntropyLoss()
    
    # Fixed Batch for Stability Verification
    inputs = torch.randint(0, 1000, (8, 32)).to(device)
    targets = torch.randint(0, 1000, (8, 32)).to(device)
    
    history = {"grad_norm": [], "energy": [], "loss": []}
    
    print("  [*] Simulating Hamiltonian Dynamics...")
    for step in range(steps):
        optimizer.zero_grad()
        
        # Sequence Rollout
        state = None
        velocities, outputs = [], []
        for t in range(inputs.size(1)):
            logit, state, _ = model(inputs[:, t:t+1], state=state)
            velocities.append(state[1])
            outputs.append(logit)
            
        logits = torch.cat(outputs, dim=1)
        
        # Losses
        loss_task = criterion(logits.view(-1, 1000), targets.view(-1))
        loss_ham = hamiltonian_loss(velocities, lambda_h=0.01)
        total_loss = loss_task + loss_ham
        
        total_loss.backward()
        
        # Metrics
        grad_norm = sum(p.grad.detach().data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
        kinetic_energy = velocities[-1].pow(2).sum(dim=-1).mean().item() * 0.5
        
        history["loss"].append(total_loss.item())
        history["grad_norm"].append(grad_norm)
        history["energy"].append(kinetic_energy)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            print(f"    Step {step:>4}: Loss={total_loss.item():.4f} | Energy={kinetic_energy:.4f}")

    # 2. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    axes[0].plot(history["grad_norm"], color='#E76F51', linewidth=2)
    axes[0].set_title("Gradient Stability: Global L2 Norm", fontweight='bold')
    axes[0].set_ylabel("||∇θ||")
    axes[0].grid(alpha=0.2)
    
    axes[1].plot(history["energy"], color='#2A9D8F', linewidth=2)
    axes[1].set_title("Hamiltonian Conservation: Kinetic Energy", fontweight='bold')
    axes[1].set_ylabel("KE = ½mv²")
    axes[1].grid(alpha=0.2)
    
    axes[2].plot(history["loss"], color='#264653', linewidth=2)
    axes[2].set_title("Task Convergence (CE + Hamiltonian)", fontweight='bold')
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Optimization Steps")
    axes[2].grid(alpha=0.2)
    
    fig.suptitle("Symplectic Stability & Convergence Analysis", fontsize=20, fontweight='bold', y=0.96)
    logger.save_plot(fig, "stability_metrics_dashboard.png")
    
    # 3. Save Metrics
    logger.save_json({
        "final_grad_norm": grad_norm,
        "energy_drift_var": float(np.var(history["energy"])),
        "final_loss": history["loss"][-1],
        "config": {"integrator": "leapfrog", "steps": steps}
    })
    
    print(f"✓ Stability Analysis Complete. Energy Drift: {np.var(history['energy']):.8f}")

if __name__ == "__main__":
    run_stability_test()

if __name__ == "__main__":
    run_stability_test()
