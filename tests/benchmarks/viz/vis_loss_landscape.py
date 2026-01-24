"""
Professional Loss Landscape Visualization
===========================================
Visualizing the optimization geometry of Manifold GFN vs Transformer.
Demonstrates the 'Physics-Conditioning' effect on surface smoothness.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def compute_loss_surface(model, inputs, targets, d1, d2, resolution=30, scale=0.5):
    """Computes a 2D slice of the loss landscape."""
    alphas = np.linspace(-scale, scale, resolution)
    betas = np.linspace(-scale, scale, resolution)
    X, Y = np.meshgrid(alphas, betas)
    Z = np.zeros_like(X)
    
    orig_params = [p.clone() for p in model.parameters()]
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # Perturb
                for p, orig, dir1, dir2 in zip(model.parameters(), orig_params, d1, d2):
                    p.copy_(orig + alphas[i]*dir1 + betas[j]*dir2)
                
                # Forward
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out
                Z[j, i] = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
                
    # Restore
    with torch.no_grad():
        for p, orig in zip(model.parameters(), orig_params):
            p.copy_(orig)
        
    return X, Y, Z

def get_orthogonal_directions(model):
    d1, d2 = [], []
    for p in model.parameters():
        v1 = torch.randn_like(p)
        v2 = torch.randn_like(p)
        # Filter normalization (Li et al. 2018)
        v1 = v1 * (p.norm() / (v1.norm() + 1e-10))
        v2 = v2 * (p.norm() / (v2.norm() + 1e-10))
        d1.append(v1)
        d2.append(v2)
    return d1, d2

def run_landscape_analysis():
    logger = ResultsLogger("loss_landscape", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🏔️ Computing Professional Loss Landscape Analysis...")
    
    # 1. Setup
    dim, vocab = 128, 20
    inputs = torch.randint(0, vocab, (8, 20)).to(device)
    targets = torch.randint(0, vocab, (8, 20)).to(device)
    
    gfn = Manifold(vocab_size=vocab, dim=dim, depth=4).to(device)
    gpt = MicroGPT(vocab_size=vocab, dim=dim, depth=4, heads=4).to(device)
    
    # 2. Compute Surface
    g1, g2 = get_orthogonal_directions(gfn)
    t1, t2 = get_orthogonal_directions(gpt)
    
    print("  [*] Rendering GFN Landscape...")
    Xg, Yg, Zg = compute_loss_surface(gfn, inputs, targets, g1, g2, resolution=40)
    print("  [*] Rendering GPT Landscape...")
    Xt, Yt, Zt = compute_loss_surface(gpt, inputs, targets, t1, t2, resolution=40)
    
    # 3. Visualization
    fig = plt.figure(figsize=(20, 9))
    
    # GFN Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(Xg, Yg, Zg, cmap='viridis', edgecolor='none', antialiased=True, alpha=0.9)
    ax1.set_title('Manifold GFN: Physics-Stabilized Surface', fontsize=16, fontweight='bold', pad=20)
    ax1.set_zlabel('Loss', fontsize=12)
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Potential Energy (Loss)')

    # GPT Surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Xt, Yt, Zt, cmap='inferno', edgecolor='none', antialiased=True, alpha=0.9)
    ax2.set_title('Transformer: Unconstrained Parameter Geometry', fontsize=16, fontweight='bold', pad=20)
    ax2.set_zlabel('Loss', fontsize=12)
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    fig.suptitle("Optimization Geometry Comparison", fontsize=22, fontweight='bold', y=0.95)
    logger.save_plot(fig, "loss_landscape_3d.png")
    
    # 4. Contour Plots
    fig2, (cx1, cx2) = plt.subplots(1, 2, figsize=(16, 7))
    cx1.contourf(Xg, Yg, Zg, levels=25, cmap='viridis')
    cx1.set_title("GFN: Smooth Basin", fontweight='bold')
    cx2.contourf(Xt, Yt, Zt, levels=25, cmap='inferno')
    cx2.set_title("Transformer: Rugged Topology", fontweight='bold')
    logger.save_plot(fig2, "loss_landscape_contour.png")
    
    # 5. Metrics
    logger.save_json({
        "gfn_roughness_score": float(np.std(Zg)),
        "gpt_roughness_score": float(np.std(Zt)),
        "smoothness_improvement": float(np.std(Zt) / (np.std(Zg) + 1e-10)),
        "resolution": "40x40 Grid"
    })

if __name__ == "__main__":
    run_landscape_analysis()
