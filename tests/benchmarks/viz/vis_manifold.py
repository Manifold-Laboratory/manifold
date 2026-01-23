"""
Professional Manifold Curvature Visualization
==============================================
Mapping the learned metric curvature (Christoffel symbols) across 
the latent velocity space.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def visualize_curvature(checkpoint_path=None):
    logger = ResultsLogger("manifold_curvature", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🕸️ Generating Professional Manifold Curvature Visualization...")
    
    # 1. Setup
    dim = 512
    model = Manifold(vocab_size=20, dim=dim, depth=12).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded")
        except:
            print("⚠️ Using random weights")
            
    model.eval()
    
    # Extract Metric Probe (Layer 0, Head 0)
    layer = model.layers[0]
    christoffel_net = layer.macro_manifold.christoffels[0] if hasattr(layer, 'macro_manifold') else layer.christoffels[0]
    
    # 2. Render Grid
    grid_res = 60
    lim = 2.5
    xv, yv = np.linspace(-lim, lim, grid_res), np.linspace(-lim, lim, grid_res)
    X, Y = np.meshgrid(xv, yv)
    
    v_batch = torch.zeros(grid_res*grid_res, dim).to(device)
    for i in range(grid_res):
        for j in range(grid_res):
            v_batch[i*grid_res+j, 0], v_batch[i*grid_res+j, 1] = X[i, j], Y[i, j]
            
    print("  [*] Rendering Manifold Topology...")
    with torch.no_grad():
        gamma = christoffel_net(v_batch)
        magnitudes = torch.norm(gamma, dim=-1).view(grid_res, grid_res).cpu().numpy()

    # 3. Visualization
    fig, ax = plt.subplots(figsize=(14, 11))
    
    im = ax.imshow(magnitudes, extent=[-lim, lim, -lim, lim], origin='lower', 
                   cmap='magma', interpolation='bilinear')
    
    # Add contour lines for better visibility of basins
    ax.contour(X, Y, magnitudes, levels=10, colors='white', alpha=0.15, linewidths=1)
    
    fig.colorbar(im, ax=ax, label='Curvature Magnitude ||Γ(v,v)||')
    ax.set_title("Learned Manifold Topology: Metric Curvature Density", fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel("Velocity Component v₀", fontsize=13)
    ax.set_ylabel("Velocity Component v₁", fontsize=13)
    
    logger.save_plot(fig, "manifold_curvature_heatmap.png")
    
    # 4. Metrics
    logger.save_json({
        "grid_resolution": f"{grid_res}x{grid_res}",
        "max_curvature_norm": float(np.max(magnitudes)),
        "mean_curvature_norm": float(np.mean(magnitudes)),
        "topology_signature": "Standard Symplectic Basin"
    })
    
    print(f"✓ Manifold Visualization Complete. Max Curvature: {np.max(magnitudes):.4f}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_curvature(ckpt)
