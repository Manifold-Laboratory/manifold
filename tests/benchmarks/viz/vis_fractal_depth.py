"""
Professional Fractal Manifold Visualization
===========================================
Visualizing the nested geometric structures (FractalMLayer) that enable 
infinite resolution and recursive logic in GFN.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def visualize_fractal_zoom(checkpoint_path=None):
    logger = ResultsLogger("fractals", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("💠 Visualizing Professional Fractal Tunneling (Recursive Zoom)...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'fractal': {'enabled': True, 'depth': 2, 'threshold': 0.0},
        'active_inference': {'enabled': True}
    }
    model = Manifold(vocab_size=len(vocab), dim=512, depth=1, heads=1, physics_config=physics_config).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded")
        except:
            print("⚠️ Using random weights")
            
    model.eval()
    layer = model.layers[0]
    macro, micro = layer.macro_manifold, layer.micro_manifold
    
    # 2. Render Grids
    grid_res = 50
    
    print("  [*] Rendering Fractal Scales...")
    with torch.no_grad():
        # Macro Scale
        lim_m = 2.5
        xm, ym = np.linspace(-lim_m, lim_m, grid_res), np.linspace(-lim_m, lim_m, grid_res)
        Xm, Ym = np.meshgrid(xm, ym)
        v_m = torch.zeros(grid_res*grid_res, 512).to(device)
        for i in range(grid_res):
            for j in range(grid_res):
                v_m[i*grid_res+j, 0], v_m[i*grid_res+j, 1] = Xm[i, j], Ym[i, j]
        mag_m = torch.norm(macro.christoffels[0](v_m), dim=-1).view(grid_res, grid_res).cpu().numpy()

        # Micro Scale (10x Zoom)
        lim_z = 0.25
        xz, yz = np.linspace(-lim_z, lim_z, grid_res), np.linspace(-lim_z, lim_z, grid_res)
        Xz, Yz = np.meshgrid(xz, yz)
        v_z = torch.zeros(grid_res*grid_res, 512).to(device)
        for i in range(grid_res):
            for j in range(grid_res):
                v_z[i*grid_res+j, 0], v_z[i*grid_res+j, 1] = Xz[i, j], Yz[i, j]
        mag_z = torch.norm(micro.christoffels[0](v_z), dim=-1).view(grid_res, grid_res).cpu().numpy()

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    im1 = axes[0].imshow(mag_m, extent=[-lim_m, lim_m, -lim_m, lim_m], cmap='viridis', origin='lower')
    axes[0].set_title("Macro-Geometry: Global Stability", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("v₀ (State Component)")
    axes[0].set_ylabel("v₁")
    fig.colorbar(im1, ax=axes[0], label='Metric Curvature')
    
    im2 = axes[1].imshow(mag_z, extent=[-lim_z, lim_z, -lim_z, lim_z], cmap='magma', origin='lower')
    axes[1].set_title("Micro-Geometry: 10x Focal Zoom", fontsize=15, fontweight='bold')
    axes[1].set_xlabel("v₀ (Zoomed)")
    fig.colorbar(im2, ax=axes[1], label='Local Resolution')

    fig.suptitle("Fractal Manifold Depth: Multi-Scale Geometric Reasoning", fontsize=22, fontweight='bold', y=0.98)
    logger.save_plot(fig, "fractal_depth_comparison.png")
    
    # 4. Metrics
    logger.save_json({
        "zoom_factor": 10.0,
        "grid_resolution": f"{grid_res}x{grid_res}",
        "macro_mean_force": float(np.mean(mag_m)),
        "micro_peak_resolution": float(np.max(mag_z)),
        "layer_type": "FractalMLayer (NestedSymplectic)"
    })
    
    print(f"✓ Fractal Depth Analysis Complete. Micro Complexity: {np.max(mag_z):.4f}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_fractal_zoom(ckpt)

if __name__ == "__main__":
    ckpt = "checkpoints/v0.3/epoch_0.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    visualize_fractal_zoom(ckpt)
