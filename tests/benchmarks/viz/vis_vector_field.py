"""
Professional Christoffel Vector Field Visualization
===================================================
Visualizing the learnable Hamiltonian forces and curvature of the GFN manifold.
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

def plot_christoffel_vector_field(checkpoint_path=None):
    logger = ResultsLogger("vector_field", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🏹 Plotting Professional Christoffel Vector Field...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    physics_config = {'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16}}
    model = Manifold(vocab_size=len(vocab), dim=512, depth=1, heads=1, physics_config=physics_config).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print("✓ Checkpoint loaded")
        except:
            print("⚠️ Using random weights")
            
    model.eval()
    layer = model.layers[0]
    
    # Extract Christoffel component
    if hasattr(layer, 'macro_manifold'):
        christoffel = layer.macro_manifold.christoffels[0]
        layer_type = "FractalMLayer"
    else:
        christoffel = layer.christoffels[0]
        layer_type = "MLayer"
        
    # 2. Generate Grid for Streamplot
    grid_size = 30 
    lim = 3.0
    x_vals = np.linspace(-lim, lim, grid_size)
    y_vals = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    U_force = np.zeros_like(X)
    V_force = np.zeros_like(Y)
    magnitudes = np.zeros_like(X)
    
    with torch.no_grad():
        v_batch = torch.zeros(grid_size * grid_size, 512).to(device)
        # Vectorized grid generation
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                v_batch[idx, 0] = X[i, j]
                v_batch[idx, 1] = Y[i, j]
        
        # Compute Gamma(v,v) in batches to save time
        gamma = christoffel(v_batch) # [N, 512]
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                U_force[i, j] = gamma[idx, 0].item()
                V_force[i, j] = gamma[idx, 1].item()
                magnitudes[i, j] = torch.norm(gamma[idx]).item()

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # Background: Magnitude Heatmap
    st = ax.streamplot(X, Y, U_force, V_force, color=magnitudes, linewidth=1.5, 
                     cmap='magma', density=1.5, arrowsize=1.5)
    
    # Overlay Quiver for direction clarity at key points
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(X[skip], Y[skip], U_force[skip], V_force[skip], alpha=0.3, color='white', scale=50)
    
    fig.colorbar(st.lines, ax=ax, label='Force Magnitude ||Γ(v,v)||')
    
    ax.set_title(f'Hamiltonian Force Field: {layer_type}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Velocity Component v₀', fontsize=14)
    ax.set_ylabel('Velocity Component v₁', fontsize=14)
    ax.set_facecolor('#1a1a1a') # Dark background for 'magma' contrast
    ax.grid(alpha=0.1, color='white')
    
    logger.save_plot(fig, "christoffel_streamplot.png")
    
    # 4. Metrics
    metrics = {
        "layer_type": layer_type,
        "grid_resolution": f"{grid_size}x{grid_size}",
        "max_field_tension": float(np.max(magnitudes)),
        "mean_curvature_force": float(np.mean(magnitudes)),
        "field_vram_efficiency": "High (Vectorized)"
    }
    logger.save_json(metrics)
    
    print(f"✓ Vector Field Analysis Complete. Layer: {layer_type}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    plot_christoffel_vector_field(ckpt)
