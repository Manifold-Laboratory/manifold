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

from src.model import Manifold
from tests.benchmarks.bench_utils import measure_peak_memory

def plot_christoffel_vector_field(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🏹 Plotting Christoffel Vector Field (Quiver Plot)...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    physics_config = {'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16}}
    model = Manifold(vocab_size=len(vocab), dim=512, depth=1, heads=1, physics_config=physics_config).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    layer = model.layers[0]
    
    # Handle both FractalMLayer and standard MLayer
    if hasattr(layer, 'macro_manifold'):
        # FractalMLayer
        christoffel = layer.macro_manifold.christoffels[0]
        layer_type = "FractalMLayer"
    else:
        # Standard MLayer
        christoffel = layer.christoffels[0]
        layer_type = "MLayer"
        
    print(f"   Detected Layer Type: {layer_type}")
    
    # 2. Generate Grid
    grid_size = 20 # Lower for quiver visibility
    x_vals = np.linspace(-2.5, 2.5, grid_size)
    y_vals = np.linspace(-2.5, 2.5, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    U_force = np.zeros((grid_size, grid_size))
    V_force = np.zeros((grid_size, grid_size))
    magnitudes = np.zeros((grid_size, grid_size))
    
    peak_mem = 0.0
    
    with torch.no_grad():
        # Measure VRAM for single point calculation
        # We define a dummy closure using the first point
        def dummy_forward():
            v = torch.zeros(1, 512).to(device)
            return christoffel(v)
            
        peak_mem = measure_peak_memory(model, dummy_forward)
        
        for i in range(grid_size):
            for j in range(grid_size):
                v_sample = torch.zeros(1, 512).to(device)
                v_sample[0, 0] = X[i, j]
                v_sample[0, 1] = Y[i, j]
                
                # Compute Gamma(v,v)
                gamma = christoffel(v_sample) # [1, 512]
                
                # Extract force vectors for dimensions 0 and 1
                U_force[i, j] = gamma[0, 0].item()
                V_force[i, j] = gamma[0, 1].item()
                magnitudes[i, j] = torch.norm(gamma).item()

    # 3. Plot
    plt.figure(figsize=(12, 10))
    sns.set_style("white")
    
    # Background Heatmap of force magnitude
    plt.imshow(magnitudes, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='YlGnBu', alpha=0.3)
    
    # Quiver Plot
    q = plt.quiver(X, Y, U_force, V_force, magnitudes, cmap='YlGnBu', scale=50, width=0.003)
    plt.colorbar(q, label='Force Magnitude ||Γ||')
    
    plt.title(f"Manifold v1.0: Christoffel Vector Field ({layer_type})\n(Peak VRAM: {peak_mem:.1f} MB)", fontsize=16, fontweight='bold')
    plt.xlabel("Velocity Component v_0")
    plt.ylabel("Velocity Component v_1")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Save Results
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "christoffel_vector_field"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "christoffel_vector_field.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Metrics to JSON
    import json
    metrics_data = {
        "layer_type": layer_type,
        "grid_size": grid_size,
        "mean_magnitude": float(magnitudes.mean()),
        "max_magnitude": float(magnitudes.max()),
        "peak_vram_mb": peak_mem
    }
    
    json_path = results_dir / "vector_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"✅ Vector Field plot saved to: {out_path}")
    print(f"✅ Metrics saved to: {json_path}")

if __name__ == "__main__":
    ckpt = "checkpoints/v0.3/epoch_0.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    plot_christoffel_vector_field(ckpt)
