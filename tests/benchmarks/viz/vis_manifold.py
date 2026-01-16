
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

# Import centralized VRAM utility
from tests.benchmarks.bench_utils import measure_peak_memory
from src import GFN

def visualize_curvature(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA if available for accurate VRAM
    
    print("Generating Manifold Visualization...")
    
    # Init model
    model = GFN(vocab_size=20, dim=512, depth=12, rank=16).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
             # Handle 'model_state_dict' key
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
        except:
             print("Warning: Weight mismatch, using random weights.")
    
    # Extract First Layer's Christoffel Network
    layer = model.layers[0]
    
    # Handle FractalMLayer vs MLayer
    if hasattr(layer, 'macro_manifold'):
        christoffel_net = layer.macro_manifold.christoffels[0]
    else:
        christoffel_net = layer.christoffels[0]
    
    # Generate 2D grid in velocity space
    grid_size = 50
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    magnitudes = np.zeros((grid_size, grid_size))
    
    peak_mem = 0.0
    
    def compute_grid():
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    v_sample = torch.zeros(1, 512).to(device)
                    v_sample[0, 0] = X[i, j]
                    v_sample[0, 1] = Y[i, j]
                    
                    # Compute Gamma(v, v)
                    gamma = christoffel_net(v_sample)
                    
                    # Store L2 norm of the curvature vector
                    magnitudes[i, j] = torch.norm(gamma).item()
    
    # Measure VRAM
    peak_mem = measure_peak_memory(model, compute_grid)
    
    # We need to run it again to populate 'magnitudes' if measure_peak_memory cleared or didn't return values (it returns memory)
    # Actually, magnitudes is modified in place, so if measure_peak_memory runs the function, it should be populated.
    # checking bench_utils.py: "return max_mem" and it calls func().
    # Yes, it runs the function.
                
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitudes, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
    plt.colorbar(label='Curvature Magnitude ||Γ(v,v)||')
    plt.title(f"Learned Manifold Curvature (Layer 0)\n(Peak VRAM: {peak_mem:.1f} MB)", fontsize=14)
    plt.xlabel("Velocity Component v_0")
    plt.ylabel("Velocity Component v_1")
    
    # Update Output Directory
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "manifold_curvature"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = results_dir / "vis_manifold.png"
    plt.savefig(out_path)
    plt.close()
    
    # Save Metrics to JSON
    import json
    metrics_data = {
        "layer_idx": 0,
        "grid_size": grid_size,
        "mean_curvature": float(np.mean(magnitudes)),
        "max_curvature": float(np.max(magnitudes)),
        "peak_vram_mb": peak_mem
    }
    
    json_path = results_dir / "curvature_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Manifold Visualization Saved to: {out_path}")
    print(f"Metrics saved to: {json_path}")

if __name__ == "__main__":
    ckpt = "checkpoints/medium_fast/epoch_15.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    visualize_curvature(ckpt)
