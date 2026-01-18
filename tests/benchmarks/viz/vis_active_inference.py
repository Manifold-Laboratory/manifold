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

def visualize_active_inference_distortion(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🧠 Visualizing Active Inference Manifold Distortion...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'active_inference': {'enabled': True}
    }
    model = Manifold(vocab_size=len(vocab), dim=512, depth=1, heads=1, physics_config=physics_config).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    layer = model.layers[0]
    manifold_macro = layer.macro_manifold
    
    # 2. Generate Grid
    grid_size = 40
    x_vals = np.linspace(-3, 3, grid_size)
    y_vals = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # scenario 1: Baseline (Low Plasticity)
    # scenario 2: Active (High Plasticity/Singularity)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_style("white")
    
    with torch.no_grad():
        for ax_idx, (title, plast, sing) in enumerate([
            ("Baseline Manifold (Passive)", 0.0, 1.0),
            ("Active Manifold (Curiosity Distorted)", 5.0, 10.0)
        ]):
            magnitudes = np.zeros((grid_size, grid_size))
            
            # Simulated context for singularity
            # x_context is the token state, V_w is the weight
            # We'll simulate a singularity at the center (0,0)
            x_sim = torch.zeros(1, 512).to(device)
            # Make V_w/v_w such that the singularity is triggered near origin
            v_w = torch.ones(512).to(device) # High potential everywhere for demo
            
                for j in range(grid_size):
                    v_sample = torch.zeros(1, 512).to(device)
                    v_sample[0, 0] = X[i, j]
                    v_sample[0, 1] = Y[i, j]
                    
                    # Ensure we pass the expected inputs to the Christoffel network
                    # Note: Physical modulation depends on layer implementation
                    gamma = manifold_macro.christoffels[0](v_sample, x=x_sim)
                    magnitudes[i, j] = torch.norm(gamma).item()
            
            im = axes[ax_idx].imshow(magnitudes, extent=[-3, 3, -3, 3], origin='lower', cmap='magma')
            axes[ax_idx].set_title(title, fontsize=14, fontweight='bold')
            axes[ax_idx].set_xlabel("v_0")
            axes[ax_idx].set_ylabel("v_1")
            plt.colorbar(im, ax=axes[ax_idx], label='Curvature ||Γ||')

    plt.suptitle("Active Inference: Adaptive Manifold Distortion\n(How Curiosity & Singularities shape the Thought Space)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "active_inference_distortion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Active Inference distortion plot saved to: {results_dir}/active_inference_distortion.png")

if __name__ == "__main__":
    ckpt = "checkpoints/v0.3/epoch_0.pt" # Placeholder
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    visualize_active_inference_distortion(ckpt)
