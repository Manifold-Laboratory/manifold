import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.decomposition import PCA

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
# Import centralized VRAM utility
from tests.benchmarks.bench_utils import measure_peak_memory

def plot_geodesic_flow(checkpoint_path, text="123 + 456 = 579"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌀 Visualizing Geodesic Flow for: '{text}'")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    input_ids = torch.tensor([token_to_id[c] for c in text]).unsqueeze(0).to(device)
    
    model = Manifold(vocab_size=len(vocab), dim=512, depth=8, heads=8).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
            # Handle 'model_state_dict' key
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
        except:
             print("Warning: Weight mismatch, using random weights.")

    model.eval()
    
    # 2. Extract Trajectory
    trajectory = []
    
    peak_mem = 0.0
    
    def run_flow():
        with torch.no_grad():
            x = model.x0.expand(1, -1)
            v = model.v0.expand(1, -1)
            forces = model.embedding(input_ids)
            
            for t in range(input_ids.size(1)):
                # Record state
                trajectory.append(x.clone().cpu().numpy())
                
                # Step
                output = model.layers[0](x, v, forces[:, t])
                if isinstance(output, tuple):
                     x, v = output[0], output[1]
                else:
                     x, v = output # Should not happen with new MLayer, but safe fallback
    
    # Measure VRAM
    peak_mem = measure_peak_memory(model, run_flow)
    
    # run_flow appends to trajectory list in-place
    
    traj_data = np.concatenate(trajectory, axis=0) # [seq_len, dim]
    
    # 3. PCA to 3D
    pca = PCA(n_components=3)
    traj_3d = pca.fit_transform(traj_data)
    
    # 4. Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sns.set_style("white")
    
    # Plot line
    ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='#2A9D8F', linewidth=4, alpha=0.8)
    
    # Plot points with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))
    ax.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], c=colors, s=100, edgecolors='black')
    
    # Annotate tokens
    for i, char in enumerate(text):
        if i < len(traj_3d):
             ax.text(traj_3d[i, 0], traj_3d[i, 1], traj_3d[i, 2], char, fontsize=12, fontweight='bold')
 
    ax.set_title(f"Manifold v1.0: Geodesic Reasoning Trajectory\n'{text}' (Peak VRAM: {peak_mem:.1f} MB)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    ax.set_zlabel("Latent PC3")
    
    # Save
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "geodesic_flow"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "geodesic_flow_3d.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Metrics to JSON
    import json
    metrics_data = {
        "text": text,
        "trajectory_length": len(traj_3d),
        "peak_vram_mb": peak_mem
    }
    
    json_path = results_dir / "geodesic_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"✅ Geodesic Flow visualization saved to: {out_path}")
    print(f"Metrics saved to: {json_path}")

if __name__ == "__main__":
    ckpt = "checkpoints/v1_0_math_oracle/epoch_1.pt"
    text = "123 + 456 = 579"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    plot_geodesic_flow(ckpt, text)
