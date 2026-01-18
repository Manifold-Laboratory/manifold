"""
Advanced Trajectory Visualization
==================================

Visualize how information flows through GFN vs Transformer.

- GFN: Smooth continuous trajectories in state space
- Transformer: Discrete attention pattern jumps
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
try:
    from tests.benchmarks.baselines import MicroGPT
except ImportError:
    from baselines import MicroGPT
from tests.benchmarks.bench_utils import measure_peak_memory


class Arrow3D(FancyArrowPatch):
    """Helper for 3D arrows."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


def visualize_gfn_trajectory(model, input_seq, device):
    """Capture full GFN trajectory through state space."""
    
    model.eval()
    trajectory_x = []
    trajectory_v = []
    
    x = model.x0.expand(1, -1)
    v = model.v0.expand(1, -1)
    all_forces = model.embedding(input_seq)
    
    with torch.no_grad():
        for t in range(input_seq.size(1)):
            force = all_forces[:, t]
            
            # Save state before evolution
            trajectory_x.append(x.clone().cpu())
            trajectory_v.append(v.clone().cpu())
            
            # Evolve
            for layer in model.layers:
                output = layer(x, v, force)
                x, v = output[0], output[1]
    
    # Convert to numpy
    traj_x = torch.cat(trajectory_x, dim=0).numpy()  # [seq_len, dim]
    traj_v = torch.cat(trajectory_v, dim=0).numpy()
    
    return traj_x, traj_v


def visualize_transformer_attention(model, input_seq, device):
    """Extract transformer attention patterns."""
    
    model.eval()
    
    # We'll hook into the attention mechanism
    # For simplicity, we'll compute attention manually
    b, t = input_seq.size()
    
    x = model.token_emb(input_seq) + model.pos_emb[:, :t, :]
    x = model.drop(x)
    
    # Get first transformer layer
    first_layer = model.blocks.layers[0]
    
    # Extract query, key from self-attention
    with torch.no_grad():
        # MultiheadAttention in PyTorch
        # We'll compute attention weights manually
        
        # Simplified: just get the final hidden states
        mask = torch.triu(torch.ones(t, t, device=device) * float('-inf'), diagonal=1)
        hidden = model.blocks(x, mask=mask, is_causal=True)
    
    return hidden.squeeze(0).cpu().numpy()  # [seq_len, dim]


def create_trajectory_comparison(checkpoint_path=None):
    """Side-by-side comparison of GFN vs Transformer information flow."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  TRAJECTORY VISUALIZATION")
    print("=" * 60)
    print(f"Device: {device}\n")
    
    # Load models
    vocab_size = 20
    dim = 512
    
    gfn_model = Manifold(
        vocab_size=vocab_size, dim=dim, depth=12,
        physics_config={'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16}}
    ).to(device)
    gpt_model = MicroGPT(vocab_size=vocab_size, dim=dim, depth=12, heads=4).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading GFN checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
            gfn_model.load_state_dict(ckpt['model_state_dict'])
            print("✓ Checkpoint loaded\n")
        except:
            print("⚠️  Using random weights\n")
    
    # Generate test sequence
    seq_len = 50
    input_seq = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    
    print(f"Extracting trajectories (sequence length: {seq_len})...")
    
    # Measure VRAM for GFN
    gfn_mem = measure_peak_memory(gfn_model, lambda: visualize_gfn_trajectory(gfn_model, input_seq, device))
    gfn_traj_x, gfn_traj_v = visualize_gfn_trajectory(gfn_model, input_seq, device)
    
    # Measure VRAM for Transformer
    tf_mem = measure_peak_memory(gpt_model, lambda: visualize_transformer_attention(gpt_model, input_seq, device))
    gpt_hidden = visualize_transformer_attention(gpt_model, input_seq, device)
    
    print(f"  Manifold Peak VRAM:    {gfn_mem:.1f} MB")
    print(f"  Transformer Peak VRAM: {tf_mem:.1f} MB")
    
    print("✓ Trajectories extracted")
    print("\nReducing to 3D via PCA...")
    
    # Reduce to 3D for visualization
    pca_gfn = PCA(n_components=3)
    gfn_3d = pca_gfn.fit_transform(gfn_traj_x)
    
    pca_gpt = PCA(n_components=3)
    gpt_3d = pca_gpt.fit_transform(gpt_hidden)
    
    # === VISUALIZATION ===
    fig = plt.figure(figsize=(18, 7))
    
    # GFN Trajectory (LEFT)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot smooth trajectory
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(gfn_3d)))
    
    for i in range(len(gfn_3d) - 1):
        ax1.plot(gfn_3d[i:i+2, 0], gfn_3d[i:i+2, 1], gfn_3d[i:i+2, 2],
                color=time_colors[i], linewidth=2.5, alpha=0.8)
    
    # Mark start and end
    ax1.scatter(*gfn_3d[0], s=300, c='green', marker='o', edgecolors='black', 
                linewidths=2, label='Start', zorder=5)
    ax1.scatter(*gfn_3d[-1], s=300, c='red', marker='X', edgecolors='black', 
                linewidths=2, label='End', zorder=5)
    
    # Add velocity arrows at key points
    arrow_indices = np.linspace(0, len(gfn_traj_v)-1, 5, dtype=int)
    for idx in arrow_indices[:-1]:  # Skip last
        # Velocity direction
        v_3d = pca_gfn.transform(gfn_traj_v[idx:idx+1])[0]
        start = gfn_3d[idx]
        # Scale velocity for visibility
        end = start + v_3d * 0.3
        
        arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                       mutation_scale=15, lw=2, arrowstyle='-|>', color='orange', alpha=0.6)
        ax1.add_artist(arrow)
    
    ax1.set_xlabel('PC1', fontsize=11)
    ax1.set_ylabel('PC2', fontsize=11)
    ax1.set_zlabel('PC3', fontsize=11)
    ax1.set_title('GFN: Smooth Geodesic Flow\n(Continuous trajectory in state space)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.view_init(elev=20, azim=45)
    
    # Transformer Trajectory (RIGHT)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot discrete jumps
    time_colors_gpt = plt.cm.plasma(np.linspace(0, 1, len(gpt_3d)))
    
    # Show as discrete points with connections
    ax2.scatter(gpt_3d[:, 0], gpt_3d[:, 1], gpt_3d[:, 2], 
               c=time_colors_gpt, s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Connect with thin lines to show sequence
    ax2.plot(gpt_3d[:, 0], gpt_3d[:, 1], gpt_3d[:, 2], 
            color='gray', linewidth=1, alpha=0.3, linestyle='--')
    
    # Mark start and end
    ax2.scatter(*gpt_3d[0], s=300, c='green', marker='o', edgecolors='black', 
                linewidths=2, label='Start', zorder=5)
    ax2.scatter(*gpt_3d[-1], s=300, c='red', marker='X', edgecolors='black', 
                linewidths=2, label='End', zorder=5)
    
    ax2.set_xlabel('PC1', fontsize=11)
    ax2.set_ylabel('PC2', fontsize=11)
    ax2.set_zlabel('PC3', fontsize=11)
    ax2.set_title('Transformer: Attention-Based Updates\n(Discrete token representations)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save
    # Save
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "trajectories"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "trajectory_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved to: {out_path}")
    print("=" * 60)
    
    # Additional plot: Velocity magnitude over time
    fig, ax = plt.subplots(figsize=(10, 5))
    
    v_magnitudes = np.linalg.norm(gfn_traj_v, axis=1)
    ax.plot(v_magnitudes, linewidth=2, color='#2A9D8F', label='Velocity ||v||')
    ax.fill_between(range(len(v_magnitudes)), 0, v_magnitudes, alpha=0.3, color='#2A9D8F')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Velocity Magnitude', fontsize=12)
    ax.set_title('GFN Velocity Evolution (Information Flow Speed)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.savefig(results_dir / "velocity_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Metrics to JSON
    import json
    metrics_data = {
        "sequence_length": seq_len,
        "manifold_vram_mb": float(gfn_mem),
        "transformer_vram_mb": float(tf_mem),
        "manifold_trajectory_variance": float(np.var(gfn_3d)),
        "transformer_trajectory_variance": float(np.var(gpt_3d))
    }
    
    json_path = results_dir / "trajectory_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"✓ Velocity plot saved to: {results_dir}/velocity_evolution.png")
    print(f"✓ Metrics saved to: {json_path}")


if __name__ == "__main__":
    ckpt_path = None
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    create_trajectory_comparison(ckpt_path)
