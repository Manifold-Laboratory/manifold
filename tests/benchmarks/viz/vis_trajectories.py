"""
Professional Trajectory Visualization
====================================
Standardized scientific visualization of geodesic flow in GFN vs Transformer.
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

from gfn.model import Manifold
try:
    from tests.benchmarks.baselines import MicroGPT
except ImportError:
    from baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

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
            trajectory_x.append(x.clone().cpu())
            trajectory_v.append(v.clone().cpu())
            
            for layer in model.layers:
                output = layer(x, v, force)
                x, v = output[0], output[1]
    
    traj_x = torch.cat(trajectory_x, dim=0).numpy()
    traj_v = torch.cat(trajectory_v, dim=0).numpy()
    return traj_x, traj_v

def visualize_transformer_attention(model, input_seq, device):
    """Extract transformer attention patterns."""
    model.eval()
    b, t = input_seq.size()
    x = model.token_emb(input_seq) + model.pos_emb[:, :t, :]
    x = model.drop(x)
    
    with torch.no_grad():
        mask = torch.triu(torch.ones(t, t, device=device) * float('-inf'), diagonal=1)
        hidden = model.blocks(x, mask=mask, is_causal=True)
    
    return hidden.squeeze(0).cpu().numpy()

def create_trajectory_comparison(checkpoint_path=None):
    """Side-by-side comparison of GFN vs Transformer information flow."""
    logger = ResultsLogger("trajectories", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 Running Professional Trajectory Visualization...")
    
    # Configuration
    vocab_size, dim, seq_len = 20, 512, 50
    input_seq = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    
    # Models
    gfn_model = Manifold(
        vocab_size=vocab_size, dim=dim, depth=12,
        physics_config={'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16}}
    ).to(device)
    gpt_model = MicroGPT(vocab_size=vocab_size, dim=dim, depth=12, heads=4).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            gfn_model.load_state_dict(ckpt['model_state_dict'])
            print("✓ GFN checkpoint loaded")
        except:
            print("⚠️ Using random weights")

    # Measurements
    gfn_mem = PerformanceStats.measure_peak_memory(gfn_model, lambda: visualize_gfn_trajectory(gfn_model, input_seq, device))
    gfn_traj_x, gfn_traj_v = visualize_gfn_trajectory(gfn_model, input_seq, device)
    
    tf_mem = PerformanceStats.measure_peak_memory(gpt_model, lambda: visualize_transformer_attention(gpt_model, input_seq, device))
    gpt_hidden = visualize_transformer_attention(gpt_model, input_seq, device)
    
    # Dimensionality Reduction
    gfn_3d = PCA(n_components=3).fit_transform(gfn_traj_x)
    gpt_3d = PCA(n_components=3).fit_transform(gpt_hidden)
    
    # Plotting
    fig = plt.figure(figsize=(20, 9))
    
    # Subplot 1: GFN (Geodesic Flow)
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(gfn_3d)))
    
    # Draw cinematic gradient trace
    for i in range(len(gfn_3d) - 1):
        ax1.plot(gfn_3d[i:i+2, 0], gfn_3d[i:i+2, 1], gfn_3d[i:i+2, 2],
                color=colors[i], linewidth=3, alpha=0.9)
    
    ax1.scatter(*gfn_3d[0], s=350, c='#2ecc71', marker='o', edgecolors='white', linewidths=2, label='Origin', zorder=5)
    ax1.scatter(*gfn_3d[-1], s=350, c='#e74c3c', marker='X', edgecolors='white', linewidths=2, label='Target', zorder=5)
    
    # Velocity vectors
    pca_gfn = PCA(n_components=3).fit(gfn_traj_x)
    arrow_indices = np.linspace(0, len(gfn_traj_v)-1, 6, dtype=int)
    for idx in arrow_indices[:-1]:
        v_3d = pca_gfn.transform(gfn_traj_v[idx:idx+1])[0]
        start = gfn_3d[idx]
        end = start + v_3d * 0.4
        arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                       mutation_scale=15, lw=2, arrowstyle='-|>', color='#f39c12', alpha=0.7)
        ax1.add_artist(arrow)
    
    ax1.set_title('Manifold GFN: Continuous Geodesic Flow', fontsize=16, fontweight='bold', pad=20)
    ax1.view_init(elev=25, azim=40)
    ax1.legend()

    # Subplot 2: Transformer (Discrete Jumps)
    ax2 = fig.add_subplot(122, projection='3d')
    colors_tf = plt.cm.plasma(np.linspace(0, 1, len(gpt_3d)))
    
    ax2.scatter(gpt_3d[:, 0], gpt_3d[:, 1], gpt_3d[:, 2], c=colors_tf, s=70, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax2.plot(gpt_3d[:, 0], gpt_3d[:, 1], gpt_3d[:, 2], color='gray', linewidth=1, alpha=0.4, linestyle='--')
    
    ax2.scatter(*gpt_3d[0], s=350, c='#2ecc71', marker='o', edgecolors='white', linewidths=2, label='Start', zorder=5)
    ax2.scatter(*gpt_3d[-1], s=350, c='#e74c3c', marker='X', edgecolors='white', linewidths=2, label='End', zorder=5)
    
    ax2.set_title('Transformer: Discrete Attention Steps', fontsize=16, fontweight='bold', pad=20)
    ax2.view_init(elev=25, azim=40)
    ax2.legend()
    
    logger.save_plot(fig, "trajectory_comparison.png")
    
    # Velocity Magnitude Plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    v_mags = np.linalg.norm(gfn_traj_v, axis=1)
    ax.plot(v_mags, linewidth=3, color='#2A9D8F', label='Momentum Magnitude')
    ax.fill_between(range(len(v_mags)), 0, v_mags, alpha=0.2, color='#2A9D8F')
    ax.set_title('Inertial Dynamics: GFN Information Forward Flow', fontsize=15, fontweight='bold')
    ax.set_xlabel('Integration Step (t)', fontsize=12)
    ax.set_ylabel('||v(t)||', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    logger.save_plot(fig2, "velocity_profile.png")
    
    # Save Metrics
    metrics = {
        "vram_comparison": {"gfn": float(gfn_mem), "transformer": float(tf_mem)},
        "trajectory_smoothness": float(np.mean(np.diff(gfn_3d, axis=0)**2)),
        "transformer_jitter": float(np.mean(np.diff(gpt_3d, axis=0)**2))
    }
    logger.save_json(metrics)

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    create_trajectory_comparison(ckpt)
