"""
Professional Geodesic Flow Visualization
========================================
3D PCA-projected trajectory of the latent state through the manifold 
during sequence processing.
"""

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

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def plot_geodesic_flow(checkpoint_path=None, text="123 + 456 = 579"):
    logger = ResultsLogger("geodesic_flow", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🌀 Visualizing Professional Geodesic Flow for: '{text}'")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    input_ids = torch.tensor([token_to_id[c] for c in text if c in token_to_id]).unsqueeze(0).to(device)
    
    model = Manifold(vocab_size=len(vocab), dim=512, depth=8, heads=4).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded")
        except:
            print("⚠️ Using random weights")
            
    model.eval()
    
    # 2. Extraction
    trajectory = []
    with torch.no_grad():
        x = model.x0.expand(1, -1)
        v = model.v0.expand(1, -1)
        forces = model.embedding(input_ids)
        
        for t in range(input_ids.size(1)):
            trajectory.append(x.clone().cpu().numpy())
            # Use layer 0 update proxy for flow visualization
            x, v, _, _ = model.layers[0](x, v, forces[:, t])
            
    traj_data = np.concatenate(trajectory, axis=0)
    
    # 3. Dimensionality Reduction
    pca = PCA(n_components=3)
    traj_3d = pca.fit_transform(traj_data)
    
    # 4. Visualization
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Path with gradient
    for i in range(len(traj_3d) - 1):
        ax.plot(traj_3d[i:i+2, 0], traj_3d[i:i+2, 1], traj_3d[i:i+2, 2], 
                color=plt.cm.viridis(i/len(traj_3d)), linewidth=4, alpha=0.8)
        
    # Tokens as anchors
    colors = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))
    ax.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], c=colors, s=150, edgecolors='white', linewidths=1.5)
    
    for i, char in enumerate(text):
        if i < len(traj_3d):
            ax.text(traj_3d[i, 0], traj_3d[i, 1], traj_3d[i, 2], f" '{char}'", 
                    fontsize=12, fontweight='bold', color='black')

    ax.set_title(f"Manifold Geodesic Flow: Dimensionality Collapse Theory\nInput: '{text}'", 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("PC1", fontweight='bold')
    ax.set_ylabel("PC2", fontweight='bold')
    ax.set_zlabel("PC3", fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    logger.save_plot(fig, "geodesic_flow_3d.png")
    
    # 5. Metrics
    logger.save_json({
        "input": text,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_path_length": float(np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)))
    })
    
    print(f"✓ Geodesic Flow Analysis Complete. Path Length: {np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)):.4f}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    plot_geodesic_flow(ckpt)
