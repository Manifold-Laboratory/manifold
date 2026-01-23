"""
Professional Noether Invariance Visualization
=============================================
Mapping semantic symmetries and conserved quantities in latent space.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.manifold import TSNE

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def verify_noether_symmetries(checkpoint_path=None):
    logger = ResultsLogger("symmetries", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("⚖️ Verifying Professional Semantic Symmetries (Noether Invariance)...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    
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

    # 2. Define Isomeric Pairs (Semantically identical but structurally different)
    pairs = [
        ("2 + 3 = 5", "3 + 2 = 5"),     # Commutativity
        ("4 * 2 = 8", "2 * 4 = 8"),     # Commutativity
        ("10 - 3 = 7", "10 - 3 = 7"),   # Identity
        ("5 + 5 = 10", "2 * 5 = 10"),   # Semantic Equivalence
        ("9 / 3 = 3", "3 * 1 = 3")      # Cross-operation Symmetry
    ]
    
    latent_reps = []
    
    with torch.no_grad():
        for s1, s2 in pairs:
            for s in [s1, s2]:
                ids = torch.tensor([token_to_id[c] for c in s]).unsqueeze(0).to(device)
                x = model.x0.expand(1, -1)
                v = model.v0.expand(1, -1)
                forces = model.embedding(ids)
                
                # Full integration through the sequence
                for t in range(ids.size(1)):
                    output = model.layers[0](x, v, forces[:, t])
                    x, v = output[0], output[1]
                
                latent_reps.append(x.cpu().numpy())

    data = np.concatenate(latent_reps, axis=0)
    
    # 3. Dimensionality Reduction
    # High perplexity for small datasets to maintain local structure
    tsne = TSNE(n_components=2, perplexity=len(pairs)-1, random_state=42, init='pca', learning_rate='auto')
    reps_2d = tsne.fit_transform(data)
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    # Use a sophisticated palette
    palette = sns.color_palette("husl", len(pairs))
    
    for i in range(len(pairs)):
        idx_a, idx_b = i * 2, i * 2 + 1
        color = palette[i]
        
        # Plot points
        ax.scatter(reps_2d[idx_a, 0], reps_2d[idx_a, 1], c=[color], s=350, 
                  label=f"Sym {i}: {pairs[i][0]}", edgecolors='white', linewidths=2, zorder=3)
        ax.scatter(reps_2d[idx_b, 0], reps_2d[idx_b, 1], c=[color], s=350, 
                  marker='X', edgecolors='white', linewidths=2, zorder=3)
        
        # Displacement Vector (The Invariance Gap)
        ax.plot([reps_2d[idx_a, 0], reps_2d[idx_b, 0]], 
                [reps_2d[idx_a, 1], reps_2d[idx_b, 1]], 
                c=color, linestyle='--', alpha=0.6, linewidth=2, zorder=2)
                
        # Annotate
        center = (reps_2d[idx_a] + reps_2d[idx_b]) / 2
        ax.annotate(f"Gap: {np.linalg.norm(reps_2d[idx_a]-reps_2d[idx_b]):.2f}", 
                   xy=center, xytext=(5, 5), textcoords="offset points", 
                   fontsize=9, color=color, fontweight='bold')

    ax.set_title("Manifold Noether Map: Visualization of Semantic Invariance", fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel("Isomeric Component 1 (t-SNE)", fontsize=13)
    ax.set_ylabel("Isomeric Component 2 (t-SNE)", fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Symmetry Groups")
    ax.grid(alpha=0.2, linestyle=':')
    
    logger.save_plot(fig, "noether_invariance_map.png")
    
    # 5. Metrics
    gaps = []
    for i in range(len(pairs)):
        dist = np.linalg.norm(data[i*2] - data[i*2+1])
        gaps.append({"pair": f"{pairs[i][0]} ~ {pairs[i][1]}", "v_space_distance": float(dist)})
        
    logger.save_json({
        "num_symmetries_tested": len(pairs),
        "mean_invariance_gap": float(np.mean([g['v_space_distance'] for g in gaps])),
        "detailed_gaps": gaps
    })
    
    print(f"✓ Noether Invariance Analysis Complete. Mean Gap: {np.mean([g['v_space_distance'] for g in gaps]):.4f}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    verify_noether_symmetries(ckpt)
