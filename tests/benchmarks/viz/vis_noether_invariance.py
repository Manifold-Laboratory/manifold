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

from src.model import Manifold
from tests.benchmarks.bench_utils import measure_peak_memory

def verify_noether_symmetries(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("⚖️ Verifying Semantic Symmetries (Noether Invariance)...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    
    physics_config = {'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16}}
    model = Manifold(vocab_size=len(vocab), dim=512, depth=1, heads=1, physics_config=physics_config).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # 2. Generate Symmetric Pairs
    # Example: "2 + 3 = 5" and "3 + 2 = 5" (Commutative Symmetry)
    # Or identity transformations in latent space
    pairs = [
        ("2 + 3 = 5", "3 + 2 = 5"),
        ("1 + 1 = 2", "1 + 1 = 2"), # Identity
        ("4 * 2 = 8", "2 * 4 = 8"),
        ("9 - 5 = 4", "9 - 5 = 4")
    ]
    
    latent_reps = []
    labels = []
    
    latent_reps = []
    labels = []
    
    peak_mem = 0.0
    
    def run_inference_pairs():
         with torch.no_grad():
            for i, (s1, s2) in enumerate(pairs):
                for s, name in [(s1, f"Pair {i} A"), (s2, f"Pair {i} B")]:
                    ids = torch.tensor([token_to_id[c] for c in s]).unsqueeze(0).to(device)
                    # Get final state x after full sequence
                    x = model.x0.expand(1, -1)
                    v = model.v0.expand(1, -1)
                    forces = model.embedding(ids)
                    
                    for t in range(ids.size(1)):
                         _ = model.layers[0](x, v, forces[:, t])
    
    # Measure VRAM
    peak_mem = measure_peak_memory(model, run_inference_pairs)
    
    with torch.no_grad():
        for i, (s1, s2) in enumerate(pairs):
            for s, name in [(s1, f"Pair {i} A"), (s2, f"Pair {i} B")]:
                ids = torch.tensor([token_to_id[c] for c in s]).unsqueeze(0).to(device)
                # Get final state x after full sequence
                x = model.x0.expand(1, -1)
                v = model.v0.expand(1, -1)
                forces = model.embedding(ids)
                
                for t in range(ids.size(1)):
                    output = model.layers[0](x, v, forces[:, t])
                    x, v = output[0], output[1]
                
                latent_reps.append(x.cpu().numpy())
                labels.append(name)

    data = np.concatenate(latent_reps, axis=0)
    
    # 3. TSNE Visualization
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    reps_2d = tsne.fit_transform(data)
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    
    # Group pairs by color
    colors = ['#E76F51', '#264653', '#2A9D8F', '#E9C46A']
    
    for i in range(len(pairs)):
        idx_a = i * 2
        idx_b = i * 2 + 1
        
        plt.scatter(reps_2d[idx_a, 0], reps_2d[idx_a, 1], c=colors[i], s=200, label=f"Symmetry {i}", edgecolors='black')
        plt.scatter(reps_2d[idx_b, 0], reps_2d[idx_b, 1], c=colors[i], s=200, marker='X', edgecolors='black')
        
        # Draw line between them (The 'Invariance Gap')
        plt.plot([reps_2d[idx_a, 0], reps_2d[idx_b, 0]], 
                 [reps_2d[idx_a, 1], reps_2d[idx_b, 1]], 
                 c=colors[i], linestyle='--', alpha=0.5)

    plt.title(f"Isomeric Manifolds: Noether Invariance Map\n(Close proximity = High Semantic Symmetry) (Peak VRAM: {peak_mem:.1f} MB)", fontsize=14, fontweight='bold')
    plt.legend()
    
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "symmetries"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "noether_invariance.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Metrics to JSON
    import json
    # Calculate Distances
    distances = []
    for i in range(len(pairs)):
        idx_a = i * 2
        idx_b = i * 2 + 1
        dist = np.linalg.norm(reps_2d[idx_a] - reps_2d[idx_b])
        distances.append({
            "pair": f"{pairs[i][0]} <-> {pairs[i][1]}",
            "euclidean_distance": float(dist)
        })

    metrics_data = {
        "analysis_type": "Noether Invariance (Semantic Symmetry)",
        "invariance_gaps": distances,
        "peak_vram_mb": peak_mem
    }
    
    json_path = results_dir / "invariance_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"✅ Noether Invariance map saved to: {out_path}")
    print(f"Data saved to: {json_path}")

if __name__ == "__main__":
    ckpt = "checkpoints/v0.3/epoch_0.pt" # Placeholder
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    verify_noether_symmetries(ckpt)
