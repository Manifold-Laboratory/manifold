"""
Professional Manifold Internals (X-Ray) Visualization
=====================================================
Analyzing the step-by-step cognitive physics: Curvature peaks, 
Hamiltonian energy evolution, and Fractal tunneling activity.
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

def analyze_model_internals(checkpoint_path=None, input_text="999 + 1 = 1000"):
    logger = ResultsLogger("internal_physics", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔬 Analyzing Professional Manifold Internals (X-Ray Mode)...")
    print(f"  Input: '{input_text}'")
    
    # 1. Setup
    vocab = "0123456789+-*= <" 
    token_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_token = {i: c for c, i in token_to_id.items()}
    
    input_ids = torch.tensor([token_to_id[c] for c in input_text if c in token_to_id]).unsqueeze(0).to(device)
    
    model = Manifold(vocab_size=16, dim=512, depth=8, heads=4).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded")
        except:
            print("⚠️ Using structure-only (random weights)")
            
    model.eval()
    
    # 2. Sequential Audit
    seq_len = input_ids.size(1)
    curvature_hist, energy_hist, tunneling_hist = [], [], []
    
    print("  [*] Performing X-Ray Scan...")
    with torch.no_grad():
        forces = model.embedding(input_ids)
        x = model.x0.expand(1, -1)
        v = model.v0.expand(1, -1)
        
        for t in range(seq_len):
            force = forces[:, t]
            step_curv, step_energy, step_fractal = 0.0, 0.0, 0.0
            
            # Layer Audit
            curr_x, curr_v = x, v
            for layer in model.layers:
                v_norm = layer.macro_manifold.norm_v(curr_v) if hasattr(layer, 'macro_manifold') else layer.norm_v(curr_v)
                v_head = v_norm.chunk(model.heads, dim=-1)[0]
                christoffel = layer.macro_manifold.christoffels[0] if hasattr(layer, 'macro_manifold') else layer.christoffels[0]
                
                gamma = christoffel(v_head)
                step_curv += torch.norm(gamma).item()
                step_energy += 0.5 * torch.norm(curr_v).item()**2
                
                if hasattr(layer, 'threshold'):
                    if torch.norm(gamma) > layer.threshold:
                        step_fractal += 1.0
                
                curr_x, curr_v, _, _ = layer(curr_x, curr_v, force)
            
            curvature_hist.append(step_curv / len(model.layers))
            energy_hist.append(step_energy / len(model.layers))
            tunneling_hist.append(step_fractal)
            x, v = curr_x, curr_v

    # 3. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    tokens = [id_to_token[i.item()] for i in input_ids[0]]
    x_ticks = np.arange(len(tokens))
    
    axes[0].plot(curvature_hist, color='#D81B60', linewidth=3, marker='o')
    axes[0].set_title("Cognitive Curvature: Intellectual Effort Map", fontsize=15, fontweight='bold')
    axes[0].set_ylabel("Curvature (Γ)")
    axes[0].grid(alpha=0.2)
    
    axes[1].plot(energy_hist, color='#1E88E5', linewidth=3, marker='s')
    axes[1].set_title("Conservation of Logic: Hamiltonian Flow", fontsize=15, fontweight='bold')
    axes[1].set_ylabel("Energy (H)")
    axes[1].grid(alpha=0.2)
    
    axes[2].bar(x_ticks, tunneling_hist, color='#43A047', alpha=0.7)
    axes[2].set_title("Recursive Resolution: Fractal Tunneling Activity", fontsize=15, fontweight='bold')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(tokens, fontsize=12, fontweight='bold')
    
    fig.suptitle(f"X-Ray Analysis: Cognitive Physics of '{input_text}'", fontsize=22, fontweight='bold', y=0.98)
    logger.save_plot(fig, "xray_manifold_analysis.png")
    
    # 4. Metrics
    logger.save_json({
        "input_string": input_text,
        "token_audit": [{"token": t, "curvature": float(c)} for t, c in zip(tokens, curvature_hist)]
    })
    
    print(f"✓ X-Ray Scan Complete. Peak Thinking at token: '{tokens[np.argmax(curvature_hist)]}'")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    text = sys.argv[2] if len(sys.argv) > 2 else "999 + 1 = 1000"
    analyze_model_internals(ckpt, text)
