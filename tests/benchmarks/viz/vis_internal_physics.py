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

def analyze_model_internals(checkpoint_path, input_text="999 + 1 = 1000"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔬 Analyzing Manifold Internals (X-Ray Mode)...")
    print(f"Input: '{input_text}'")
    
    # 1. Setup Model (Assuming Math Oracle vocabulary for now)
    # 1. Setup Model
    # Mapping: '0'-'9', '+', '-', '*', '=', ' ', '<PAD>'
    vocab = "0123456789+-*= <" 
    # Note: We need 16 tokens to match the checkpoint. 
    # The training dataset likely adds a PAD token.
    
    token_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_token = {i: c for c, i in token_to_id.items()}
    
    # Simple tokenizer
    # Filter chars not in vocab or map to ?
    input_ids = []
    for c in input_text:
        if c in token_to_id:
            input_ids.append(token_to_id[c])
        else:
            # Fallback or skip
            pass
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Use v1.0 Golden Config dimensions or load from ckpt if available
    # Matches 'gfn_medium.yaml'
    model = Manifold(
        vocab_size=16, 
        dim=512, 
        depth=12, 
        heads=8,
        rank=64,
        physics_config={
            'active_inference': {'enabled': True, 'reactive_curvature': {'enabled': True, 'plasticity': 0.1}, 'singularities': {'enabled': True, 'strength': 10.0, 'threshold': 0.8}},
            'fractal': {'enabled': True, 'threshold': 0.001},
            'symmetries': {'enabled': True}
        }
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Handle 'model_state_dict' key
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
        # Approximate loading (Physics params might be new/different)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("!! WARNING: No checkpoint found. Using random weights (Structural Visualization).")

    model.eval()
    
    # 2. Capture Metrics
    # We'll run step by step to capture internal state
    seq_len = input_ids.size(1)
    curvature_history = []
    hamiltonian_history = []
    fractal_activity = []
    
    peak_mem = 0.0
    
    # Measure VRAM for the analysis process
    # We wrap the main analysis logic in a closure for measurement, or measure a representative pass
    def full_analysis_pass():
        with torch.no_grad():
            forces = model.embedding(input_ids)
            x = model.x0.expand(1, -1)
            v = model.v0.expand(1, -1)
            for t in range(seq_len):
                force = forces[:, t]
                for layer in model.layers:
                    layer(x, v, force)
                    
    try:
        peak_mem = measure_peak_memory(model, full_analysis_pass)
    except:
        peak_mem = 0.0
        
    with torch.no_grad():
        # First-pass to get embedding forces
        forces = model.embedding(input_ids)
        
        x = model.x0.expand(1, -1)
        v = model.v0.expand(1, -1)
        
        for t in range(seq_len):
            force = forces[:, t]
            
            step_curvature = 0.0
            step_hamiltonian = 0.0
            step_fractal = 0.0
            
            # Layer-by-layer evolution to capture internals
            curr_x, curr_v = x, v
            for layer_idx, layer in enumerate(model.layers):
                # We need to peek into the layer forward
                # For this tool, we'll re-calculate the metrics manually 
                # to avoid changing the core library code too much
                
                # Estimate Curvature (L2 norm of Christoffel output)
                # In Manifold v1.0, layers are FractalMLayers
                # Macro Christoffel
                macro_gamma = 0.0
                
                # FIX: Normalize and Chunk for Multi-Head
                # Handle FractalMLayer vs MLayer
                if hasattr(layer, 'macro_manifold'):
                    active_layer = layer.macro_manifold
                else:
                    active_layer = layer
                
                # MLayer applies LayerNorm then chunks. We must replicate this logic.
                v_norm = active_layer.norm_v(curr_v)
                x_norm = active_layer.norm_x(curr_x)
                
                v_heads = v_norm.chunk(model.heads, dim=-1)
                x_heads = x_norm.chunk(model.heads, dim=-1)
                
                for i, c in enumerate(active_layer.christoffels):
                    # Pass head-specific v and x
                    gamma = c(v_heads[i], x_heads[i])
                    macro_gamma += torch.norm(gamma).item()
                
                # Check for Fractal Tunneling (Proxy check for visualization)
                threshold = layer.threshold if hasattr(layer, 'threshold') else 0.1 # Low threshold for random weights/viz
                gamma_0 = active_layer.christoffels[0](v_heads[0], x_heads[0])
                curv = torch.norm(gamma_0, dim=-1)
                
                if curv.mean() > threshold:
                    step_fractal += 1.0
                    
                # If truly recursive (FractalMLayer), we would check inner layers here
                if hasattr(layer, 'macro_manifold'):
                     pass # Already counted above as "macro" activity, strictly speaking tunneling implies going deeper
                
                # Symplectic Update
                curr_x, curr_v, _, _ = layer(curr_x, curr_v, force)
                
                # Hamiltonian Estimate (H = Kinetic + Potential)
                # K = 0.5 * v^2, P = -grad(phi) which is related to force
                # Here we use the Geodesic Deviation as a proxy for Energy
                ke = 0.5 * torch.norm(curr_v)**2
                step_hamiltonian += ke.item()
                step_curvature += macro_gamma
                
            curvature_history.append(step_curvature / len(model.layers))
            hamiltonian_history.append(step_hamiltonian / len(model.layers))
            fractal_activity.append(step_fractal)
            
            x, v = curr_x, curr_v

    # 3. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    sns.set_style("whitegrid")
    
    tokens = [id_to_token[i.item()] for i in input_ids[0]]
    x_ticks = np.arange(len(tokens))
    
    # Plot 1: Curvature (The "Difficulty" Meter)
    axes[0].plot(curvature_history, color='#E76F51', linewidth=3, marker='o', label="Semantic Curvature (Γ)")
    axes[0].set_ylabel("Curvature Magnitude", fontsize=12)
    axes[0].set_title("Cognitive Physics: Internal Difficulty Map", fontsize=14, fontweight='bold')
    axes[0].fill_between(x_ticks, 0, curvature_history, color='#E76F51', alpha=0.2)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Hamiltonian (The "Stability" Meter)
    axes[1].plot(hamiltonian_history, color='#2A9D8F', linewidth=3, marker='s', label="Hamiltonian Energy (H)")
    axes[1].set_ylabel("System Energy", fontsize=12)
    axes[1].set_title("System Stability: Hamiltonian Evolution", fontsize=14, fontweight='bold')
    axes[1].fill_between(x_ticks, 0, hamiltonian_history, color='#2A9D8F', alpha=0.2)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Fractal Activity (The "Zoom" Meter)
    axes[2].bar(x_ticks, fractal_activity, color='#264653', alpha=0.8, label="Fractal Tunneling (Active Layers)")
    axes[2].set_ylabel("Active Sub-Layers", fontsize=12)
    axes[2].set_title("Recursive Resolution: Fractal Activity", fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(tokens, fontsize=12)
    axes[2].set_xlabel("Input Sequence", fontsize=14)
    
    plt.tight_layout()
    
    # Save Result
    # Save Result
    output_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "internal_physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "xray_analysis.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Metrics to JSON
    import json
    metrics_data = {
        "input_text": input_text,
        "peak_vram_mb": peak_mem,
        "token_analysis": [
            {
                "token": t,
                "curvature": float(c),
                "energy": float(h),
                "fractal_activity": float(f)
            }
            for t, c, h, f in zip(tokens, curvature_history, hamiltonian_history, fractal_activity)
        ]
    }
    
    json_path = output_dir / "xray_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"\n✅ X-Ray analysis complete!")
    print(f"Chart saved to: {out_path}")
    print(f"Data saved to: {json_path}")
    print(f"Visual evidence of 'Carry Thinking': Look for Curvature peaks at logic transitions (e.g. '+', '=', or carries).")

if __name__ == "__main__":
    ckpt = "checkpoints/v1_0_math_oracle/epoch_1.pt" # Default path
    text = "999 + 1 = 1000"
    
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    if len(sys.argv) > 2:
        text = sys.argv[2]
        
    analyze_model_internals(ckpt, text)
