
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
from tests.benchmarks.bench_utils import measure_peak_memory

def run_integrator_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Benchmarking Symplectic Integrators on {device}...")
    
    integrators = ['heun', 'rk4', 'rk45', 'leapfrog', 'symplectic']
    results = {
        'integrator': [],
        'drift_mean': [],
        'drift_std': [],
        'throughput': [],
        'vram': []
    }
    
    # Config
    vocab_size = 64
    dim = 512
    depth = 6
    heads = 8
    seq_len = 100
    batch_size = 16
    steps = 50 # Number of batches to measure
    
    for integ in integrators:
        print(f"\n[*] Testing Integrator: {integ.upper()}")
        
        try:
            model = Manifold(
                vocab_size=vocab_size, 
                dim=dim, 
                depth=depth, 
                heads=heads, 
                integrator_type=integ
            ).to(device)
            model.eval()
            
            # --- 1. VRAM Measurement ---
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
            # define closure for measurement
            def forward_pass():
                with torch.no_grad():
                     model(dummy_input)
                     
            peak_mem = measure_peak_memory(model, forward_pass)
            results['vram'].append(peak_mem)
            print(f"   Peak VRAM: {peak_mem:.1f} MB")
            
            # --- 2. Energy Drift Analysis ---
            # We measure drift over a long sequence to see stability
            # Manually stepping through layers to check v-norm conservation
            
            drift_values = []
            
            x = model.x0.expand(batch_size, -1)
            v = model.v0.expand(batch_size, -1)
            v_start_norm = v.norm(dim=-1).mean().item()
            
            # Create a long "thought" trajectory (no force)
            # F=0 => Should conserve energy exactly if symplectic
            with torch.no_grad():
                for _ in range(50): # 50 steps deep
                    for layer in model.layers:
                         # Force = 0
                        out = layer(x, v, force=None)
                        x, v = out[0], out[1]
                
                v_end_norm = v.norm(dim=-1).mean().item()
                # Cumulative energy drift proxy: |v_end - v_start| / v_start
                # This is cumulative drift over all 50 steps, not per-step drift
                drift = abs(v_end_norm - v_start_norm) / (v_start_norm + 1e-8)
                results['drift_mean'].append(drift * 100) # Percentage
                results['drift_std'].append(0.0) # Single run for drift
                print(f"   Cumulative Energy Drift (50 steps): {drift*100:.4f}%")

            # --- 3. Throughput ---
            # Run X batches
            start_time = time.time()
            with torch.no_grad():
                for _ in range(steps):
                    model(dummy_input)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = (steps * batch_size) / total_time
            results['throughput'].append(throughput)
            results['integrator'].append(integ)
            
            print(f"   Throughput: {throughput:.1f} seq/s")
            
        except Exception as e:
            print(f"[*] Failed {integ}: {e}")
            results['integrator'].append(integ)
            results['drift_mean'].append(None)
            results['drift_std'].append(None)
            results['throughput'].append(None)
            results['vram'].append(None)

    # === Visualization ===
    print("\n[*] Generating Benchmark Plot...")
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/integrators"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_style("whitegrid")
    
    # filter failed
    valid_indices = [i for i, v in enumerate(results['drift_mean']) if v is not None]
    valid_integs = [results['integrator'][i] for i in valid_indices]
    
    # 1. Energy Drift
    drifts = [results['drift_mean'][i] for i in valid_indices]
    sns.barplot(x=valid_integs, y=drifts, ax=axes[0], palette="magma")
    axes[0].set_title("Energy Drift (Lower is Better)")
    axes[0].set_ylabel("Drift % (50 steps)")
    axes[0].set_yscale("log") # Drift can vary wildly
    
    # 2. VRAM
    vrams = [results['vram'][i] for i in valid_indices]
    sns.barplot(x=valid_integs, y=vrams, ax=axes[1], palette="viridis")
    axes[1].set_title("Peak VRAM Usage")
    axes[1].set_ylabel("MB")
    
    # 3. Throughput
    thrus = [results['throughput'][i] for i in valid_indices]
    sns.barplot(x=valid_integs, y=thrus, ax=axes[2], palette="rocket")
    axes[2].set_title("Inference Speed")
    axes[2].set_ylabel("Sequences / sec")
    
    plt.suptitle("Manifold Integrator Benchmark: Symplectic vs Runge-Kutta", fontsize=16)
    plt.tight_layout()
    plt.savefig(res_dir / "integrator_comparison.png")
    
    # Save JSON
    import json
    json_path = res_dir / "integrator_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[*] Results saved to {res_dir}")

if __name__ == "__main__":
    run_integrator_benchmark()
