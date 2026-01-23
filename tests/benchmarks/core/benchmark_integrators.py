"""
Standardized Integrator Performance Analysis
============================================

Scientific evaluation of all implemented integrators:
- Symplectic: Leapfrog, Yoshida (4th), ForestRuth (4th), Omelyan (4th), Coupling (Inf).
- Runge-Kutta: Euler (1st), Heun (2nd), RK4 (4th), DormandPrince (5th).
- Neural: NeuralIntegrator (Learnable).

Metrics:
- Cumulative Energy Drift (%)
- Inference Throughput (seq/s)
- Peak VRAM (MB)
"""

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

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

def run_integrator_suite():
    logger = ResultsLogger("integrators", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # All implemented integrators
    integrators = [
        'euler', 'heun', 'rk4', 'rk45', 
        'leapfrog', 'symplectic', 'yoshida', 'forest_ruth', 'omelyan',
        'coupling', 'neural'
    ]
    
    # Config
    vocab_size = 64
    dim = 512
    depth = 4   # Shallower for faster stats
    heads = 1   # Single head for pure integrator drift test
    seq_len = 128
    batch_size = 16
    drift_steps = 100 # Steps to measure drift
    
    results = []

    print(f"🚀 Starting Integrator Suite on {device}...")

    for integ in integrators:
        print(f"\n📊 Testing: {integ.upper()}")
        try:
            model = Manifold(
                vocab_size=vocab_size,
                dim=dim,
                depth=depth,
                heads=heads,
                integrator_type=integ
            ).to(device)
            model.eval()

            # 1. Performance Statistics
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
            
            def run_inference():
                with torch.no_grad():
                    model(dummy_input)

            vram = PerformanceStats.measure_peak_memory(model, run_inference)
            
            # Throughput
            start = time.time()
            with torch.no_grad():
                for _ in range(10): model(dummy_input)
            tput = (10 * batch_size) / (time.time() - start)

            # 2. Energy Drift (The Physics Metric)
            # Concept: In a system with Force=0, velocity magnitude should be constant.
            # Any change is numerical drift.
            
            with torch.no_grad():
                # Get initial state from internal embeddings
                x = torch.zeros(batch_size, dim).to(device) # Simplified initial state
                v = torch.randn(batch_size, dim).to(device)
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-6) # Unit magnitude
                
                v_start_norm = v.norm(dim=-1).mean().item()
                
                # Iterate through drift steps using a single layer representing the dynamics
                # We isolate the integrator to avoid random projections distorting scientific drift measurement
                layer = model.layers[0]
                integrator = layer.integrators[0]
                
                # Call integrator once with all drift steps (RECURRENT FUSION)
                x, v = integrator(x, v, force=None, steps=drift_steps)
                
                v_end_norm = v.norm(dim=-1).mean().item()
                drift_percent = abs(v_end_norm - v_start_norm) / (v_start_norm + 1e-8) * 100

            print(f"   Drift: {drift_percent:.6f}% | Memory: {vram:.1f} MB | Speed: {tput:.1f} seq/s")
            
            results.append({
                "Integrator": integ,
                "Type": "Symplectic" if integ in ['leapfrog', 'symplectic', 'yoshida', 'forest_ruth', 'omelyan', 'coupling'] else "Standard",
                "Drift (%)": drift_percent,
                "Inference Speed": tput,
                "VRAM (MB)": vram
            })

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    # 3. Save & Plot
    logger.save_json(results)
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    if not results:
        print("❌ All tests failed. No results to plot.")
        return pd.DataFrame()

    plt.tight_layout()
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.barplot(data=df, x="Integrator", y="Drift (%)", ax=axes[0], palette="viridis")
        axes[0].set_title("Numerical Energy Drift")
        axes[0].set_yscale("log")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        
        sns.barplot(data=df, x="Integrator", y="Inference Speed", ax=axes[1], palette="magma")
        axes[1].set_title("Throughput (seq/s)")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        
        sns.barplot(data=df, x="Integrator", y="VRAM (MB)", ax=axes[2], palette="rocket")
        axes[2].set_title("VRAM Consumption")
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        logger.save_plot(fig, "integrator_comparison.png")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    return df

if __name__ == "__main__":
    run_integrator_suite()
