import torch
import torch.nn as nn
import sys
import gc
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Create results directory
RESULTS_DIR = PROJECT_ROOT / 'tests' / 'benchmarks' / 'results' / 'infinite_scaling'
os.makedirs(RESULTS_DIR, exist_ok=True)

from gfn.model import Manifold

def measure_vram_infinite(vocab_size, device='cuda'):
    """
    Measures Peak VRAM and Param Count for Infinite Mode only.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Config for Infinite Mode
    physics_config = {
        'embedding': {
            'type': 'functional',
            'coord_dim': 32 # Higher fidelity for massive vocabs
        },
        'readout': {
            'type': 'implicit'
        },
        'active_inference': {'enabled': False},
        'mixture': {'enabled': False}
    }
    
    try:
        # Init Model
        model = Manifold(
            vocab_size=vocab_size,
            dim=256, 
            depth=2, 
            heads=4,
            integrator_type='heun',
            physics_config=physics_config
        ).to(device)
        
        # Params in Millions
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Dummy Input
        batch_size = 4
        seq_len = 32
        # Inputs must be within vocab range
        x = torch.randint(0, min(vocab_size, 10000), (batch_size, seq_len)).to(device)
        
        # Forward + Backward
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        logits, _, _ = model(x)
        
        # For implicit readout, logits are coordinate vectors, not class probs
        # So we don't compute CE loss (which would be O(V)). 
        # We compute a dummy regression loss to simulate gradient flow.
        loss = logits.mean() 
        loss.backward()
        
        # Measure Peak VRAM
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        
        del model, x, logits, loss, optimizer
        return params, peak_mem
        
    except Exception as e:
        print(f"Error for {vocab_size}: {e}")
        return None, None

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available. VRAM measurement invalid.")
        return

    # Massive Vocab Sizes
    vocab_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000] 
    
    results = []
    
    print(f"{'Vocab':<15} | {'Params (M)':<12} | {'VRAM (MB)':<10}")
    print("-" * 50)
    
    for v in vocab_sizes:
        p, mem = measure_vram_infinite(v, device)
        
        if p is not None:
            print(f"{v:<15} | {p:<12.2f} | {mem:<10.2f}")
            results.append({
                'Vocab': v, 
                'Params': p, 
                'VRAM': mem
            })
        else:
            print(f"{v:<15} | FAILED")
             
    # Save Data
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f'data_{timestamp}.json'
    df.to_json(json_path, orient='records', indent=2)
    
    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        
        # VRAM Plot
        plt.plot(df['Vocab'], df['VRAM'], marker='o', linewidth=2, color='#00aaff', label='Manifold Infinite')
        
        plt.xscale('log')
        plt.title("Infinite Readout: VRAM Scaling vs Vocabulary Size")
        plt.xlabel("Vocabulary Size (Log Scale)")
        plt.ylabel("Peak VRAM (MB)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        # Annotate last point
        last = df.iloc[-1]
        plt.annotate(f"{last['VRAM']:.1f} MB @ 1B Vocab", 
                     (last['Vocab'], last['VRAM']),
                     xytext=(10, 10), textcoords='offset points')

        plot_path = RESULTS_DIR / 'infinite_scaling_plot.png'
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_benchmark()
