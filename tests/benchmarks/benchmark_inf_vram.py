import torch
import torch.nn as nn
import sys
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import Manifold

def measure_vram(vocab_size, embedding_type, device='cuda'):
    """
    Measures Peak VRAM and Param Count for a specific configuration.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Config
    physics_config = {
        'embedding': {
            'type': embedding_type,
            'coord_dim': 16
        },
        'active_inference': {'enabled': False}, # Disable for pure memory test
        'mixture': {'enabled': False}
    }
    
    try:
        # Init Model
        model = Manifold(
            vocab_size=vocab_size,
            dim=256, # Fixed hidden dim
            depth=2, # Shallow for embedding focus
            heads=4,
            integrator_type='heun',
            physics_config=physics_config
        ).to(device)
        
        # Params in Millions
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Dummy Input
        batch_size = 16
        seq_len = 64
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        # Forward + Backward to include activation/grad memory
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        logits, _, _ = model(x)
        loss = logits.mean()
        loss.backward()
        
        # Measure Peak VRAM
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        
        del model, x, logits, loss, optimizer
        return params, peak_mem
        
    except Exception as e:
        print(f"OOM or Error for {vocab_size} {embedding_type}: {e}")
        return None, None

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available. VRAM measurement invalid.")
        return

    vocab_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    
    results = []
    
    print(f"{'Vocab':<10} | {'Type':<10} | {'Params (M)':<12} | {'VRAM (MB)':<10}")
    print("-" * 50)
    
    for v in vocab_sizes:
        # Standard
        p_std, mem_std = measure_vram(v, 'standard', device)
        if p_std is not None:
             print(f"{v:<10} | {'Standard':<10} | {p_std:<12.2f} | {mem_std:<10.2f}")
             results.append({'Vocab': v, 'Type': 'Standard', 'Params': p_std, 'VRAM': mem_std})
        
        # Implicit
        p_inf, mem_inf = measure_vram(v, 'implicit', device)
        if p_inf is not None:
             print(f"{v:<10} | {'Implicit':<10} | {p_inf:<12.2f} | {mem_inf:<10.2f}")
             results.append({'Vocab': v, 'Type': 'Implicit', 'Params': p_inf, 'VRAM': mem_inf})
             
        # Functional
        p_fun, mem_fun = measure_vram(v, 'functional', device)
        if p_fun is not None:
             print(f"{v:<10} | {'Functional':<10} | {p_fun:<12.2f} | {mem_fun:<10.2f}")
             results.append({'Vocab': v, 'Type': 'Functional', 'Params': p_fun, 'VRAM': mem_fun})

    # Plotting
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Params
    plt.subplot(1, 2, 1)
    for t in ['Standard', 'Implicit', 'Functional']:
        data = df[df['Type'] == t]
        if not data.empty:
            plt.plot(data['Vocab'], data['Params'], marker='o', label=t)
    plt.title("Model Parameters vs Vocab Size")
    plt.xlabel("Vocab Size")
    plt.ylabel("Parameters (Millions)")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: VRAM
    plt.subplot(1, 2, 2)
    for t in ['Standard', 'Implicit', 'Functional']:
        data = df[df['Type'] == t]
        if not data.empty:
            plt.plot(data['Vocab'], data['VRAM'], marker='o', label=t)
    plt.title("Peak VRAM (Train Step) vs Vocab Size")
    plt.xlabel("Vocab Size")
    plt.ylabel("VRAM (MB)")
    plt.legend()
    plt.grid(True)
    
    out_path = PROJECT_ROOT / 'docs' / 'inf_vram_scaling.png'
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
