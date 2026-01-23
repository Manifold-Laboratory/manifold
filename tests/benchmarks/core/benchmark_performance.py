"""
Enhanced Performance Benchmarks
================================

Professional-grade comparative analysis:
- GFN vs Transformer vs Mamba
- Memory scaling analysis with curve fitting
- Forward/backward pass breakdown
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import sys
import gc

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

# Optional Baselines
try:
    from tests.benchmarks.baselines import MicroGPT, MicroMamba
except ImportError:
    # Minimal Fallbacks
    class MicroGPT(nn.Module):
        def __init__(self, vocab_size, dim, depth, heads, max_len=16384):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True) for _ in range(depth)])
            self.head = nn.Linear(dim, vocab_size)
        def forward(self, x):
            x = self.emb(x)
            for b in self.blocks: x = b(x)
            return self.head(x)
    MicroMamba = None

def benchmark_memory_speed(model, batch_size, seq_len, device, model_name):
    """Measures Peak VRAM and Throughput."""
    model.train() # Measure training mode (most heavy)
    x = torch.randint(0, 16, (batch_size, seq_len)).to(device)
    
    # 1. Throughput (Forward + Backward)
    warmup = 5
    runs = 10
    
    for _ in range(warmup):
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        out.mean().backward()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        out.mean().backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    throughput = (runs * batch_size) / elapsed
    
    # 2. Peak VRAM
    def run_step():
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        out.mean().backward()
        
    vram = PerformanceStats.measure_peak_memory(model, run_step)
    
    return throughput, vram

def polynomial(x, a, b, c):
    """y = a * x^b + c"""
    return a * np.power(x, b) + c

def run_performance_suite():
    logger = ResultsLogger("performance", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("❌ Speed benchmark requires CUDA.")
        return

    # 1. Configs
    dim = 256
    depth = 6
    heads = 4
    vocab_size = 16
    
    models = {
        "Manifold (O(1))": Manifold(vocab_size, dim, depth, heads).to(device),
        "Transformer (O(N^2))": MicroGPT(vocab_size, dim, depth, heads).to(device)
    }
    
    if MicroMamba:
        models["Mamba (O(N))"] = MicroMamba(vocab_size, dim, depth).to(device)
        
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192, 16384]
    batch_size = 1
    
    results = []
    
    for name, model in models.items():
        print(f"\n🚀 Benchmarking {name}...")
        for L in seq_lengths:
            try:
                gc.collect()
                torch.cuda.empty_cache()
                
                tput, vram = benchmark_memory_speed(model, batch_size, L, device, name)
                results.append({
                    "Model": name,
                    "Sequence Length": L,
                    "Throughput (seq/s)": tput,
                    "VRAM (MB)": vram
                })
                print(f"  L={L:5d} | {tput:6.1f} seq/s | {vram:7.1f} MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  L={L:5d} | OOM")
                    break
                else:
                    raise e
                    
    df = pd.DataFrame(results)
    logger.save_json(results)
    
    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {"Manifold (O(1))": "#2A9D8F", "Transformer (O(N^2))": "#E76F51", "Mamba (O(N))": "#264653"}
    
    for name in df["Model"].unique():
        data = df[df["Model"] == name]
        color = colors.get(name, None)
        
        # Memory Plot
        ax1.plot(data["Sequence Length"], data["VRAM (MB)"], 'o', label=f"{name} (Observed)", color=color)
        
        # Fit Theoretical Curve
        if len(data) >= 3:
            try:
                popt, _ = curve_fit(polynomial, data["Sequence Length"], data["VRAM (MB)"], p0=[0.001, 1, 100])
                x_fit = np.linspace(min(data["Sequence Length"]), 16384, 100)
                y_fit = polynomial(x_fit, *popt)
                ax1.plot(x_fit, y_fit, '--', alpha=0.5, color=color)
            except:
                ax1.plot(data["Sequence Length"], data["VRAM (MB)"], '-', alpha=0.5, color=color)

        # Throughput Plot
        ax2.plot(data["Sequence Length"], data["Throughput (seq/s)"], 'o-', label=name, color=color)

    ax1.set_title("VRAM Usage Scaling")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Peak Memory (MB)")
    ax1.set_xscale('log', base=2)
    ax1.legend()
    
    ax2.set_title("Training Throughput")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Sequences per second")
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    logger.save_plot(fig, "scaling_analysis.png")
    
    return df

if __name__ == "__main__":
    run_performance_suite()
