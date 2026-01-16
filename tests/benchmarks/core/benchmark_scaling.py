"""
Scaling Laws Benchmark
======================
Tests how Manifold scales with model size.

Answers: "How does Manifold scale?"
"""

import torch
import time
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold


def measure_scaling(dim, depth, heads, device, batch_size=16, seq_len=128):
    """Measure metrics for a model configuration."""
    model = Manifold(
        vocab_size=64,
        dim=dim,
        depth=depth,
        heads=heads,
        integrator_type='heun'
    ).to(device).eval()
    
    params = sum(p.numel() for p in model.parameters())
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            x = torch.randint(0, 64, (batch_size, seq_len)).to(device)
            model(x)
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    # Measure
    runs = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            x = torch.randint(0, 64, (batch_size, seq_len)).to(device)
            model(x)
    elapsed = time.time() - start
    
    vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    throughput = (runs * batch_size) / elapsed
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'params': params,
        'params_M': round(params / 1e6, 2),
        'vram_mb': round(vram, 2),
        'throughput': round(throughput, 2)
    }


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Scaling Laws Benchmark on {device}")
    print("="*60)
    
    # Test configurations
    configs = [
        {'dim': 128, 'depth': 2, 'heads': 2},
        {'dim': 128, 'depth': 4, 'heads': 2},
        {'dim': 256, 'depth': 4, 'heads': 4},
        {'dim': 256, 'depth': 8, 'heads': 4},
        {'dim': 512, 'depth': 6, 'heads': 8},
        {'dim': 512, 'depth': 12, 'heads': 8},
    ]
    
    results = []
    
    for cfg in configs:
        name = f"d{cfg['dim']}_L{cfg['depth']}"
        print(f"\n[*][*]  Testing: {name}...")
        try:
            metrics = measure_scaling(cfg['dim'], cfg['depth'], cfg['heads'], device)
            metrics['config'] = cfg
            metrics['name'] = name
            results.append(metrics)
            print(f"   Params: {metrics['params_M']:.2f}M | "
                  f"VRAM: {metrics['vram_mb']:.1f} MB | "
                  f"Throughput: {metrics['throughput']:.1f} seq/s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   OOM - skipping")
            else:
                raise e
    
    # Save results
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/validation"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / "scaling_laws.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    params = [r['params_M'] for r in results]
    vram = [r['vram_mb'] for r in results]
    throughput = [r['throughput'] for r in results]
    names = [r['name'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Params vs VRAM
    axes[0].scatter(params, vram, s=100, c='steelblue')
    for i, name in enumerate(names):
        axes[0].annotate(name, (params[i], vram[i]), fontsize=8)
    axes[0].set_xlabel('Parameters (M)')
    axes[0].set_ylabel('VRAM (MB)')
    axes[0].set_title('Memory Scaling')
    
    # Params vs Throughput
    axes[1].scatter(params, throughput, s=100, c='seagreen')
    for i, name in enumerate(names):
        axes[1].annotate(name, (params[i], throughput[i]), fontsize=8)
    axes[1].set_xlabel('Parameters (M)')
    axes[1].set_ylabel('Throughput (seq/s)')
    axes[1].set_title('Speed Scaling')
    
    # Bar chart
    x = range(len(names))
    axes[2].bar(x, params, color='steelblue', alpha=0.7, label='Params (M)')
    ax2 = axes[2].twinx()
    ax2.plot(x, throughput, 'o-', color='seagreen', label='Throughput')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_ylabel('Parameters (M)')
    ax2.set_ylabel('Throughput (seq/s)')
    axes[2].set_title('Size vs Speed Trade-off')
    
    plt.tight_layout()
    plt.savefig(res_dir / "scaling_curves.png", dpi=150, bbox_inches='tight')
    
    print(f"\n[*] Results saved to {res_dir}")
    
    return results


if __name__ == "__main__":
    run_benchmark()
