"""
Feature Overhead Benchmark
==========================
Measures the cost (VRAM, throughput, latency) of each physics feature.

Answers: "What does each feature cost?"
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


def get_physics_configs():
    """Return configurations for ablation."""
    return {
        "baseline": None,
        "+active_inference": {
            'active_inference': {'enabled': True}
        },
        "+reactive_curvature": {
            'active_inference': {
                'enabled': True,
                'reactive_curvature': {'enabled': True, 'plasticity': 0.1}
            }
        },
        "+singularities": {
            'active_inference': {
                'enabled': True,
                'singularities': {'enabled': True, 'strength': 5.0}
            }
        },
        "+dynamic_time": {
            'active_inference': {
                'enabled': True,
                'dynamic_time': {'enabled': True, 'range': [0.1, 3.0]}
            }
        },
        "+symmetries": {
            'active_inference': {'enabled': True},
            'symmetries': {'enabled': True, 'isomeric_groups': [[0, 1], [2, 3]]}
        },
        "+fractal": {
            'active_inference': {'enabled': True},
            'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2}
        },
        "all_features": {
            'active_inference': {
                'enabled': True,
                'reactive_curvature': {'enabled': True, 'plasticity': 0.1},
                'singularities': {'enabled': True, 'strength': 5.0},
                'dynamic_time': {'enabled': True, 'range': [0.1, 3.0]}
            },
            'symmetries': {'enabled': True, 'isomeric_groups': [[0, 1], [2, 3]]},
            'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
            'stability': {'curvature_clamp': 10.0}
        }
    }


def measure_overhead(config_name, physics_config, device, batch_size=16, seq_len=128, warmup=3, runs=10):
    """Measure VRAM and throughput for a configuration."""
    model = Manifold(
        vocab_size=64,
        dim=256,
        depth=6,
        heads=4,
        integrator_type='heun',
        physics_config=physics_config
    ).to(device).eval()
    
    params = sum(p.numel() for p in model.parameters())
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.randint(0, 64, (batch_size, seq_len)).to(device)
            model(x)
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    # Timed runs
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            x = torch.randint(0, 64, (batch_size, seq_len)).to(device)
            model(x)
    elapsed = time.time() - start
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    throughput = (runs * batch_size) / elapsed
    latency_ms = (elapsed / runs) * 1000
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'params': params,
        'vram_mb': round(peak_vram, 2),
        'throughput': round(throughput, 2),
        'latency_ms': round(latency_ms, 2)
    }


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Feature Overhead Benchmark on {device}")
    print("="*60)
    
    configs = get_physics_configs()
    results = {}
    
    for name, physics_config in configs.items():
        print(f"\n[*][*]  Testing: {name}...")
        try:
            metrics = measure_overhead(name, physics_config, device)
            results[name] = metrics
            print(f"   VRAM: {metrics['vram_mb']:.1f} MB | "
                  f"Throughput: {metrics['throughput']:.1f} seq/s | "
                  f"Latency: {metrics['latency_ms']:.1f} ms")
        except Exception as e:
            print(f"   ERROR: {e}")
            results[name] = {'error': str(e)}
    
    # Save results
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/validation"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / "overhead_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plot
    valid_names = [k for k, v in results.items() if 'error' not in v]
    vram_vals = [results[k]['vram_mb'] for k in valid_names]
    throughput_vals = [results[k]['throughput'] for k in valid_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # VRAM chart
    bars1 = ax1.barh(valid_names, vram_vals, color='steelblue')
    ax1.set_xlabel('Peak VRAM (MB)')
    ax1.set_title('Memory Overhead by Feature')
    ax1.bar_label(bars1, fmt='%.1f')
    
    # Throughput chart  
    bars2 = ax2.barh(valid_names, throughput_vals, color='seagreen')
    ax2.set_xlabel('Throughput (seq/s)')
    ax2.set_title('Speed Impact by Feature')
    ax2.bar_label(bars2, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(res_dir / "overhead_comparison.png", dpi=150, bbox_inches='tight')
    
    print(f"\n[*] Results saved to {res_dir}")
    
    # Summary
    if 'baseline' in results and 'all_features' in results:
        base = results['baseline']
        full = results['all_features']
        vram_overhead = ((full['vram_mb'] - base['vram_mb']) / base['vram_mb']) * 100
        speed_overhead = ((base['throughput'] - full['throughput']) / base['throughput']) * 100
        print(f"\n[*] SUMMARY:")
        print(f"   Full features add {vram_overhead:.1f}% VRAM overhead")
        print(f"   Full features reduce throughput by {speed_overhead:.1f}%")
    
    return results


if __name__ == "__main__":
    run_benchmark()
