"""
Enhanced Performance Benchmarks
================================

Professional-grade comparative analysis:
- GFN vs Transformer vs Mamba
- Memory scaling analysis with curve fitting
- Forward/backward pass breakdown
- Long sequence stress testing (up to 16K tokens)
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import GFN, AdjointGFN
try:
    from tests.benchmarks.baselines import MicroGPT, MicroMamba
except ImportError:
    from baselines import MicroGPT
    # Mamba might not be available
    MicroMamba = None


def benchmark_memory_speed(model, batch_size, seq_len, device, model_name, split_passes=False):
    """
    Enhanced benchmark with forward/backward split and better error handling.
    
    Args:
        split_passes: If True, measure forward and backward separately
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    inputs = torch.randint(0, 20, (batch_size, seq_len)).to(device)
    
    # Warmup
    try:
        with torch.no_grad():
            if "GFN" in model_name:
                output, _ = model(inputs)
            else:
                output = model(inputs)
    except RuntimeError as e:
        if "out of memory" in str(e):
            return {"memory_forward": float('nan'), "memory_backward": float('nan'), 
                    "memory_total": float('nan'), "throughput": 0, "oom": True}
        raise e
    
    torch.cuda.synchronize()
    
    # Measurement
    iters = 5
    total_tokens = batch_size * seq_len * iters
    
    results = {
        "memory_forward": 0,
        "memory_backward": 0,
        "memory_total": 0,
        "throughput": 0,
        "oom": False
    }
    
    try:
        if split_passes:
            # Measure forward pass only
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(iters):
                    if "GFN" in model_name:
                        output, _ = model(inputs)
                    else:
                        output = model(inputs)
            torch.cuda.synchronize()
            results["memory_forward"] = torch.cuda.max_memory_allocated() / 1024**2
            
            # Measure full pass (forward + backward)
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            for _ in range(iters):
                if "GFN" in model_name:
                    output, _ = model(inputs)
                else:
                    output = model(inputs)
                loss = output.mean()
                loss.backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            results["memory_total"] = torch.cuda.max_memory_allocated() / 1024**2
            results["memory_backward"] = results["memory_total"] - results["memory_forward"]
            results["throughput"] = total_tokens / (end_time - start_time)
        else:
            # Standard full measurement
            start_time = time.time()
            for _ in range(iters):
                if "GFN" in model_name:
                    output, _ = model(inputs)
                else:
                    output = model(inputs)
                loss = output.mean()
                loss.backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            results["memory_total"] = torch.cuda.max_memory_allocated() / 1024**2
            results["throughput"] = total_tokens / (end_time - start_time)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"      OOM at L={seq_len}")
            results["oom"] = True
            for key in results:
                if key != "oom":
                    results[key] = float('nan')
        else:
            raise e
    
    return results


def polynomial(x, a, b, c):
    """Helper for curve fitting: y = a*x^b + c"""
    return a * (x ** b) + c


def run_enhanced_suite():
    """Enhanced benchmark suite with scaling analysis."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("⚠️  WARNING: Running on CPU. Results will not be representative.")
        print("   For accurate benchmarks, use CUDA device.\n")
    
    print("=" * 70)
    print("  ENHANCED GFN PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    # Model configurations (~1.6M params each)
    vocab_size = 20
    dim = 512
    depth = 12
    rank = 16
    heads = 4
    
    models = {
        "GFN-Adjoint": AdjointGFN(vocab_size, dim, depth, rank).to(device),
        "MicroGPT": MicroGPT(vocab_size, dim, depth, heads).to(device)
    }
    
    # Add Mamba if available
    if MicroMamba is not None:
        try:
            models["Mamba"] = MicroMamba(vocab_size, dim, depth).to(device)
        except:
            print("⚠️  Mamba baseline not available, skipping\n")
    
    # Print parameter counts
    print("Model Configurations:")
    for name, m in models.items():
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {name:15s}: {params:.2f}M params")
    print()
    
    # Test configurations
    if device.type == 'cuda':
        lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
        batch_size = 4
    else:
        lengths = [128, 256, 512]
        batch_size = 2
    
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print(f"{'='*70}")
        model.train()
        
        for length in lengths:
            print(f"  Length {length:5d}...", end="", flush=True)
            metrics = benchmark_memory_speed(model, batch_size, length, device, name, split_passes=True)
            
            if metrics["oom"]:
                print(f" OOM")
            else:
                print(f" Mem: {metrics['memory_total']:6.1f}MB | Speed: {metrics['throughput']:7.1f} tok/s")
            
            results.append({
                "Model": name,
                "Sequence Length": length,
                "Peak Memory (MB)": metrics['memory_total'],
                "Forward Memory (MB)": metrics.get('memory_forward', float('nan')),
                "Backward Memory (MB)": metrics.get('memory_backward', float('nan')),
                "Throughput (tok/s)": metrics['throughput'],
                "OOM": metrics["oom"]
            })
    
    # Save results
    df = pd.DataFrame(results)
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "performance"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "enhanced_benchmark_data.csv", index=False)
    
    # Save Metrics to JSON (Standardized format)
    import json
    json_path = results_dir / "benchmark_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate visualizations
    _generate_plots(df, results_dir)
    
    print("\n" + "=" * 70)
    print("✓ Benchmark complete! Results saved to:")
    print(f"  {results_dir}/enhanced_benchmark_data.csv")
    print(f"  {results_dir}/benchmark_metrics.json")
    print(f"  {results_dir}/memory_scaling_enhanced.png")
    print("=" * 70)
    
    return df


def _generate_plots(df, results_dir):
    """Generate publication-quality plots with curve fitting."""
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Filter out OOM results for plotting
    df_valid = df[~df["OOM"]].copy()
    
    # === PLOT 1: Memory Scaling with Curve Fitting ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {"GFN-Adjoint": "#2A9D8F", "MicroGPT": "#E76F51", "Mamba": "#F4A261"}
    markers = {"GFN-Adjoint": "o", "MicroGPT": "s", "Mamba": "^"}
    
    for model_name in df_valid["Model"].unique():
        model_data = df_valid[df_valid["Model"] == model_name]
        x = model_data["Sequence Length"].values
        y = model_data["Peak Memory (MB)"].values
        
        # Plot actual data
        color = colors.get(model_name, "#264653")
        marker = markers.get(model_name, "o")
        ax1.plot(x, y, marker=marker, markersize=8, linewidth=2.5, 
                label=f'{model_name} (data)', color=color, alpha=0.7)
        
        # Fit curve: y = a*x^b + c
        try:
            if len(x) >= 3:
                popt, _ = curve_fit(polynomial, x, y, p0=[0.01, 1.5, 100], maxfev=5000)
                
                # Generate fitted curve
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = polynomial(x_fit, *popt)
                
                # Determine complexity class
                exponent = popt[1]
                if exponent < 1.2:
                    complexity = "O(N)"
                elif exponent < 1.8:
                    complexity = f"O(N^{exponent:.1f})"
                else:
                    complexity = "O(N²)"
                
                ax1.plot(x_fit, y_fit, '--', linewidth=2, color=color, alpha=0.5,
                        label=f'{model_name} fit: {complexity}')
        except:
            print(f"  ⚠️  Could not fit curve for {model_name}")
    
    ax1.set_xlabel('Sequence Length', fontsize=13)
    ax1.set_ylabel('Peak Memory (MB)', fontsize=13)
    ax1.set_title('Memory Scaling: GFN vs Baselines', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # === PLOT 2: Throughput Comparison ===
    sns.barplot(data=df_valid, x="Sequence Length", y="Throughput (tok/s)", 
                hue="Model", ax=ax2, palette=colors)
    ax2.set_xlabel('Sequence Length', fontsize=13)
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=13)
    ax2.set_title('Training Speed Comparison', fontsize=15, fontweight='bold')
    ax2.legend(title='Model', fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / "memory_scaling_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === PLOT 3: Forward vs Backward Memory Breakdown ===
    if "Forward Memory (MB)" in df_valid.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Reshape data for stacked bar
        models = df_valid["Model"].unique()
        lengths = sorted(df_valid["Sequence Length"].unique())
        
        x = np.arange(len(lengths))
        width = 0.25
        
        for i, model_name in enumerate(models):
            model_data = df_valid[df_valid["Model"] == model_name].sort_values("Sequence Length")
            forward = model_data["Forward Memory (MB)"].values
            backward = model_data["Backward Memory (MB)"].values
            
            offset = width * (i - len(models)/2 + 0.5)
            ax.bar(x + offset, forward, width, label=f'{model_name} Forward', 
                  color=colors.get(model_name, "#264653"), alpha=0.7)
            ax.bar(x + offset, backward, width, bottom=forward, 
                  label=f'{model_name} Backward', 
                  color=colors.get(model_name, "#264653"), alpha=0.4, hatch='//')
        
        ax.set_xlabel('Sequence Length', fontsize=13)
        ax.set_ylabel('Memory (MB)', fontsize=13)
        ax.set_title('Memory Breakdown: Forward vs Backward Pass', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lengths)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(results_dir / "memory_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n✓ Plots generated:")
    print("  - memory_scaling_enhanced.png (with curve fits)")
    print("  - memory_breakdown.png (forward/backward split)")


if __name__ == "__main__":
    df = run_enhanced_suite()
