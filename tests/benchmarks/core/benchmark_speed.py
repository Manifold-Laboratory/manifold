import torch
import time
import sys
import os
from pathlib import Path
import yaml

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

def run_performance_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🚀 MANIFOLD v1.0: PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Setup Standard Unified Model (Small to Large)
    configs = [
        {"name": "Small (v1.0)", "dim": 256, "depth": 6, "heads": 4},
        {"name": "Medium (v1.0)", "dim": 512, "depth": 12, "heads": 8},
        {"name": "Large (v1.0)", "dim": 1024, "depth": 24, "heads": 16}
    ]
    
    batch_size = 32
    seq_len = 128
    warmup_steps = 10
    bench_steps = 50
    
    results = []
    
    for cfg in configs:
        print(f"\nBenchmarking {cfg['name']}...")
        model = Manifold(
            vocab_size=100, 
            dim=cfg['dim'], 
            depth=cfg['depth'], 
            heads=cfg['heads'],
            physics_config={
                "active_inference": {"enabled": True},
                "symmetries": {"enabled": True},
                "fractal": {"enabled": True}
            }
        ).to(device)
        model.eval()
        
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        
        # Warmup
        for _ in range(warmup_steps):
            with torch.no_grad():
                _ = model(input_ids)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(bench_steps):
            with torch.no_grad():
                _ = model(input_ids)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / bench_steps) * 1000  # ms
        throughput = (batch_size * bench_steps) / total_time
        
        print(f">> Latency: {avg_time:.2f} ms/batch")
        print(f">> Throughput: {throughput:.2f} ex/s")
        
        results.append({
            "name": cfg['name'],
            "latency": avg_time,
            "throughput": throughput
        })

    # Summary Report
    print("\n" + "=" * 60)
    print("🏆 FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Model Name':<20} | {'Latency (ms)':<15} | {'Throughput (ex/s)':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['name']:<20} | {res['latency']:<15.2f} | {res['throughput']:<15.2f}")
    print("=" * 60)

if __name__ == "__main__":
    run_performance_benchmark()
