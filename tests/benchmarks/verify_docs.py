"""
Verification Script for Documentation Claims
=============================================
Measures actual parameter counts, memory usage, and throughput for Manifold models.
"""

import torch
import sys
from pathlib import Path
import time
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_memory_and_throughput(model, batch_size=32, seq_len=128, warmup=5, runs=20):
    """Measure peak memory and throughput."""
    device = next(model.parameters()).device
    
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
    end = time.time()
    
    # Metrics
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    throughput = (runs * batch_size) / (end - start)
    
    return peak_mem, throughput

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*60)
    
    # Define model configurations to test
    configs = [
        {"name": "Small", "dim": 256, "depth": 6, "heads": 4, "rank": 16},
        {"name": "Medium", "dim": 512, "depth": 12, "heads": 8, "rank": 32},
        {"name": "Large", "dim": 1024, "depth": 24, "heads": 16, "rank": 64},
    ]
    
    results = {}
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {cfg['name']}")
        print(f"Config: dim={cfg['dim']}, depth={cfg['depth']}, heads={cfg['heads']}")
        
        try:
            model = Manifold(
                vocab_size=64,
                dim=cfg['dim'],
                depth=cfg['depth'],
                heads=cfg['heads'],
                rank=cfg['rank']
            ).to(device)
            
            params = count_parameters(model)
            print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
            
            if torch.cuda.is_available():
                peak_mem, throughput = measure_memory_and_throughput(model)
                print(f"Peak VRAM (Inference): {peak_mem:.3f} GB")
                print(f"Throughput: {throughput:.2f} seq/s (batch=32, seq=128)")
            else:
                peak_mem, throughput = 0, 0
                print("CUDA not available - skipping memory/throughput")
            
            results[cfg['name']] = {
                "params": params,
                "params_M": round(params/1e6, 2),
                "peak_vram_gb": round(peak_mem, 3),
                "throughput": round(throughput, 2)
            }
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"ERROR: {e}")
            results[cfg['name']] = {"error": str(e)}
    
    # Save results
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    res_path = PROJECT_ROOT / "tests/benchmarks/results/verification.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {res_path}")

if __name__ == "__main__":
    main()
