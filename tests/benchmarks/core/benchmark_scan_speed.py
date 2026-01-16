import torch
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import Manifold

def benchmark():
    print("=" * 60)
    print("🚀 BENCHMARK: Sequential (O(N)) vs Parallel Scan (O(log N))")
    print("=" * 60)
    
    vocab_size = 1000
    dim = 256
    depth = 4
    heads = 4
    batch_size = 4
    
    # Test lengths
    seq_lengths = [128, 512, 1024, 2048]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    results = []
    
    for L in seq_lengths:
        print(f"\n📏 Sequence Length: {L}")
        
        # Inputs
        dummy_input = torch.randint(0, vocab_size, (batch_size, L)).to(device)
        
        # --- Sequential Model ---
        model_seq = Manifold(vocab_size, dim, depth, heads=heads, use_scan=False).to(device)
        model_seq.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model_seq(dummy_input)
            
        # Time
        start = time.time()
        with torch.no_grad():
            for _ in range(5): # Run 5 times
                _ = model_seq(dummy_input)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.time()
        time_seq = (end - start) / 5.0
        
        print(f"   🐢 Sequential: {time_seq*1000:.2f} ms")
        
        # --- Parallel Model ---
        model_par = Manifold(vocab_size, dim, depth, heads=heads, use_scan=True).to(device)
        model_par.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model_par(dummy_input)
            
        # Time
        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model_par(dummy_input)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.time()
        time_par = (end - start) / 5.0
        
        print(f"   ⚡ Parallel:   {time_par*1000:.2f} ms")
        
        speedup = time_seq / time_par
        print(f"   🚀 Speedup:    {speedup:.2f}x")
        
        results.append((L, time_seq, time_par, speedup))
        
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Seq Len':<10} | {'Sequential':<15} | {'Parallel':<15} | {'Speedup':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res[0]:<10} | {res[1]*1000:6.2f} ms      | {res[2]*1000:6.2f} ms      | {res[3]:.2f}x")
    print("-" * 60)

if __name__ == "__main__":
    benchmark()
