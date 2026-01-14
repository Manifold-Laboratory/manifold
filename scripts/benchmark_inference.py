
import time
import torch
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import needed modules
from src.model import Manifold
import src.cuda.ops

def run_benchmark(model, device, use_cuda_kernels=True, num_steps=100, batch_size=32, seq_len=128):
    """Run inference loop and measure speed."""
    
    # Configure global kernel usage
    # We monkeypatch the ops module to simulate availability/unavailability
    original_availability = src.cuda.ops.CUDA_AVAILABLE
    if not use_cuda_kernels:
        src.cuda.ops.CUDA_AVAILABLE = False
        print(">>> FORCING CUDA KERNELS: OFF (Using PyTorch Fallback)")
    else:
        # Ensure it's true if available
        if not original_availability:
             print(">>> WARNING: CUDA Kernels not actually available on system!")
        else:
             print(">>> FORCING CUDA KERNELS: ON")
            
    # Dummy input
    x = torch.randint(0, 65, (batch_size, seq_len)).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking {num_steps} forward passes...")
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_steps):
            if i % 10 == 0:
                print(f"  Step {i}/{num_steps}...", flush=True)
            # Feed previous output as input (simple autoregressive simulation)
            # Just running forward pass is enough to exercise the kernels
            _ = model(x)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Restore state
    src.cuda.ops.CUDA_AVAILABLE = original_availability
    
    avg_time = (end_time - start_time) / num_steps
    tokens_per_sec = (batch_size * seq_len) / avg_time
    
    print(f"Time per step: {avg_time*1000:.2f} ms")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    return tokens_per_sec

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model (Medium Config)
    model = Manifold(
        vocab_size=65,
        dim=512,
        depth=24,
        rank=16,
        heads=8,
        use_scan=False # Must be False to use sequential LeapfrogIntegrator where kernels are!
    ).to(device)
    model.eval()
    
    print("\n" + "="*50)
    print("BENCHMARK: PyTorch Fallback")
    print("="*50)
    speed_py = run_benchmark(model, device, use_cuda_kernels=False)
    
    print("\n" + "="*50)
    print("BENCHMARK: Fused CUDA Kernels")
    print("="*50)
    speed_cuda = run_benchmark(model, device, use_cuda_kernels=True)
    
    print("\n" + "="*50)
    print(f"SPEEDUP: {speed_cuda / speed_py:.2f}x")
    print("="*50)

if __name__ == "__main__":
    main()
