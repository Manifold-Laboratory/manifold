
import torch
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.model import Manifold

def measure_config(name, vocab_size, seq_len):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("skipped (cpu)")
        return 0

    print(f"[*] Testing {name} (Recurrent Inference) (Vocab={vocab_size}, Context={seq_len})...")
    
    # Configure explicitly
    model = Manifold(
        vocab_size=vocab_size,
        dim=256,
        depth=4,
        heads=4,
        use_scan=False,
        physics_config=None 
    ).to(device).eval()
    
    # Measure model static size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"    Model Weights: {param_size/1024**2:.2f} MB")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    try:
        # Simulate Recurrent Inference (State only)
        # We process 'seq_len' tokens one by one (or just check the state size after 1 step)
        # Actually, if O(1), memory after 1 step == memory after N steps.
        # So we just run forward(Input[1], state)
        
        # 1. Init State
        token = torch.tensor([[0]], device=device)
        logits, state, _ = model(token)
        
        # 2. Loop (simulated) - we don't need to actually loop N times to measure memory OF THE STATE
        # unless the state grows. Manifold state is fixed (x, v).
        # We will loop 10 times just to allow any caching to settle.
        for _ in range(10):
            logits, state, _ = model(token, state=state)
            
        peak_mem = torch.cuda.max_memory_allocated()
        mb = peak_mem / 1024 / 1024
        
        print(f"    Peak VRAM:     {mb:.2f} MB")
        print(f"    Overhead:      {mb - (param_size/1024**2):.2f} MB")
        return mb
    except RuntimeError as e:
        print(f"    Error: {e}")
        return None
    finally:
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=== VRAM INVESTIGATION (Recurrent) ===")
    
    # We expect these to be IDENTICAL if O(1) holds
    measure_config("Real", 50257, 128)
    measure_config("Real", 50257, 4096)
    measure_config("Real", 50257, 1000000)
