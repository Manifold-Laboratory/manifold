import torch
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from gfn.model import Manifold

def debug_kernel():
    print("[-] Initializing Minimal Manifold...")
    device = torch.device('cuda')
    
    # Tiny Config
    dim = 128
    model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1, integrator_type='leapfrog').to(device)
    
    # Dummy Input
    B, L = 16, 20
    x = torch.randint(0, 10, (B, L)).to(device)
    
    print("[-] Starting Forward Pass...")
    t0 = time.time()
    try:
        logits, (state_x, state_v), _ = model(x, collect_christ=False)
        torch.cuda.synchronize()
        print(f"[+] Forward Done. Time: {time.time()-t0:.4f}s")
    except Exception as e:
        print(f"[!] Forward Failed: {e}")
        return

    print("[-] Starting Backward Pass...")
    t0 = time.time()
    try:
        loss = logits.sum()
        loss.backward()
        torch.cuda.synchronize()
        print(f"[+] Backward Done. Time: {time.time()-t0:.4f}s")
    except Exception as e:
        print(f"[!] Backward Failed: {e}")
        
    print("[SUCCESS] Kernel is operational.")

if __name__ == "__main__":
    debug_kernel()
