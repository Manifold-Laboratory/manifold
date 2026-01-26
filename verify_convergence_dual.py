
import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam

def run_experiment(mode_name, use_cuda_kernel):
    print(f"\n{'='*60}")
    print(f"Running Experiment: {mode_name}")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # Model
    model = Manifold(
        vocab_size=2, 
        dim=128, 
        depth=2, # Reduced depth for speed
        heads=4, 
        integrator_type='leapfrog'
    ).to(device)
    
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    model.train()
    
    # Data
    batch_size = 64
    seq_len = 16
    
    losses = []
    
    start_time = time.time()
    
    for step in range(50):
        # Task: Parity (Cumulative Sum % 2)
        x = torch.randint(0, 2, (batch_size, seq_len), device=device)
        y = torch.cumsum(x, dim=1) % 2
        
        optimizer.zero_grad()
        
        # Forward
        if use_cuda_kernel:
            # Normal path (uses leapfrog_fused if available)
            outputs = model(x)
        else:
            # Force Python path by requesting Christoffel symbols
            # This disables the CUDA kernel call in LeapfrogIntegrator
            outputs = model(x, collect_christ=True)
            
        logits = outputs[0]
        
        # Loss (MSE on implicit coordinates)
        coord_dim = 16
        mask = 2**torch.arange(coord_dim).to(device)
        target_bits = (y.unsqueeze(-1) & mask) > 0
        target_coords = target_bits.float() * 2 - 1
        
        # Readout output shape: [B, T, coord_dim] (ImplicitReadout)
        # But wait, model.readout depends on config. 
        # Default Manifold uses ImplicitReadout? 
        # Let's check output shape.
        if logits.shape[-1] == 2: # If vocab projection
             loss = nn.CrossEntropyLoss()(logits.view(-1, 2), y.view(-1))
        else:
             # Implicit Readout gives high dim
             # For this test we assume standard implicit readout behavior if set, 
             # but let's just use a simple linear projection on top if needed.
             # Actually the test_convergence.py assumed MSE on logits[:,:,0].
             # Let's replicate that logic, assuming ImplicitReadout.
             loss = nn.MSELoss()(logits[:, :, 0], target_coords[:, :, 0])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 10 == 0:
            pred_bits = (logits[:, :, 0] > 0.0).long()
            acc = (pred_bits == y).float().mean().item()
            print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%")
            
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Duration: {duration:.2f}s")
    
    return losses

if __name__ == "__main__":
    print("Verifying convergence for Leapfrog Integrator...")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping dual-mode test.")
        sys.exit(0)
        
    # 1. Python Mode
    losses_py = run_experiment("PYTHON Mode (Emulated)", use_cuda_kernel=False)
    
    # 2. CUDA Mode
    losses_cuda = run_experiment("CUDA Mode (Fused Kernel)", use_cuda_kernel=True)
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Python Final Loss: {losses_py[-1]:.4f}")
    print(f"CUDA Final Loss:   {losses_cuda[-1]:.4f}")
    
    # Check if both converged reasonably
    converged_py = losses_py[-1] < losses_py[0] * 0.8
    converged_cuda = losses_cuda[-1] < losses_cuda[0] * 0.8
    
    if converged_py and converged_cuda:
        print("SUCCESS: Both modes are learning.")
    else:
        print("WARNING: One or both modes failed to converge significantly.")
        
    # Check similarity of learning curves
    # They won't be identical because random data is generated each step inside the function
    # (unless we fix the data seed sequence, which we didn't perfectly do for the data generation loop)
    # But they should be in the same ballpark.
