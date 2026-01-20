import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GFN Models & Physics
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import hamiltonian_loss, geodesic_regularization

# Import Baselines & Utils
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import measure_peak_memory

def setup_plotting():
    plt.rcParams.update({'font.size': 12})
    sns.set_style("whitegrid")

class ParityTask:
    """
    Parity Check (Modulo 2).
    Input: "0 1 0 1 1"
    Target: "0 1 1 0 1" (Cumulative XOR / Sum Mod 2)
    """
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y = torch.cumsum(x, dim=1) % self.mod
        return x, y

def train_step_manifold(model, optimizer, scheduler, inputs, targets, device):
    """
    Training step for Manifold with O(1) Binary Logic & Physics.
    """
    optimizer.zero_grad()
    
    # 1. Forward Pass
    # Returns: logits (coords), state (x,v), christoffels
    # Note: Sequential mode returns (logits, (x, v), christoffels)
    logits, (x_final, v_final), christoffels = model(inputs)
    
    # 2. O(1) Target Generation
    # Targets are IDs {0, 1}. We map them to Binary Coordinates {-1, 1}
    # This aligns the latent space with the implicit readout.
    coord_dim = model.physics_config.get('embedding', {}).get('coord_dim', 16)
    
    # Generate bitmask for target IDs
    mask = 2**torch.arange(coord_dim).to(device)
    target_bits = (targets.unsqueeze(-1) & mask) > 0 # [B, L, D]
    target_coords = target_bits.float() * 2 - 1      # {-1, 1}
    
    # 3. Task Loss (MSE on Coordinates)
    # We only care about the bits relevant to the vocab size
    # For Parity (vocab=2), strictly 1 bit is needed.
    v_bits = int(np.ceil(np.log2(2))) # = 1
    
    # Regression Loss
    loss_mse = nn.MSELoss()(logits[:, :, :v_bits], target_coords[:, :, :v_bits])
    
    # 4. Physics Regularization
    loss_phy = 0.0
    
    # Geodesic Reg (Controls curvature / Energy Stability)
    # This acts as the proxy for Hamiltonian stability by preventing metric explosion
    if christoffels:
        loss_phy += geodesic_regularization(None, christoffels, lambda_g=0.001)
        
    total_loss = loss_mse + loss_phy
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    optimizer.step()
    if scheduler: scheduler.step()
    
    # Accuracy Check
    pred_bits = (logits[:, :, 0] > 0.0).long()
    acc = (pred_bits == targets).float().mean().item()
    
    return total_loss.item(), acc

def train_step_gpt(model, optimizer, scheduler, inputs, targets, device):
    """
    Standard Training Step for Transformer (MicroGPT).
    """
    optimizer.zero_grad()
    logits = model(inputs) # [B, L, V]
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, 2), targets.view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    if scheduler: scheduler.step()
    
    preds = logits.argmax(dim=-1)
    acc = (preds == targets).float().mean().item()
    
    return loss.item(), acc

def train_until_convergence(model, task_cls, max_steps=1500, lr=3e-4, device='cuda'):
    is_manifold = isinstance(model, Manifold)
    
    # === Speed Optimizations ===
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.benchmark = True
    
    # NOTE: torch.compile is disabled for Windows compatibility (Missing Triton)
    # The gain is marginal for this specific recurrence loop anyway.
    
    if is_manifold:
        print("[*] Training Manifold (RiemannianAdam + O(1) MSE)")
        optimizer = RiemannianAdam(model.parameters(), lr=lr, weight_decay=1e-4, max_norm=10.0)
    else:
        print("[*] Training MicroGPT (AdamW + CrossEntropy)")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, pct_start=0.2)
    model.train()
    
    norm_loss = 1.0
    norm_acc = 0.0
    
    pbar = tqdm(range(max_steps))
    
    for i in pbar:
        # Generate Data (Fixed Length 20)
        task = task_cls(length=20)
        x, y = task.generate_batch(128, device=device)
        
        if is_manifold:
            loss, acc = train_step_manifold(model, optimizer, scheduler, x, y, device)
        else:
            loss, acc = train_step_gpt(model, optimizer, scheduler, x, y, device)
            
        # Update metrics
        norm_loss = 0.95 * norm_loss + 0.05 * loss
        norm_acc = 0.95 * norm_acc + 0.05 * acc
        
        pbar.set_description(f"Loss: {norm_loss:.4f} | Acc: {norm_acc*100:.1f}%")
        
        # Convergence Check
        if norm_loss < 0.005 and i > 200:
            print(f"🚀 Converged at step {i}")
            break
            
    return norm_loss

def evaluate_generalization(model, lengths, device='cuda'):
    model.eval()
    accuracies = []
    vrams = []
    
    print(f"\nEvaluating Generalization...")
    for L in lengths:
        task = ParityTask(length=L)
        x, y = task.generate_batch(200, device=device)
        
        # 1. Measure VRAM (Streaming Inference)
        def forward_streaming():
            with torch.no_grad():  # CRITICAL: Prevent graph accumulation for O(1) memory
                B, Seq = x.shape
                state = None
                last_logits = None
                for t in range(Seq):
                     inp = x[:, t:t+1]
                     # Manifold: inputs, state -> logits, state, christoffels
                     # GPT: inputs -> logits (Standard)
                     if isinstance(model, Manifold):
                         l, state, _ = model(inp, state=state)
                     else:
                         l = model(inp) # Fake streaming for GPT (it recomputes or uses kv cache if impl)
                         # MicroGPT likely doesn't support state passing easily, 
                         # but let's assume standard forward for memory measurement baseline
                         pass 
                     last_logits = l
                return last_logits

        def forward_full():
            if isinstance(model, Manifold):
                l, _, _ = model(x)
            else:
                l = model(x)
            return l

        try:
             # Peak Memory Check
             # For Manifold: We verify O(1) by stepping token-by-token
             if isinstance(model, Manifold):
                 mem = measure_peak_memory(model, forward_streaming)
                 
                 # CRITICAL: Use streaming for accuracy too (not full forward!)
                 # Otherwise we break O(1) for long sequences
                 with torch.no_grad():
                     state = None
                     preds_list = []
                     for t in range(x.shape[1]):
                         inp = x[:, t:t+1]
                         logits, state, _ = model(inp, state=state)
                         pred = (logits[:, 0, 0] > 0.0).long()
                         preds_list.append(pred)
                     preds = torch.stack(preds_list, dim=1)
             else:
                 # For GPT: Standard full attention matrix forward
                 mem = measure_peak_memory(model, forward_full)
                 with torch.no_grad():
                     logits = model(x)
                     preds = logits.argmax(dim=-1)
                 
             vrams.append(mem)
             
             # 2. Check Accuracy
             acc = (preds == y).float().mean().item()
             accuracies.append(acc)
             
             print(f"  L={L}: {acc*100:.1f}% | Mem: {mem:.1f}MB")
             
        except Exception as e:
            print(f"  L={L}: Failed ({e})")
            accuracies.append(0.0)
            vrams.append(-1.0)
        
        # CRITICAL: Clear CUDA cache between tests to prevent accumulation
        torch.cuda.empty_cache()
            
    return accuracies, vrams

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_plotting()
    
    print("🥊 GFN vs Transformer: The 'State Tracking' Rumble (v1.0 Professional)")
    
    # 1. Setup Models
    dim = 128
    
    # Manifold (V1.0 Config: O(1) + Physics)
    manifold = Manifold(
        vocab_size=2, dim=dim, depth=6, heads=4,
        use_scan=False,  # Sequential mode with optimized MLayer
        integrator_type='leapfrog',
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'implicit', 'coord_dim': 16},
            'active_inference': {'enabled': True, 'plasticity': 0.1},
            'fractal': {'enabled': True},
            'singularities': {'enabled': True, 'strength': 5.0} # Black Holes for state trapping
        }
    ).to(device)
    
    # MicroGPT Baseline
    gpt = MicroGPT(vocab_size=2, dim=dim, depth=4, heads=4, max_len=1000).to(device)
    
    # 2. Train
    print("\n--- Training Manifold ---")
    loss_m = train_until_convergence(manifold, ParityTask, max_steps=1000, lr=3e-4, device=device)
    
    # Save Manifold checkpoint
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "manifold_parity_superiority.pt"
    
    torch.save({
        'model_state_dict': manifold.state_dict(),
        'model_config': {
            'vocab_size': 2,
            'dim': dim,
            'depth': 6,
            'heads': 4,
            'integrator_type': 'leapfrog',
            'physics_config': manifold.physics_config
        },
        'final_loss': loss_m,
        'task': 'Parity (Cumulative XOR)'
    }, checkpoint_path)
    print(f"✅ Manifold checkpoint saved to {checkpoint_path}")
    
    print("\n--- Training GPT ---")
    loss_g = train_until_convergence(gpt, ParityTask, max_steps=2000, lr=1e-3, device=device)
    
    # 3. Benchmark
    lengths = [20, 50, 100, 200, 500, 1000, 10000, 100000]
    acc_m, mem_m = evaluate_generalization(manifold, lengths, device)
    acc_g, mem_g = evaluate_generalization(gpt, lengths, device)
    
    # 3.5. Save Metrics to JSON
    out_dir = PROJECT_ROOT / "tests/benchmarks/results/gfn_superiority"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "lengths": lengths,
        "manifold": {
            "accuracy": acc_m,
            "vram_mb": mem_m
        },
        "transformer": {
            "accuracy": acc_g,
            "vram_mb": mem_g
        }
    }
    
    with open(out_dir / "parity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {out_dir}/parity_metrics.json")

    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy
    ax1.plot(lengths, acc_m, 'o-', color='#E76F51', label='Manifold (O(1))', linewidth=2.5)
    ax1.plot(lengths, acc_g, 's--', color='#2A9D8F', label='Transformer (O(N^2))', linewidth=2.5)
    ax1.set_title("Long-Context Generalization (Parity)")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    
    # Memory
    ax2.plot(lengths, mem_m, 'o-', color='#E76F51', label='Manifold', linewidth=2.5)
    ax2.plot(lengths, mem_g, 's--', color='#2A9D8F', label='Transformer', linewidth=2.5)
    ax2.set_title("VRAM Usage Scaling")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.savefig(out_dir / "parity_result.png", dpi=300)
    print(f"\n✅ Result saved to {out_dir}/parity_result.png")

if __name__ == "__main__":
    run_benchmark()
