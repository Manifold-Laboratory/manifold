"""
Professional GFN Superiority Dashboard
======================================
Comprehensive comparison of Manifold GFN vs Transformer on state-tracking tasks.
Visualizes convergence speed, long-context generalization, and VRAM scaling.
"""

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
from gfn.losses import geodesic_regularization

# Import Baselines & Utils
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

class ParityTask:
    """Parity Check (Modulo 2) for state tracking."""
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y = torch.cumsum(x, dim=1) % self.mod
        return x, y

def train_step_manifold(model, optimizer, scheduler, inputs, targets, device):
    optimizer.zero_grad()
    logits, (x_final, v_final), christoffels = model(inputs)
    
    # Implicit Readout Target Alignment
    coord_dim = model.physics_config.get('embedding', {}).get('coord_dim', 16)
    mask = 2**torch.arange(coord_dim).to(device)
    target_bits = (targets.unsqueeze(-1) & mask) > 0
    target_coords = target_bits.float() * 2 - 1
    
    # Regression Loss (strictly for relevant bit)
    loss_mse = nn.MSELoss()(logits[:, :, 0], target_coords[:, :, 0])
    
    # Physics Regularization
    loss_phy = 0.0
    if christoffels:
        loss_phy += geodesic_regularization(None, christoffels, lambda_g=0.001)
        
    total_loss = loss_mse + loss_phy
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    if scheduler: scheduler.step()
    
    pred_bits = (logits[:, :, 0] > 0.0).long()
    acc = (pred_bits == targets).float().mean().item()
    return total_loss.item(), acc

def train_step_gpt(model, optimizer, scheduler, inputs, targets, device):
    optimizer.zero_grad()
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, 2), targets.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    if scheduler: scheduler.step()
    
    preds = logits.argmax(dim=-1)
    acc = (preds == targets).float().mean().item()
    return loss.item(), acc

def train_model(model, max_steps=1000, device='cuda'):
    is_manifold = isinstance(model, Manifold)
    if is_manifold:
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3 if is_manifold else 3e-4, total_steps=max_steps)
    model.train()
    
    history = {"loss": [], "acc": []}
    pbar = tqdm(range(max_steps), desc=f"Training {'Manifold' if is_manifold else 'GPT'}")
    
    for i in pbar:
        task = ParityTask(length=20)
        x, y = task.generate_batch(128, device=device)
        
        loss, acc = train_step_manifold(model, optimizer, scheduler, x, y, device) if is_manifold else \
                    train_step_gpt(model, optimizer, scheduler, x, y, device)
            
        history["loss"].append(loss)
        history["acc"].append(acc)
        if i % 10 == 0:
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc*100:.1f}%")
            
    return history

def evaluate_scaling(model, lengths, device='cuda'):
    model.eval()
    results = {"acc": [], "mem": []}
    
    for L in lengths:
        task = ParityTask(length=L)
        x, y = task.generate_batch(100, device=device)
        
        def run_inf():
            with torch.no_grad():
                if isinstance(model, Manifold):
                    # O(1) Streaming Inference
                    state = None
                    preds_list = []
                    for t in range(x.shape[1]):
                        l, state, _ = model(x[:, t:t+1], state=state)
                        preds_list.append((l[:, 0, 0] > 0.0).long())
                    return torch.stack(preds_list, dim=1)
                else:
                    return model(x).argmax(dim=-1)

        mem = PerformanceStats.measure_peak_memory(model, run_inf)
        preds = run_inf()
        acc = (preds == y).float().mean().item()
        
        results["acc"].append(acc)
        results["mem"].append(mem)
        print(f"  L={L}: Acc={acc*100:.1f}% | Mem={mem:.1f}MB")
        torch.cuda.empty_cache()
            
    return results

def run_superiority_benchmark():
    logger = ResultsLogger("superiority", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models
    dim = 128
    manifold = Manifold(vocab_size=2, dim=dim, depth=6, heads=4, integrator_type='leapfrog').to(device)
    gpt = MicroGPT(vocab_size=2, dim=dim, depth=4, heads=4, max_len=2000).to(device)
    
    # 2. Training Phase
    h_m = train_model(manifold, max_steps=800, device=device)
    h_g = train_model(gpt, max_steps=1200, device=device)
    
    # 3. Scaling Phase
    lengths = [20, 50, 100, 200, 500, 1000, 2000]
    print("\n--- Evaluating Manifold Scaling ---")
    s_m = evaluate_scaling(manifold, lengths, device)
    print("\n--- Evaluating GPT Scaling ---")
    s_g = evaluate_scaling(gpt, lengths, device)
    
    # 4. Dashboard Plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Plot A: Convergence Loss
    axes[0, 0].plot(h_m["loss"], color='#E76F51', label='Manifold (Hamiltonian)', alpha=0.8)
    axes[0, 0].plot(h_g["loss"], color='#264653', label='Transformer (CE)', alpha=0.8)
    axes[0, 0].set_title("Training Convergence (Loss)", fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()

    # Plot B: Accuracy Trend
    axes[0, 1].plot(np.convolve(h_m["acc"], np.ones(10)/10, mode='valid'), color='#E76F51', label='Manifold')
    axes[0, 1].plot(np.convolve(h_g["acc"], np.ones(10)/10, mode='valid'), color='#264653', label='Transformer')
    axes[0, 1].set_title("Learning Dynamics (Smoothed Accuracy)", fontweight='bold')
    axes[0, 1].legend()

    # Plot C: Long-Context Generalization
    axes[1, 0].plot(lengths, s_m["acc"], 'o-', color='#2A9D8F', label='Manifold (O(1))', linewidth=3)
    axes[1, 0].plot(lengths, s_g["acc"], 's--', color='#E9C46A', label='Transformer (O(N²))', linewidth=3)
    axes[1, 0].set_title("Out-of-Distribution Generalization", fontweight='bold')
    axes[1, 0].set_xlabel("Sequence Length")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    # Plot D: VRAM Scaling
    axes[1, 1].plot(lengths, s_m["mem"], 'o-', color='#2A9D8F', label='Manifold', linewidth=3)
    axes[1, 1].plot(lengths, s_g["mem"], 's--', color='#E9C46A', label='Transformer', linewidth=3)
    axes[1, 1].set_title("Memory Scaling (VRAM)", fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel("Sequence Length")
    axes[1, 1].set_ylabel("Peak MB")
    axes[1, 1].legend()

    fig.suptitle("Manifold GFN vs Transformer: Superiority Dashboard", fontsize=22, fontweight='bold', y=0.96)
    logger.save_plot(fig, "gfn_superiority_dashboard.png")
    
    # Save Metrics
    logger.save_json({
        "hyperparameters": {"dim": dim, "lengths": lengths},
        "manifold": {"final_acc": h_m["acc"][-1], "ood_scaling": s_m},
        "transformer": {"final_acc": h_g["acc"][-1], "ood_scaling": s_g}
    })

if __name__ == "__main__":
    run_superiority_benchmark()
