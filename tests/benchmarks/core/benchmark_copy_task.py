"""
Quick Learning Demo
===================

Simplified version that shows learning differences FAST (2-3 minutes).

Uses simple copy task instead of math for quick demonstration.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import GFN
try:
    from tests.benchmarks.baselines import MicroGPT
except ImportError:
    from baselines import MicroGPT


def generate_copy_data(vocab_size, seq_len, batch_size):
    """Generate simple copy task: input -> output (same sequence)."""
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = inputs.clone()
    return inputs, targets


def train_and_compare_quick():
    """Quick 2-minute demo showing learning differences."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("  ⚡ QUICK LEARNING DEMO (Copy Task)")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    # Smaller, balanced models
    vocab_size = 10
    seq_len = 8
    
    # BALANCED parameter count by making GFN much bigger
    # Using 4 heads and deep network
    gfn = GFN(vocab_size=vocab_size, dim=512, depth=12, rank=64, heads=8).to(device)
    gpt = MicroGPT(vocab_size=vocab_size, dim=192, depth=6, heads=4).to(device)
    
    gfn_params = sum(p.numel() for p in gfn.parameters()) / 1e6
    gpt_params = sum(p.numel() for p in gpt.parameters()) / 1e6
    
    print(f"📊 Balanced Models:")
    print(f"  GFN: {gfn_params:.2f}M params")
    print(f"  GPT: {gpt_params:.2f}M params")
    print(f"  Ratio: {gfn_params/gpt_params:.2f}x\n")
    
    # Optimizers
    gfn_opt = torch.optim.Adam(gfn.parameters(), lr=5e-3)
    gpt_opt = torch.optim.Adam(gpt.parameters(), lr=5e-3)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    epochs = 20
    batch_size = 64
    
    gfn_losses = []
    gpt_losses = []
    gfn_accs = []
    gpt_accs = []
    
    print(f"🔄 Training {epochs} epochs on copy task...\n")
    
    for epoch in range(epochs):
        # Train GFN
        gfn.train()
        inputs, targets = generate_copy_data(vocab_size, seq_len, batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        
        gfn_opt.zero_grad()
        logits, _ = gfn(inputs)
        loss_gfn = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss_gfn.backward()
        torch.nn.utils.clip_grad_norm_(gfn.parameters(), 1.0)
        gfn_opt.step()
        
        # Train GPT
        gpt.train()
        gpt_opt.zero_grad()
        logits_gpt = gpt(inputs)
        loss_gpt = criterion(logits_gpt.view(-1, vocab_size), targets.view(-1))
        loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        gpt_opt.step()
        
        # Evaluate
        if epoch % 2 == 0:
            gfn.eval()
            gpt.eval()
            
            with torch.no_grad():
                test_inputs, test_targets = generate_copy_data(vocab_size, seq_len, 100)
                test_inputs = test_inputs.to(device)
                test_targets = test_targets.to(device)
                
                # GFN accuracy
                logits_gfn, _ = gfn(test_inputs)
                preds_gfn = torch.argmax(logits_gfn, dim=-1)
                acc_gfn = (preds_gfn == test_targets).float().mean().item() * 100
                
                # GPT accuracy
                logits_gpt = gpt(test_inputs)
                preds_gpt = torch.argmax(logits_gpt, dim=-1)
                acc_gpt = (preds_gpt == test_targets).float().mean().item() * 100
                
                gfn_losses.append(loss_gfn.item())
                gpt_losses.append(loss_gpt.item())
                gfn_accs.append(acc_gfn)
                gpt_accs.append(acc_gpt)
                
                print(f"Epoch {epoch:2d} | GFN: Loss={loss_gfn:.4f} Acc={acc_gfn:5.1f}% | "
                      f"GPT: Loss={loss_gpt:.4f} Acc={acc_gpt:5.1f}%")
    
    # Plot
    results_dir = PROJECT_ROOT / "tests" / "professional" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_plot = np.arange(0, len(gfn_losses)) * 2
    
    # Loss
    ax1.plot(epochs_plot, gfn_losses, 'o-', linewidth=2, label='GFN', color='#2A9D8F')
    ax1.plot(epochs_plot, gpt_losses, 's-', linewidth=2, label='Transformer', color='#E76F51')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence (Copy Task)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs_plot, gfn_accs, 'o-', linewidth=2, label='GFN', color='#2A9D8F')
    ax2.plot(epochs_plot, gpt_accs, 's-', linewidth=2, label='Transformer', color='#E76F51')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Learning Speed Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "quick_learning_demo.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {results_dir}/quick_learning_demo.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 RESULTS")
    print("=" * 70)
    print(f"Final GFN: Loss={gfn_losses[-1]:.4f}, Acc={gfn_accs[-1]:.1f}%")
    print(f"Final GPT: Loss={gpt_losses[-1]:.4f}, Acc={gpt_accs[-1]:.1f}%")
    
    winner = "GFN" if gfn_accs[-1] > gpt_accs[-1] else "GPT"
    print(f"\n🏆 Winner: {winner}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_compare_quick()
