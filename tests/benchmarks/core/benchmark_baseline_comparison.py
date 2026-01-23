"""
Professional Baseline Comparison Benchmark
==========================================

Objective:
- Compare Manifold with standard RNN architectures (GRU, LSTM).
- Evaluate learning dynamics and final convergence on core sequence tasks.
- Standardized reporting with publication-quality metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger

# ============== BASELINES ==============

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.gru = nn.GRU(dim, dim, num_layers=depth, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.gru(x)
        return self.head(x)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=depth, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.head(x)

# ============== DATA GENERATION ==============

def create_parity_task(batch_size, seq_len=50):
    data = torch.randint(0, 2, (batch_size, seq_len))
    targets = data.cumsum(dim=1) % 2
    return data, targets

def run_baseline_comparison():
    logger = ResultsLogger("baseline_comparison", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Config
    vocab_size = 16 # for embedding test, though parity uses 2
    dim = 128
    depth = 2
    batch_size = 32
    num_steps = 500
    
    models_to_test = {
        "Manifold": Manifold(vocab_size, dim, depth, heads=4, integrator_type='yoshida'),
        "GRU": SimpleGRU(vocab_size, dim, depth),
        "LSTM": SimpleLSTM(vocab_size, dim, depth)
    }

    results = []
    
    print(f"🚀 Starting Baseline Comparison on {device}...")

    for name, model in models_to_test.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        print(f"  Training {name}...")
        losses = []
        accuracies = []
        
        start_time = time.time()
        for step in range(num_steps):
            inputs, targets = create_parity_task(batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            out = model(inputs)
            if isinstance(out, tuple): out = out[0]
            
            # Predict
            loss = criterion(out.view(-1, out.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 50 == 0:
                with torch.no_grad():
                    preds = out.argmax(dim=-1)
                    acc = (preds == targets).float().mean().item()
                    accuracies.append(acc)
                    print(f"    Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc*100:5.1f}%")

        elapsed = time.time() - start_time
        final_acc = accuracies[-1] * 100
        
        results.append({
            "Model": name,
            "Final Accuracy (%)": final_acc,
            "Training Time (s)": elapsed,
            "Params": sum(p.numel() for p in model.parameters())
        })

    # 2. Saving and Plotting
    logger.save_json(results)
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Final Accuracy (%)", palette="viridis", ax=ax)
    ax.set_title("Parity Task Convergence @ 500 Steps")
    ax.set_ylim(45, 105) # Random is 50
    
    # Add labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                    
    plt.tight_layout()
    logger.save_plot(fig, "baseline_accuracy.png")
    
    return df

if __name__ == "__main__":
    run_baseline_comparison()
