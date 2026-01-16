"""
Baseline Comparison Benchmark
=============================
Compares Manifold against simple baselines on synthetic tasks.

Answers: "Does Manifold beat standard architectures?"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold


# ============== BASELINE MODELS ==============

class SimpleGRU(nn.Module):
    """Simple GRU baseline with similar param count."""
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
    """Simple LSTM baseline with similar param count."""
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

def create_copy_task(batch_size, seq_len=20, vocab_size=16):
    """Copy task: input -> copy input."""
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    return data, data.clone()


def create_reverse_task(batch_size, seq_len=20, vocab_size=16):
    """Reverse task: input -> reversed input."""
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    return data, data.flip(dims=[1])


def create_parity_task(batch_size, seq_len=50, vocab_size=2):
    """Parity task: predict XOR of all previous tokens."""
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    parity = data.cumsum(dim=1) % 2
    return data, parity


# ============== TRAINING ==============

def train_model(model, task_fn, device, num_steps=300, batch_size=32, lr=1e-3):
    """Train a model on a task."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    model.train()
    
    for step in range(num_steps):
        inputs, targets = task_fn(batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Handle different model return types
        out = model(inputs)
        if isinstance(out, tuple):
            logits = out[0]  # Manifold returns (logits, state, christoffels)
        else:
            logits = out
        
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses


def evaluate_accuracy(model, task_fn, device, num_samples=100):
    """Evaluate accuracy on a task."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            inputs, targets = task_fn(1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            out = model(inputs)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    
    return correct / total * 100


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*]  Baseline Comparison Benchmark on {device}")
    print("="*60)
    
    vocab_size = 16
    dim = 128
    depth = 4
    
    # Tasks to test
    tasks = {
        'copy': lambda bs: create_copy_task(bs, seq_len=20, vocab_size=vocab_size),
        'reverse': lambda bs: create_reverse_task(bs, seq_len=20, vocab_size=vocab_size),
        'parity': lambda bs: create_parity_task(bs, seq_len=50, vocab_size=2),
    }
    
    results = {}
    
    for task_name, task_fn in tasks.items():
        print(f"\n[*] Task: {task_name.upper()}")
        print("-" * 40)
        
        task_results = {}
        
        # Create models
        models = {
            'Manifold': Manifold(vocab_size, dim=dim, depth=depth, heads=4, integrator_type='heun'),
            'GRU': SimpleGRU(vocab_size, dim, depth),
            'LSTM': SimpleLSTM(vocab_size, dim, depth),
        }
        
        for model_name, model in models.items():
            model = model.to(device)
            params = sum(p.numel() for p in model.parameters())
            
            print(f"  Training {model_name} ({params/1e6:.2f}M params)...", end=" ")
            
            try:
                start = time.time()
                losses = train_model(model, task_fn, device, num_steps=300)
                train_time = time.time() - start
                
                accuracy = evaluate_accuracy(model, task_fn, device)
                final_loss = sum(losses[-20:]) / 20
                
                print(f"Acc: {accuracy:.1f}% | Loss: {final_loss:.3f} | Time: {train_time:.1f}s")
                
                task_results[model_name] = {
                    'accuracy': round(accuracy, 1),
                    'final_loss': round(final_loss, 4),
                    'train_time': round(train_time, 1),
                    'params': params
                }
            except Exception as e:
                print(f"ERROR: {e}")
                task_results[model_name] = {'error': str(e)}
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        results[task_name] = task_results
    
    # Save results
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/validation"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / "baseline_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate chart
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    
    for idx, (task_name, task_results) in enumerate(results.items()):
        ax = axes[idx] if len(tasks) > 1 else axes
        
        valid_models = [m for m, r in task_results.items() if 'error' not in r]
        accuracies = [task_results[m]['accuracy'] for m in valid_models]
        
        colors = ['steelblue' if m == 'Manifold' else 'gray' for m in valid_models]
        bars = ax.bar(valid_models, accuracies, color=colors)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{task_name.upper()} Task')
        ax.set_ylim(0, 105)
        ax.bar_label(bars, fmt='%.1f%%')
    
    plt.tight_layout()
    plt.savefig(res_dir / "baseline_chart.png", dpi=150, bbox_inches='tight')
    
    print(f"\n[*] Results saved to {res_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("[*] SUMMARY")
    print("="*60)
    
    manifold_wins = 0
    total_tasks = 0
    
    for task_name, task_results in results.items():
        if 'Manifold' in task_results and 'error' not in task_results['Manifold']:
            manifold_acc = task_results['Manifold']['accuracy']
            best_baseline = max(
                (r['accuracy'] for m, r in task_results.items() 
                 if m != 'Manifold' and 'error' not in r),
                default=0
            )
            
            if manifold_acc >= best_baseline:
                manifold_wins += 1
                print(f"  {task_name}: Manifold WINS ({manifold_acc}% vs {best_baseline}%) [*]")
            else:
                print(f"  {task_name}: Manifold LOSES ({manifold_acc}% vs {best_baseline}%) [*]")
            
            total_tasks += 1
    
    print(f"\n  Manifold wins {manifold_wins}/{total_tasks} tasks")
    
    return results


if __name__ == "__main__":
    run_benchmark()
