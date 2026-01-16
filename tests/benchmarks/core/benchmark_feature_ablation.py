"""
Feature Ablation Benchmark
==========================
Tests whether each physics feature actually improves learning.

Answers: "Does each feature actually help?"

Protocol:
1. Train on synthetic task (associative recall)
2. Same training budget for all configurations
3. Measure final loss and accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold


def create_associative_recall_data(batch_size, num_pairs=5, vocab_size=32):
    """
    Create associative recall task: A->B, C->D, ... then query A->?
    
    Format: [A, B, C, D, E, F, SEP, A, ?]
    Target: B
    """
    sep_token = vocab_size - 1
    
    sequences = []
    targets = []
    
    for _ in range(batch_size):
        # Generate random pairs
        keys = torch.randint(0, vocab_size - 1, (num_pairs,))
        values = torch.randint(0, vocab_size - 1, (num_pairs,))
        
        # Build sequence: k1, v1, k2, v2, ..., SEP, query_key
        seq = []
        for k, v in zip(keys, values):
            seq.extend([k.item(), v.item()])
        seq.append(sep_token)
        
        # Query a random pair
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        seq.append(keys[query_idx].item())
        
        sequences.append(seq)
        targets.append(values[query_idx].item())
    
    return torch.tensor(sequences), torch.tensor(targets)


def get_ablation_configs():
    """Configurations to ablate."""
    return {
        "baseline": None,
        "active_inference": {
            'active_inference': {'enabled': True}
        },
        "active+singularities": {
            'active_inference': {
                'enabled': True,
                'singularities': {'enabled': True, 'strength': 5.0}
            }
        },
        "active+symmetries": {
            'active_inference': {'enabled': True},
            'symmetries': {'enabled': True, 'isomeric_groups': [[0, 1], [2, 3]]}
        },
        "all_features": {
            'active_inference': {
                'enabled': True,
                'reactive_curvature': {'enabled': True, 'plasticity': 0.1},
                'singularities': {'enabled': True, 'strength': 5.0},
            },
            'symmetries': {'enabled': True, 'isomeric_groups': [[0, 1], [2, 3]]},
            'stability': {'curvature_clamp': 10.0}
        }
    }


def train_and_evaluate(config_name, physics_config, device, 
                       num_steps=200, batch_size=32, lr=1e-3):
    """Train a model and return final metrics."""
    vocab_size = 32
    
    model = Manifold(
        vocab_size=vocab_size,
        dim=128,
        depth=4,
        heads=4,
        integrator_type='heun',
        physics_config=physics_config
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    model.train()
    for step in range(num_steps):
        inputs, targets = create_associative_recall_data(batch_size, num_pairs=5, vocab_size=vocab_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits, _, _ = model(inputs)
        
        # Predict at last position
        pred_logits = logits[:, -1, :]
        loss = criterion(pred_logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 100
    
    with torch.no_grad():
        for _ in range(total):
            inputs, targets = create_associative_recall_data(1, num_pairs=5, vocab_size=vocab_size)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits, _, _ = model(inputs)
            pred = logits[0, -1, :].argmax()
            
            if pred == targets[0]:
                correct += 1
    
    accuracy = correct / total * 100
    final_loss = sum(losses[-20:]) / 20  # Average of last 20 steps
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'final_loss': round(final_loss, 4),
        'accuracy': round(accuracy, 1),
        'loss_curve': losses
    }


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Feature Ablation Benchmark on {device}")
    print("="*60)
    print("Task: Associative Recall (A->B, C->D, ... query A->?)")
    print("Training: 200 steps per config")
    print("="*60)
    
    configs = get_ablation_configs()
    results = {}
    
    for name, physics_config in configs.items():
        print(f"\n[*] Training: {name}...")
        try:
            metrics = train_and_evaluate(name, physics_config, device)
            results[name] = {
                'final_loss': metrics['final_loss'],
                'accuracy': metrics['accuracy']
            }
            print(f"   Final Loss: {metrics['final_loss']:.4f} | Accuracy: {metrics['accuracy']:.1f}%")
        except Exception as e:
            print(f"   ERROR: {e}")
            results[name] = {'error': str(e)}
    
    # Save results
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/validation"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate chart
    valid_names = [k for k, v in results.items() if 'error' not in v]
    accuracies = [results[k]['accuracy'] for k in valid_names]
    losses = [results[k]['final_loss'] for k in valid_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['gray' if n == 'baseline' else 'steelblue' for n in valid_names]
    
    bars1 = ax1.barh(valid_names, accuracies, color=colors)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Feature Impact on Accuracy')
    ax1.axvline(x=100/32, color='red', linestyle='--', label='Random (3.1%)')
    ax1.bar_label(bars1, fmt='%.1f%%')
    ax1.legend()
    
    bars2 = ax2.barh(valid_names, losses, color=colors)
    ax2.set_xlabel('Final Loss')
    ax2.set_title('Feature Impact on Loss')
    ax2.bar_label(bars2, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(res_dir / "ablation_chart.png", dpi=150, bbox_inches='tight')
    
    print(f"\n[*] Results saved to {res_dir}")
    
    # Summary
    if 'baseline' in results and 'all_features' in results:
        base_acc = results['baseline']['accuracy']
        full_acc = results['all_features']['accuracy']
        improvement = full_acc - base_acc
        print(f"\n[*] SUMMARY:")
        print(f"   Baseline accuracy: {base_acc:.1f}%")
        print(f"   All features accuracy: {full_acc:.1f}%")
        print(f"   Improvement: {improvement:+.1f}%")
        if improvement > 0:
            print(f"   → Features HELP learning [*]")
        else:
            print(f"   → Features don't help (or hurt) [*][*]")
    
    return results


if __name__ == "__main__":
    run_benchmark()
