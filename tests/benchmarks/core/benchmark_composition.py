"""
Function Composition Benchmark
==============================

THE ULTIMATE TEST: Can GFN learn to COMPOSE functions?

This tests the core hypothesis of GFN:
- Transformers: Memorize input→output mappings
- GFN: Learn continuous FLOWS that can be composed

Task:
  Train: f(x)=x+2, g(x)=x*3, h(x)=x-1
  Test:  f(g(h(x))), g(f(x)), h(g(f(x))), etc.

If GFN truly models "geodesic flows", it should compose functions
it learned separately WITHOUT seeing the compositions during training.

This is impossible for Transformers without massive data.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import GFN
try:
    from tests.benchmarks.baselines import MicroGPT
except ImportError:
    from baselines import MicroGPT


class FunctionCompositionDataset:
    """Dataset of function applications and compositions."""
    
    def __init__(self, train_mode=True):
        """
        Functions:
          f(x) = x + 2
          g(x) = x * 3
          h(x) = x - 1
        
        If train_mode=True: Only single applications (f(x), g(x), h(x))
        If train_mode=False: Only compositions (f(g(x)), g(h(f(x))), etc.)
        """
        self.train_mode = train_mode
        
        # Vocabulary: 0-9, +, -, *, =, f, g, h, (, ), <PAD>, <EOS>
        chars = [str(i) for i in range(10)] + ['+', '-', '*', '=', 'f', 'g', 'h', '(', ')', '<PAD>', '<EOS>']
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
        
        # Function definitions
        self.funcs = {
            'f': lambda x: x + 2,
            'g': lambda x: x * 3,
            'h': lambda x: x - 1,
        }
    
    def apply_composition(self, x, composition):
        """Apply composition like 'fgh' means f(g(h(x)))."""
        result = x
        # Apply from right to left (like math notation)
        for func_name in reversed(composition):
            result = self.funcs[func_name](result)
        return result
    
    def generate_problem(self):
        """Generate a single problem."""
        if self.train_mode:
            # Training: Only single function applications
            func_name = np.random.choice(['f', 'g', 'h'])
            x = np.random.randint(0, 30)
            result = self.funcs[func_name](x)
            problem = f"{func_name}({x})={result}"
        else:
            # Test: Compositions
            # Random composition of 2-3 functions
            length = np.random.choice([2, 3])
            composition = ''.join(np.random.choice(['f', 'g', 'h'], size=length))
            x = np.random.randint(0, 10)  # Smaller to avoid overflow
            result = self.apply_composition(x, composition)
            
            # Format: f(g(5))=result
            nested = f"{composition[0]}("
            for i in range(1, len(composition)):
                nested += f"{composition[i]}("
            nested += str(x)
            nested += ")" * len(composition)
            problem = f"{nested}={result}"
        
        return problem
    
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.char_to_id[c] for c in text]
    
    def decode(self, ids):
        """Convert token IDs to text."""
        return ''.join([self.id_to_char.get(i, '?') for i in ids if i not in [self.char_to_id['<PAD>'], self.char_to_id['<EOS>']]])


def evaluate_composition(model, dataset, num_samples=100, model_name='GFN', device='cuda'):
    """Evaluate accuracy on composition tasks."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            problem = dataset.generate_problem()
            parts = problem.split('=')
            prompt = parts[0] + '='
            target = parts[1]
            
            ids = dataset.encode(prompt)
            input_seq = torch.tensor([ids]).to(device)
            
            # Generate
            if model_name == 'GFN':
                logits, state = model(input_seq)
                generated = list(ids)
                
                curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated.append(curr_token.item())
                
                for _ in range(10):  # Max 10 digits for result
                    logits, state = model(curr_token, state=state)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    tok_id = next_token.item()
                    if tok_id == dataset.char_to_id['<EOS>']:
                        break
                    generated.append(tok_id)
                    curr_token = next_token.unsqueeze(0)
            else:  # GPT
                generated = list(ids)
                for _ in range(10):
                    inp = torch.tensor([generated]).to(device)
                    logits = model(inp)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    tok_id = next_token.item()
                    if tok_id == dataset.char_to_id['<EOS>']:
                        break
                    generated.append(tok_id)
            
            # Check correctness
            pred = dataset.decode(generated).split('=')[-1].strip()
            if pred == target.strip():
                correct += 1
            total += 1
    
    return (correct / total) * 100


def run_composition_benchmark():
    """Main benchmark runner."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("  🧮 FUNCTION COMPOSITION BENCHMARK")
    print("="*70)
    print("\n📘 Hypothesis:")
    print("  GFN learns FLOWS → Can compose functions")
    print("  GPT memorizes PATTERNS → Cannot generalize to new compositions")
    
    # Create datasets
    train_dataset = FunctionCompositionDataset(train_mode=True)
    test_dataset = FunctionCompositionDataset(train_mode=False)
    
    # Create models
    vocab_size = train_dataset.vocab_size
    gfn = GFN(vocab_size=vocab_size, dim=256, depth=6, rank=16).to(device)
    gpt = MicroGPT(vocab_size=vocab_size, dim=128, depth=6, heads=4).to(device)
    
    print(f"\n📊 Models:")
    print(f"  GFN: {sum(p.numel() for p in gfn.parameters())/1e6:.2f}M params")
    print(f"  GPT: {sum(p.numel() for p in gpt.parameters())/1e6:.2f}M params")
    
    # Training
    print(f"\n🔄 Training on SIMPLE functions (f(x), g(x), h(x))...")
    
    gfn_optimizer = torch.optim.Adam(gfn.parameters(), lr=1e-3)
    gpt_optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    batch_size = 32
    
    for epoch in range(epochs):
        gfn.train()
        gpt.train()
        
        # Generate batch
        batch = [train_dataset.generate_problem() for _ in range(batch_size)]
        inputs = []
        targets = []
        
        for problem in batch:
            ids = train_dataset.encode(problem + '<EOS>')
            inputs.append(ids[:-1])
            targets.append(ids[1:])
        
        # Pad
        max_len = max(len(seq) for seq in inputs)
        padded_inputs = [seq + [train_dataset.char_to_id['<PAD>']] * (max_len - len(seq)) for seq in inputs]
        padded_targets = [seq + [-100] * (max_len - len(seq)) for seq in targets]
        
        inputs_tensor = torch.tensor(padded_inputs).to(device)
        targets_tensor = torch.tensor(padded_targets).to(device)
        
        # Train GFN
        gfn_optimizer.zero_grad()
        gfn_logits, _ = gfn(inputs_tensor)
        gfn_loss = criterion(gfn_logits.reshape(-1, vocab_size), targets_tensor.reshape(-1))
        gfn_loss.backward()
        gfn_optimizer.step()
        
        # Train GPT
        gpt_optimizer.zero_grad()
        gpt_logits = gpt(inputs_tensor)
        gpt_loss = criterion(gpt_logits.reshape(-1, vocab_size), targets_tensor.reshape(-1))
        gpt_loss.backward()
        gpt_optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: GFN Loss={gfn_loss.item():.4f}, GPT Loss={gpt_loss.item():.4f}")
    
    # Evaluation
    print(f"\n🧪 Testing on COMPOSITIONS (f(g(x)), g(h(f(x))), etc.)...")
    
    gfn_acc = evaluate_composition(gfn, test_dataset, num_samples=200, model_name='GFN', device=device)
    gpt_acc = evaluate_composition(gpt, test_dataset, num_samples=200, model_name='GPT', device=device)
    
    # Also test on training distribution
    gfn_train_acc = evaluate_composition(gfn, train_dataset, num_samples=200, model_name='GFN', device=device)
    gpt_train_acc = evaluate_composition(gpt, train_dataset, num_samples=200, model_name='GPT', device=device)
    
    # Results
    print("\n" + "="*70)
    print("  📊 RESULTS")
    print("="*70)
    
    print(f"\n✅ Training Distribution (Simple functions):")
    print(f"  GFN: {gfn_train_acc:.1f}%")
    print(f"  GPT: {gpt_train_acc:.1f}%")
    
    print(f"\n🎯 Test Distribution (Compositions - UNSEEN):")
    print(f"  GFN: {gfn_acc:.1f}% ⭐")
    print(f"  GPT: {gpt_acc:.1f}%")
    
    print(f"\n💡 Composition Gap (How well they generalize):")
    gfn_gap = gfn_train_acc - gfn_acc
    gpt_gap = gpt_train_acc - gpt_acc
    print(f"  GFN: {gfn_gap:.1f}% drop")
    print(f"  GPT: {gpt_gap:.1f}% drop")
    
    if gfn_acc > gpt_acc:
        print(f"\n🏆 WINNER: GFN by {gfn_acc - gpt_acc:.1f}%")
        print("   GFN successfully learned COMPOSITIONAL structure! 🎉")
    else:
        print(f"\n🏆 WINNER: GPT by {gpt_acc - gfn_acc:.1f}%")
        print("   (Hypothesis needs refinement)")
    
    print("="*70)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['GFN', 'Transformer']
    train_accs = [gfn_train_acc, gpt_train_acc]
    test_accs = [gfn_acc, gpt_acc]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Simple Functions (Train)', color='#2A9D8F', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Compositions (Test)', color='#E76F51', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('🧮 Function Composition: The GFN Advantage', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    results_dir = PROJECT_ROOT / "tests" / "professional" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "function_composition_benchmark.png", dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {results_dir / 'function_composition_benchmark.png'}")
    plt.close()


if __name__ == "__main__":
    run_composition_benchmark()
