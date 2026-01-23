import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold

class ArithmeticTask:
    """
    Simple arithmetic operations: a + b, a - b, a * b
    
    Format:
    Input: "3 + 5 ="
    Output: "8"
    
    Vocab: 0-9 (digits), +, -, *, =, <pad>
    """
    def __init__(self, max_num=20, operation='add'):
        self.max_num = max_num
        self.operation = operation
        
        # Vocab: 0-9, +, -, *, =, <pad>
        self.vocab_size = 15
        self.digit_offset = 0  # 0-9
        self.op_map = {'+': 10, '-': 11, '*': 12}
        self.eq_token = 13
        self.pad_token = 14
        
    def generate_batch(self, batch_size, device='cpu'):
        """
        Generate batch of arithmetic problems.
        
        Returns:
            inputs: [batch, 5] - "a op b = <pad>"
            targets: [batch, 5] - "<pad> <pad> <pad> <pad> result"
        """
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            a = np.random.randint(0, self.max_num)
            b = np.random.randint(0, self.max_num)
            
            if self.operation == 'add':
                result = a + b
                op = self.op_map['+']
            elif self.operation == 'sub':
                result = max(0, a - b)  # No negatives
                op = self.op_map['-']
            else:  # mul
                result = a * b
                op = self.op_map['*']
            
            # Clamp result to single digit
            result = min(result, 9)
            
            # Input: [a, op, b, =, pad]
            inp = [a, op, b, self.eq_token, self.pad_token]
            
            # Target: [pad, pad, pad, pad, result]
            tgt = [self.pad_token, self.pad_token, self.pad_token, self.pad_token, result]
            
            inputs.append(inp)
            targets.append(tgt)
        
        return (torch.tensor(inputs, device=device), 
                torch.tensor(targets, device=device))

def train_arithmetic(model, task, max_steps=3000, lr=3e-4, device='cuda'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, pct_start=0.2)
    
    is_binary = model.physics_config.get('readout', {}).get('type') == 'binary'
    
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    best_acc = 0.0
    
    pbar = tqdm(range(max_steps), desc=f"Training {model.__class__.__name__}")
    
    for i in pbar:
        model.train()
        x, y = task.generate_batch(64, device=device)
        
        optimizer.zero_grad()
        logits, _, _ = model(x)
        
        if is_binary:
            # Binary mode
            coord_dim = model.physics_config.get('embedding', {}).get('coord_dim', 16)
            mask = 2**torch.arange(coord_dim).to(device)
            target_bits = (y.unsqueeze(-1) & mask) > 0
            target_bits = target_bits.float()
            
            # Only train on last position
            loss = criterion(logits[:, -1, :], target_bits[:, -1, :])
        else:
            # Standard mode
            loss = criterion(logits[:, -1, :], y[:, -1])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        if is_binary:
            preds = (logits[:, -1, 0] > 0.0).long()
        else:
            preds = logits[:, -1, :].argmax(dim=-1)
        
        acc = (preds == y[:, -1]).float().mean().item()
        best_acc = max(best_acc, acc)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{acc*100:.1f}%",
            'best': f"{best_acc*100:.1f}%",
        })
        
        loss_history.append(loss.item())
        
        if acc > 0.95 and i > 200:
            print(f"\n✅ Converged at step {i} (Acc: {acc*100:.1f}%)")
            break
    
    return loss_history

def run_arithmetic_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧮 Arithmetic Task Benchmark")
    print(f"Device: {device}\n")
    
    # Create task
    task = ArithmeticTask(max_num=10, operation='add')
    
    # Create model
    dim = 128
    depth = 6
    heads = 4
    vocab = 15  # 0-9, +, -, *, =, <pad>
    
    model = Manifold(
        vocab_size=vocab, dim=dim, depth=depth, heads=heads,
        use_scan=False,
        integrator_type='leapfrog',
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'binary'},
            'active_inference': {'enabled': True, 'reactive_curvature': {'enabled': True, 'plasticity': 0.05}},
            'hyper_curvature': {'enabled': True},
            'stability': {'base_dt': 0.3, 'damping': 0.05, 'residual_scale': 0.5}
        }
    ).to(device)
    
    print(f"Model: {dim}d, {depth} layers, {heads} heads")
    print(f"Task: a + b (0-9)\n")
    
    # Train
    loss_history = train_arithmetic(model, task, max_steps=3000, lr=3e-4, device=device)
    
    print("\n✅ Training complete!")

if __name__ == '__main__':
    run_arithmetic_benchmark()
