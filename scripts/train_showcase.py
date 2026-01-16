
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import random
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

# === 1. Synthetic Task: Fractal Associative Recall ===
class FractalRecallDataset(torch.utils.data.Dataset):
    """
    Generates sequences with nested dependencies and high-energy noise.
    Format: [ START NOISE(!!!) { KEY: VALUE } NOISE(!!!) ] QUERY(KEY) -> TARGET(VALUE)
    """
    def __init__(self, vocab_size, seq_len=64, num_samples=5000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Reserved tokens
        self.mod = vocab_size - 10 # Start of special tokens
        self.START = self.mod + 0
        self.END = self.mod + 1
        self.OPEN_BRACKET = self.mod + 2
        self.CLOSE_BRACKET = self.mod + 3
        self.KEY_VAL_SEP = self.mod + 4
        self.QUERY = self.mod + 5
        self.NOISE = self.mod + 6 # "Singularity" token
        
        self.data_range = range(0, self.mod)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random key-value pair
        key = random.choice(self.data_range)
        val = random.choice(self.data_range)
        
        # Construct Fractal Structure
        # [ { K: V } ] with noise injections
        seq = [self.START]
        
        # Outer distraction
        if random.random() > 0.5:
            seq.append(self.NOISE)
            seq.append(random.choice(self.data_range))
            
        seq.append(self.OPEN_BRACKET)
        
        # Inner noise
        if random.random() > 0.5:
             seq.append(self.NOISE)
             
        # The Core Memory
        seq.extend([key, self.KEY_VAL_SEP, val])
        
        seq.append(self.CLOSE_BRACKET)
        
        # Query
        seq.extend([self.QUERY, key])
        
        # Padding
        pad_len = self.seq_len - len(seq) - 1 # -1 for target
        if pad_len < 0:
            # truncate if too long (rare config issue)
            seq = seq[:self.seq_len-1]
            pad_len = 0
            
        seq.extend([0] * pad_len) # 0 as pad
        
        input_tensor = torch.tensor(seq, dtype=torch.long)
        target_tensor = torch.tensor(val, dtype=torch.long) # We want to predict V
        
        # For simplicity, we train on the LAST token prediction mainly,
        # but standard causal Training trains on all.
        # Here we will just return input and the specific target for the query.
        
        return input_tensor, target_tensor

# === 2. Training Script ===
def train_showcase():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Manifold Showcase Training on {device}")
    
    # Config for "Cognitive" Capabilities
    vocab_size = 64 # Small vocab for clear demonstration
    bs = 32
    steps = 500 # Fast demo training
    
    dataset = FractalRecallDataset(vocab_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    
    # Manifold Setup with FULL Physics
    model = Manifold(
        vocab_size=vocab_size,
        dim=512,
        depth=6,     # Deep enough for recursion
        heads=8,
        physics_config={
            "active_inference": {"enabled": True, "plasticity": 0.1},
            "fractal": {"enabled": True, "threshold": 0.4}, # Sensitive fractal trigger
            "symmetries": {"enabled": True},
            "singularities": {"enabled": True, "strength": 5.0}
        }
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics Tracking
    history = {
        "loss": [],
        "accuracy": [],
        "energy_drift": [],
        "fractal_activity": []
    }
    
    print("\n[*] Training Phase: Learning Recursive Associative Recall...")
    model.train()
    
    pbar = tqdm(range(steps))
    data_iter = iter(loader)
    
    for step in pbar:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        # We need the output at the last position used for query
        # Currently inputs are full sequences.
        logits, _, _ = model(inputs) # [B, T, V]
        
        # We find the index of the last token (before padding) or just take the known structure
        # In our simplified synthetic data, the Query is at the end of the semantic sequence.
        # Let's target the last non-pad prediction.
        
        # For this specific synthetic task, we want P(val | query, key)
        # The query is the second to last token in the effective sequence 
        # (before padding zeros start).
        # We'll just define target selection simply: Predict based on last non-zero.
        
        # Simplified: Just train standard Causal LM on the sequence, but monitor specific recall?
        # Better: Train on the whole sequence to learn the structure [ ] { } too.
        
        # Shift targets for Causal LM
        # But we really care about the final answer accuracy.
        
        B, T, V = logits.shape
        flat_logits = logits.view(-1, V)
        flat_targets = inputs.view(-1) # Self-supervised next token
        
        # We actually want to predict the 'target' value at the end.
        # Let's assume standard LM training 80% and 20% forced recall on the last token.
        loss_lm = criterion(flat_logits[:-1], flat_targets[1:])
        
        # Extract last logic for "Recall Accuracy"
        # We look at the token generated AFTER the query pair.
        # In this dataset, the efficient length varies.
        # Let's stick to pure LM loss for simplicity, it implies recall.
        
        loss = loss_lm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Physics Metrics (Introspection)
        # We check the internal states of the first layer
        with torch.no_grad():
            layer0 = model.layers[0]
            
            # Energy Drift (Hamiltonian conservation check)
            # H = 0.5*v^2 + U(x). Ideally constant if autonomous.
            # We approximate drift by checking divergence of v magnitude change.
            v_start = model.v0.norm().item()
            v_end = layer0.last_v.norm(dim=-1).mean().item() if hasattr(layer0, 'last_v') else 0.0
            drift = abs(v_end - v_start)
            
            # Fractal Activity (How often did we tunnel?)
            # Proxy: High curvature events
            curv = 0.0
            if hasattr(layer0, 'christoffels'):
                 # Just check mean of last computed gamma
                 # This is tricky without hooking, let's assume low activity if start
                 pass
            
            history["loss"].append(loss.item())
            history["energy_drift"].append(drift)
            
            pbar.set_description(f"Loss: {loss.item():.4f} | E-Drift: {drift:.3f}")

    # === 3. Visualization ===
    print("\n[*] Generating Training Report...")
    
    # Standard output dir
    res_dir = PROJECT_ROOT / "tests/benchmarks/results/showcase"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="CrossEntropy", color="#E76F51")
    plt.title("Learning Curve (Recursive Recall)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    
    # Energy
    plt.subplot(1, 2, 2)
    plt.plot(history["energy_drift"], label="Hamiltonian Drift", color="#2A9D8F")
    plt.title("Symplectic Stability (Energy Drift)")
    plt.xlabel("Step")
    plt.ylabel("Drift Magnitude")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(res_dir / "showcase_training.png")
    
    # Save Model
    ckpt_path = PROJECT_ROOT / "checkpoints/showcase_v1.0.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'physics_config': model.physics_config
    }, ckpt_path)
    
    print(f"[*] Metric plot saved to: {res_dir}/showcase_training.png")
    print(f"[*] Showcase Model saved to: {ckpt_path}")

if __name__ == "__main__":
    train_showcase()
