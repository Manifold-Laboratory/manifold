
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
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import hamiltonian_loss, curiosity_loss, geodesic_regularization
import random

# ... (Dataset class same)

# ...

def train_showcase():
    # ... (Setup same)
    
    # Use Riemannian Optimizer (Geometry-Aware)
    optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Task Loss: O(1) MSE
    task_criterion = nn.MSELoss() 
    
    # Physics Loss Config
    lambda_h = 0.01 # energy conservation
    lambda_c = 0.05 # curiosity
    lambda_g = 0.001 # smooth curvature
    
    # Metrics Tracking
    history = {
        "loss": [],
        "mse": [],
        "hamiltonian": [],
        "curiosity": [],
        "energy_drift": [],
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
        # Returns logits, state, AND physics metadata (christoffels, velocities if tracked)
        # Note: Manifold.forward returns (logits, state, christoffels).
        # We need to ensure we access velocities for Hamiltonian loss.
        # Currently Manifold forward returns: logits, (x_final, v_final), all_christoffels
        # To get FULL velocity history for Hamiltonian loss, the model needs to return it.
        # Standard Manifold.forward DOES NOT return full velocity history in sequential mode (only final).
        # However, for 'showcase', we want to demonstrate it.
        # Let's rely on standard forward output for now.
        
        logits, (x_final, v_final), christoffels = model(inputs) 
        
        # === 1. Task Loss (O(1)) ===
        pred_coords = logits  # [B, L, 32]
        
        # We need target COORDS (32-d) to match predictions.
        # model.embedding() returns 512-d vectors. We need the raw 32-d binary codes.
        lm_targets = inputs[:, 1:] 
        
        # Replicate Binary Functional Logic
        coord_dim = 32
        mask = 2**torch.arange(coord_dim).to(device)
        bits = (lm_targets.unsqueeze(-1) & mask) > 0
        target_coords = bits.float() * 2 - 1 # [B, L-1, 32]

        pred_coords_shifted = pred_coords[:, :-1, :] # [B, L-1, 32]
        
        loss_mse = task_criterion(pred_coords_shifted, target_coords)
        total_loss = loss_mse
        
        # === 2. Physics Losses (The "Secret Sauce") ===
        # Note: Hamiltonian loss requires a sequence of velocities.
        # Our current Manifold.forward only returns the *final* state tuple.
        # To compute H-Loss properly, we would need the full trace.
        # For this showcase, we will calculate Hamiltonian loss on the Batch Variance (Curiosity)
        # and Geodesic Regularization on the Christoffel symbols we DO have.
        
        # Geodesic Reg (Penalize black holes)
        if christoffels:
             l_g = geodesic_regularization(None, christoffels, lambda_g)
             total_loss += l_g
             
        # Curiosity (Maximize Entropy of final velocity)
        if v_final is not None:
             l_c = curiosity_loss([v_final], lambda_c)
             total_loss += l_c
             
        # Hamiltonian: Approximate using Start vs End energy?
        # Exact H-loss needs steps. We'll skip exact H-loss in this specific run.
             
        loss = total_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history["loss"].append(loss.item())
        history["mse"].append(loss_mse.item())
        
        desc = f"Loss: {loss.item():.4f} | MSE: {loss_mse.item():.4f}"
        if christoffels: desc += f" | Curv: {l_g.item():.4f}"
        pbar.set_description(desc)
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
    
    # Manifold Setup with FULL Physics + INFINITE Features
    model = Manifold(
        vocab_size=vocab_size,
        dim=512,
        depth=6,     # Deep enough for recursion
        heads=8,
        physics_config={
            # Infinite Features (O(1) Memory Claims)
            "embedding": {"type": "functional", "mode": "binary", "coord_dim": 32},
            "readout": {"type": "implicit", "coord_dim": 32}, # O(1) Enabled
            
            # Cognitive Physics
            "active_inference": {"enabled": True, "plasticity": 0.1},
            "fractal": {"enabled": True, "threshold": 0.4}, # Sensitive fractal trigger
            "symmetries": {"enabled": True},
            "singularities": {"enabled": True, "strength": 5.0}
        }
    ).to(device)
    
    # Use Riemannian Optimizer (Geometry-Aware)
    optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # For O(1) Implicit Readout, we regress coordinates, not logits
    criterion = nn.MSELoss() 
    
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
        # logits are [Batch, Seq, CoordDim] because of Implicit Readout
        pred_coords, _, _ = model(inputs) 
        
        # Target Preparation for O(1) Loss
        # We need the ground truth coordinates for the target tokens (32-d)
        # Target tensor is next-token for Causal LM: inputs shifted by 1
        lm_targets = inputs[:, 1:] # [B, T-1]
        
        # O(1) Target Gen: ID -> Binary -> {-1, 1}
        coord_dim = 32
        mask = 2**torch.arange(coord_dim).to(device)
        bits = (lm_targets.unsqueeze(-1) & mask) > 0
        target_coords = bits.float() * 2 - 1 # [B, T-1, 32]
             
        # Align predictions (drop last to match)
        pred_coords_shifted = pred_coords[:, :-1, :] # [B, T-1, 32]
        
        loss = criterion(pred_coords_shifted, target_coords)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics recording happens next
        # (Cleaned up legacy code)
        
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
