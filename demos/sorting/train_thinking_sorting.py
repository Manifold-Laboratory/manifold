
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import MANIFOLD V2
from src.model import Manifold
from src.losses import GFNLoss

# --- DATASET ---
class SortingDataset(Dataset):
    """
    Sorting Task:
    Input: [5, 2, 9, 1]
    Target: [1, 2, 5, 9]
    """
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=100, seed=42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        torch.manual_seed(seed)
        # Generate random sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        # Sort them for targets
        self.targets = torch.sort(self.data, dim=1).values
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# --- TRAINING LOOP ---
def train(config_path, device='cuda'):
    print(f"[*] Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Model Params
    dim = config['model']['dim']
    depth = config['model']['depth']
    heads = config['model']['heads']
    vocab = config['task']['vocab']
    seq_len = config['task']['seq_len']
    
    # Physics Params
    physics = config['physics']
    
    print("[*] Initializing MANIFOLD v2.1 (Thinking Mode)...")
    model = Manifold(
        vocab_size=vocab,
        dim=dim,
        depth=depth,
        heads=heads,
        integrator_type=physics['solver'],
        use_scan=config['model'].get('use_scan', False),
        physics_config=physics
    ).to(device)
    
    # Optimizer (AdamW standard)
    lr = float(config['training']['lr'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # Loss Function (GFN Loss)
    # Thinking update: Add curiosity to encourage exploring thoughts?
    # For now standard loss
    criterion = GFNLoss(
        lambda_h=10.0,
        lambda_g=0.001,
        lambda_c=0.01, # Curiosity helps thinking diversity
        lambda_n=0.1
    )
    
    # Dataset
    batch_size = config['training']['batch_size']
    train_ds = SortingDataset(num_samples=config['task']['num_train'], seq_len=seq_len, vocab_size=vocab)
    val_ds = SortingDataset(num_samples=config['task']['num_val'], seq_len=seq_len, vocab_size=vocab, seed=999)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"[*] Starting Training for {config['training']['epochs']} epochs...")
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Manifold returns: logits, state, christoffels
            logits, _, christoffels = model(x)
            
            # Loss
            # We assume model.layers[0].christoffels contains the geometric info
            # The model wrapper aggregates christoffels from all layers if implemented, 
            # currently checks model.py return signature.
            # Updated model.py returns `logits, (x_final, v_final), all_christoffels`
            
            # Need velocities for Hamiltonian?
            # model.py currently doesn't return full velocity sequence in sequential mode easily
            # without collecting it. 
            # For this demo we focus on Task Loss + Geometry Reg
            
            loss, loss_dict = criterion(logits, y, christoffel_outputs=christoffels)
            
            loss.backward()
            
            # Gradient Clipping (Critical for geometric stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ce': f"{loss_dict.get('ce',0):.3f}"})
            
        avg_train_loss = epoch_loss / len(train_dl)
        train_losses.append(avg_train_loss)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x)
                
                # Accuracy (Exact Match?)
                # Or token accuracy
                preds = torch.argmax(logits, dim=-1)
                
                # Element-wise accuracy
                correct += (preds == y).sum().item()
                total += y.numel()
                
                v_loss, _ = criterion(logits, y)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_dl)
        val_losses.append(avg_val_loss)
        acc = correct / total
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")
        
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config['training']['save_dir'], f"thinking_model_ep{epoch+1}.pt")
            os.makedirs(config['training']['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    
    total_time = time.time() - start_time
    print(f"[*] Training Finished in {total_time:.2f}s")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Manifold v2.1 (Thinking) - Sorting Task')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('thinking_sorting_results.png')
    print("[*] Saved plot to thinking_sorting_results.png")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    train('configs/demos/sorting.yaml', device=device)
