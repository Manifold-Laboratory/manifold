import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import Manifold
from src.optim import RiemannianAdam
import yaml
import argparse
import time
from tqdm import tqdm

class CausalSortingDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.SEP = vocab_size     # Token for separator
        self.EOS = vocab_size + 1 # Token for end
        
        # Actual vocab size for model = range + 2 special
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random sequence
        vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
        sorted_vals = sorted(vals)
        
        # Input: [vals] [SEP] [sorted]
        # Target: [vals] [SEP] [sorted] [EOS] (Shifted by 1 in training)
        
        full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
        
        src = torch.tensor(full_seq[:-1], dtype=torch.long)
        tgt = torch.tensor(full_seq[1:], dtype=torch.long)
        
        return src, tgt

def train_inf_sorting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_range = config['task']['vocab']
    real_vocab_size = vocab_range + 2 # + SEP, EOS
    
    # Update model config with real vocab size
    config['model']['vocab_size'] = real_vocab_size
    
    # Disable Scan for INF initial testing if enabled (safeguard)
    # config['model']['use_scan'] = False
    
    train_ds = CausalSortingDataset(config['task']['num_train'], config['task']['seq_len'], vocab_range)
    val_ds = CausalSortingDataset(config['task']['num_val'], config['task']['seq_len'], vocab_range)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], num_workers=0)
    
    print(f"[*] Initializing Manifold with IMPLICIT FIELDS...")
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    
    # Params Count
    total_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(p.numel() for p in model.embedding.parameters())
    print(f"\nTotal Params: {total_params/1e6:.2f}M")
    print(f"Embedding Params: {emb_params/1e6:.4f}M (Efficient!)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    best_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # Forward
            logits, _, _ = model(src)
            loss = criterion(logits.reshape(-1, real_vocab_size), tgt.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits, _, _ = model(src)
                preds = torch.argmax(logits, dim=-1)
                
                # Check sorted part
                seq_len = config['task']['seq_len']
                # Indices: 0..9 (Input), 10 (SEP)
                # Target Sorted starts at index 11? 
                # Input: [I0..I9] [SEP] [S0..S9]
                # Tgt:   [I1..I9] [SEP] [S0..S9] [EOS]
                # The prediction for position '10' (SEP) is S0.
                # So preds[:, 10] matches tgt[:, 10] which is S0.
                
                start_check = seq_len  # After input
                # Length of sorted part is seq_len 
                
                sorted_preds = preds[:, start_check : start_check + seq_len]
                sorted_tgts = tgt[:, start_check : start_check + seq_len]
                
                row_matches = torch.all(sorted_preds == sorted_tgts, dim=1)
                correct += row_matches.sum().item()
                total += src.size(0)
                
        val_acc = correct / total
        curr_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Mean Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | LR: {curr_lr:.2e}")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"--> New Best!")

if __name__ == '__main__':
    train_inf_sorting()
