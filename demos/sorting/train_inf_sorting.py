import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import sys
from pathlib import Path
import yaml
import argparse
import time
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import Manifold
from src.optim import RiemannianAdam
from src.embeddings import FunctionalEmbedding 

class CausalSortingDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.SEP = vocab_size     # Token for separator
        self.EOS = vocab_size + 1 # Token for end
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random sequence
        vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
        sorted_vals = sorted(vals)
        
        # Input: [vals] [SEP] [sorted]
        # Target: [vals] [SEP] [sorted] [EOS]
        full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
        
        src = torch.tensor(full_seq[:-1], dtype=torch.long)
        tgt = torch.tensor(full_seq[1:], dtype=torch.long)
        
        return src, tgt

def get_binary_coords(token_ids, coord_dim, device):
    """
    Map ID -> Bits {0, 1} flat.
    """
    mask = 2**torch.arange(coord_dim).to(device)
    bits = (token_ids.unsqueeze(-1) & mask) > 0
    return bits.float() # {0.0, 1.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_range = config['task']['vocab']
    real_vocab_size = vocab_range + 2 
    config['model']['vocab_size'] = real_vocab_size
    
    # Dataset
    train_ds = CausalSortingDataset(config['task']['num_train'], config['task']['seq_len'], vocab_range)
    val_ds = CausalSortingDataset(config['task']['num_val'], config['task']['seq_len'], vocab_range)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], num_workers=0)
    
    # Initialize Model with Binary Config
    config['physics']['embedding']['mode'] = 'binary'
    config['physics']['readout']['type'] = 'binary' # Using binary readout
    coord_dim = config['physics']['embedding']['coord_dim'] # 16 bits -> 65k vocab
    
    # Needs enough bits?
    # 2^16 = 65536. Vocab is ~102. Safe.
    
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    
    # Detect Infinite Mode
    is_infinite = True # forcing it for this script
    
    print(f"\n[*] INFINITE MODE (BINARY) DETECTED")
    print(f"    - Input: Binary Functional (O(1))")
    print(f"    - Output: Binary Multi-Label Classification")
    print(f"    - Coord Dim: {coord_dim}")
        
    # Params
    total = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total/1e6:.2f}M\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    # OneCycleLR: Super-Convergence Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['training']['lr'] * 10, # Aggressive peak
        steps_per_epoch=len(train_loader),
        epochs=config['training']['epochs'],
        pct_start=0.3
    )
    
    # Loss: Binary Cross Entropy
    criterion = nn.BCEWithLogitsLoss() 
    
    best_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # Forward -> Logits [B, L, Bits]
            pred, _, _ = model(src)
            
            # Target -> Bits {0, 1} [B, L, Bits]
            tgt_bits = get_binary_coords(tgt, coord_dim, device)
            
            loss = criterion(pred, tgt_bits)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            scheduler.step() # OneCycleLR steps PER BATCH
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        correct_bits = 0
        total_bits = 0
        correct_tokens = 0
        total_tokens = 0
        correct_seqs = 0
        total_seqs = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                pred, _, _ = model(src)
                
                # Check Sorted Part (Logic reuse)
                seq_len = config['task']['seq_len']
                start_check = seq_len 
                
                # 1. Bit Accuracy
                # Pred: [B, L, Bits] (Logits)
                # Tgt: [B, L, Bits] {0,1}
                tgt_bits = get_binary_coords(tgt, coord_dim, device)
                
                # Focus only on the sorted part for metrics
                pred_sorted_logits = pred[:, start_check : start_check + seq_len]
                tgt_sorted_bits = tgt_bits[:, start_check : start_check + seq_len]
                
                # Fix: Logits > 0.0 is equivalent to Sigmoid > 0.5
                pred_bits = (pred_sorted_logits > 0.0).float() 
                
                correct_bits += (pred_bits == tgt_sorted_bits).sum().item()
                total_bits += tgt_sorted_bits.numel()
                
                # 2. Token Accuracy (All bits match)
                # [B, L, Bits] -> [B, L] (All bits match)
                token_matches = torch.all(pred_bits == tgt_sorted_bits, dim=-1)
                correct_tokens += token_matches.sum().item()
                total_tokens += token_matches.numel()

                # 3. Sequence Accuracy (All tokens match)
                # [B, L] -> [B]
                seq_matches = torch.all(token_matches, dim=-1)
                correct_seqs += seq_matches.sum().item()
                total_seqs += src.size(0)
        
        bit_acc = correct_bits / total_bits
        token_acc = correct_tokens / total_tokens
        seq_acc = correct_seqs / total_seqs
        val_acc = bit_acc  # Track bit accuracy for checkpointing
        
        curr_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Bit Acc: {bit_acc*100:.2f}% | Tok Acc: {token_acc*100:.2f}% | Seq Acc: {seq_acc*100:.2f}% | LR: {curr_lr:.2e}")
        
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            print("--> New Best Checkpoint")

if __name__ == '__main__':
    main()
