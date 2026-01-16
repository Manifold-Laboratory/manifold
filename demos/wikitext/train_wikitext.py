import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import urllib.request
import tarfile
import re
from collections import Counter, OrderedDict

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import Manifold
from src.optim import RiemannianAdam
import yaml
import argparse
import time

class SimpleVocab:
    def __init__(self, tokens, min_freq=3):
        print("Building Vocabulary...")
        counter = Counter(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.stoi = {'<unk>': 0, '<pad>': 1, '<eos>': 2}
        self.itos = ['<unk>', '<pad>', '<eos>']
        
        idx = 3
        for token, freq in sorted_by_freq_tuples:
            if freq >= min_freq:
                self.stoi[token] = idx
                self.itos.append(token)
                idx += 1
        print(f"Vocab Size: {len(self.stoi)}")
        
    def __len__(self):
        return len(self.stoi)
        
    def __call__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])

class WikiText2Custom:
    URL = 'https://s3.amazonaws.com/fast-ai-nlp/wikitext-2.tgz'
    
    def __init__(self, root='.data', split='train'):
        self.root = Path(root)
        self.dataset_path = self.root / 'wikitext-2'
        self.split = split
        self.download()
        
    def download(self):
        # Check if actual data files exist, not just directory
        train_file = self.dataset_path / 'train.csv'
        if not train_file.exists():
            print(f"Downloading WikiText-2 from {self.URL}...")
            self.root.mkdir(exist_ok=True)
            archive_path = self.root / 'wikitext-2.tgz'
            urllib.request.urlretrieve(self.URL, archive_path)
            
            print("Extracting...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(self.root)
            print("Download complete!")
                
    def read_tokens(self):
        file_map = {'train': 'train.csv', 'valid': 'test.csv', 'test': 'test.csv'}
        file_path = self.dataset_path / file_map[self.split]
        
        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header if exists
            lines = f.readlines()
            text = ' '.join(lines)
            # Basic tokenization
            text = text.replace('\n', ' <eos> ')
            tokens = re.findall(r'\S+', text.lower())
        return tokens

class SequentialDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.num_samples = len(data) // seq_len
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        # Ensure we don't go out of bounds
        chunk = self.data[start : start + self.seq_len + 1]
        
        # Determine src and tgt
        src = chunk[:-1]
        tgt = chunk[1:]
        
        # Safety check for last batch
        if len(src) < self.seq_len:
            pad_len = self.seq_len - len(src)
            src = torch.cat([src, torch.zeros(pad_len, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
            
        return src, tgt

def train_wikitext():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demos/wikitext.yaml')
    args = parser.parse_args()
    
    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_loader_raw = WikiText2Custom(split='train')
    val_loader_raw = WikiText2Custom(split='valid')
    
    train_tokens = train_loader_raw.read_tokens()
    val_tokens = val_loader_raw.read_tokens()
    
    # 2. Build Vocab from Train
    vocab = SimpleVocab(train_tokens)
    
    # 3. Numericalize
    print("Numericalizing data...")
    train_data = torch.tensor([vocab(t) for t in train_tokens], dtype=torch.long)
    val_data = torch.tensor([vocab(t) for t in val_tokens], dtype=torch.long)
    
    # Clamp to valid vocab range (safety check)
    train_data = torch.clamp(train_data, 0, len(vocab) - 1)
    val_data = torch.clamp(val_data, 0, len(vocab) - 1)
    
    # 4. Create Datasets
    seq_len = config['model']['max_len']
    train_dataset = SequentialDataset(train_data, seq_len)
    val_dataset = SequentialDataset(val_data, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], drop_last=True)
    
    # Update config
    config['model']['vocab_size'] = len(vocab)
    
    # Initialize Model
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = RiemannianAdam(model.parameters(), lr=config['training']['lr'])
    
    # Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    
    # Training Loop
    print("\nStarting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            logits, _, _ = model(src)  # Model returns (logits, state, christoffels)
            loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % config['training']['log_every'] == 0:
                cur_loss = total_loss / (i + 1)
                print(f"| Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {cur_loss:.4f} | PPL: {torch.exp(torch.tensor(cur_loss)):.2f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits, _, _ = model(src)  # Model returns (logits, state, christoffels)
                loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        ppl = torch.exp(torch.tensor(val_loss))
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"| End of Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val PPL: {ppl:.2f} | LR: {current_lr:.2e}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint every epoch
        os.makedirs(config['training']['save_dir'], exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f"{config['training']['save_dir']}/checkpoint_epoch_{epoch+1}.pt")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config['training']['save_dir']}/best_model.pt")
            print(f"✓ New best model saved (Val Loss: {val_loss:.4f})")

if __name__ == '__main__':
    train_wikitext()
