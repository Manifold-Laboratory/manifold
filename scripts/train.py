#!/usr/bin/env python3
"""
GFN Unified Training Script
============================

Usage:
    python scripts/train.py --model configs/model/gfn_medium.yaml \\
                            --training configs/training/math_oracle.yaml \\
                            --hardware configs/hardware/rtx_4090.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import Manifold, GFNLoss, RiemannianAdam
from src.math_dataset import MathDataset
from src.safety import GPUMonitor



def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(hw_cfg: dict) -> torch.device:
    """Setup compute device based on config."""
    cfg = hw_cfg.get('hardware', {})
    if torch.cuda.is_available() and cfg.get('gpu_enabled', True):
        return torch.device('cuda')
    return torch.device('cpu')


def build_model(model_cfg: dict, device: torch.device) -> nn.Module:
    """Instantiate Manifold model from config."""
    cfg = model_cfg['model']
    
    # Check if we should use the O(1) memory Adjoint version
    if cfg.get('use_adjoint', False):
        from src.adjoint import AdjointManifold, HAS_TORCHDIFFEQ
        if HAS_TORCHDIFFEQ:
            print("Initializing AdjointManifold (O(1) Memory Mode)")
            model = AdjointManifold(
                vocab_size=cfg['vocab_size'],
                dim=cfg['dim'],
                depth=cfg['depth'],
                rank=cfg['rank']
            )
            return model.to(device)
        else:
            print("Warning: use_adjoint=True but torchdiffeq missing. Falling back to Standard Manifold.")

    print("Initializing Standard Manifold (Leapfrog/Discrete Mode)")
    model = Manifold(
        vocab_size=cfg['vocab_size'],
        dim=cfg['dim'],
        depth=cfg['depth'],
        rank=cfg['rank'],
        heads=cfg.get('heads', 4),
        integrator_type=cfg.get('integrator', 'leapfrog'),
        use_scan=cfg.get('use_scan', True) # Default to True for speed if not specified!
    )
    return model.to(device)


def build_dataset(train_cfg: dict):
    """Build dataset based on training config."""
    cfg = train_cfg['training']
    if cfg['dataset'] == 'math':
        return MathDataset(
            size=cfg.get('dataset_size'),
            max_digits=cfg.get('max_digits', 8),
            file_path=cfg.get('dataset_path')
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def run_demo(model: nn.Module, dataset, device: torch.device):
    """Run live inference demo."""
    model.eval()
    test_cases = ["42+9=", "131-31=", "50*5=", "999+1=", "123*10="]
    
    print("\n[LIVE DEMO]")
    with torch.no_grad():
        for prompt in test_cases:
            ids = [dataset.char_to_id[c] for c in prompt]
            input_seq = torch.tensor([ids]).to(device)
            logits, state = model(input_seq)
            
            generated = list(ids)
            curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated.append(curr_token.item())
            
            for _ in range(12):
                logits, state = model(curr_token, state=state)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                tok_id = next_token.item()
                if tok_id == dataset.char_to_id['<EOS>']:
                    break
                generated.append(tok_id)
                curr_token = next_token.unsqueeze(0)
            
            result = dataset.decode(generated).split('=')[-1]
            print(f"  {prompt} {result}")
    
    print("-----------\n")
    model.train()


def train(model_cfg: dict, train_cfg: dict, hw_cfg: dict):
    """Main training loop."""
    # Setup
    device = setup_device(hw_cfg)
    # Build model and compile
    model = build_model(model_cfg, device)
    
    # Compilation
    # Windows support for Inductor (Triton) is limited.
    # We disable it by default to prevent crashes.
    if hasattr(torch, 'compile') and os.name != 'nt':
        print("Compiling model (PyTorch 2.0+)...")
        model = torch.compile(model)
    elif os.name == 'nt':
        print("Skipping torch.compile on Windows (Triton missing)")
        
    dataset = build_dataset(train_cfg)
    
    train_params = train_cfg['training']
    hw_params = hw_cfg['hardware']
    
    # Print info
    # Note: param_count works on compiled model too
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'='*50}")
    print(f"GFN Training Session")
    print(f"{'='*50}")
    print(f"Model: {param_count:.2f}M parameters")
    print(f"Device: {device} ({hw_params.get('gpu_name', 'Unknown')})")
    print(f"Dataset: {train_params['dataset']} (Infinite Stream)")
    print(f"Batch: {train_params['batch_size']} x {train_params.get('accumulation_steps', 1)} = {train_params['batch_size'] * train_params.get('accumulation_steps', 1)}")
    print(f"LR: {train_params['learning_rate']}, WD: {train_params['weight_decay']}")
    print(f"{'='*50}\n")
    
    # Safety monitor
    safety = GPUMonitor(threshold_temp=hw_params.get('max_temp_c', 75))
    safety.start()
    
    # Optimizer: Riemannian Adam (respects manifold geometry)
    optimizer = RiemannianAdam(
        model.parameters(),
        lr=train_params['learning_rate'],
        weight_decay=train_params['weight_decay'],
        retraction='normalize',
        max_norm=10.0
    )
    
    # Loss: GFNLoss with Hamiltonian regularization & Curiosity
    criterion = GFNLoss(
        lambda_h=train_params.get('lambda_h', 0.01),  # Hamiltonian energy conservation weight
        lambda_c=train_params.get('lambda_c', 0.0),   # Entropy-Driven Curiosity (Thermodynamics)
        ignore_index=dataset.char_to_id['<PAD>']
    )
    
    # AMP
    scaler = torch.amp.GradScaler('cuda') if train_params.get('use_amp', True) else None
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        # shuffle=True, # NOT SUPPORTED for IterableDataset
        num_workers=hw_params.get('num_workers', 4),
        pin_memory=hw_params.get('pin_memory', True),
        collate_fn=dataset.collate_fn
    )
    
    # Checkpoint path
    ckpt_path = Path(train_cfg['checkpoint']['path'])
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    vocab_size = model_cfg['model']['vocab_size']
    
    # Resume logic
    start_epoch = 0
    checkpoint_files = sorted(list(ckpt_path.glob("epoch_*.pt")), key=lambda p: int(p.stem.split('_')[1]))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"Resuming from checkpoint: {latest_ckpt}")
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Allow resetting optimizer (useful when changing architecture/physics)
            if '--reset-optimizer' in sys.argv:
                print(">>> RESETTING OPTIMIZER STATE (New Training Phase) <<<")
                # Keep start_epoch to maintain file numbering, or could reset if desired.
                # Here we keep numbering to show continuity of the "model".
                start_epoch = checkpoint['epoch'] + 1
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming at Epoch {start_epoch}")
        except RuntimeError as e:
            print(f"\n[WARNING] Checkpoint mismatch (different model architecture?): {e}")
            print("Ignoring checkpoint involved starting from scratch...")
            start_epoch = 0

    # Training loop
    model.train()
    try:
        for epoch in range(start_epoch, train_params['epochs']):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            
            for batch_idx, inputs in enumerate(dataloader):
                if batch_idx >= train_params['steps_per_epoch']:
                    break
                    
                safety.check()
                inputs = inputs.to(device, non_blocking=True)
                
                targets = torch.roll(inputs, -1, dims=1).clone()
                targets[:, -1] = dataset.char_to_id['<PAD>']
                
                # Forward
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits, (x_final, v_final) = model(inputs)
                        
                        # Use GFNLoss with Hamiltonian regularization
                        loss, loss_dict = criterion(
                            logits, 
                            targets, 
                            velocities=[v_final]  # Pass final velocity for energy tracking
                        )
                        
                        if torch.isnan(loss):
                            print("Warning: NaN loss, skipping batch.")
                            optimizer.zero_grad()
                            continue
                            
                        loss = loss / train_params['accumulation_steps']
                    
                    scaler.scale(loss).backward()
                    epoch_loss += loss.item() * train_params['accumulation_steps']
                    
                    if (batch_idx + 1) % train_params['accumulation_steps'] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['max_grad_norm'])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    logits, (x_final, v_final) = model(inputs)
                    loss, loss_dict = criterion(logits, targets, velocities=[v_final])
                    loss.backward()
                    epoch_loss += loss.item()
                    
                    if (batch_idx + 1) % train_params['accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['max_grad_norm'])
                        optimizer.step()
                        optimizer.zero_grad()

                    
                    if (batch_idx + 1) % train_params['accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['max_grad_norm'])
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Logging
                if (batch_idx // train_params['accumulation_steps']) % train_params['log_interval'] == 0:
                    step = batch_idx // train_params['accumulation_steps']
                    dt = time.time() - epoch_t0
                    speed = (step * train_params['batch_size'] * train_params['accumulation_steps']) / max(dt, 0.001)
                    print(f"Epoch {epoch} | Step {step}/{train_params['steps_per_epoch']} | "
                          f"Loss: {loss.item()*train_params['accumulation_steps']:.4f} | "
                          f"Speed: {speed:.1f} ex/s")
            
            # End of epoch
            avg_loss = epoch_loss / train_params['steps_per_epoch']
            epoch_time = time.time() - epoch_t0
            print(f"\n>>> EPOCH {epoch} COMPLETE | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Demo
            if epoch % train_params['demo_interval'] == 0:
                run_demo(model, dataset, device)
            
            # Checkpoint
            if epoch % train_cfg['checkpoint']['save_interval'] == 0:
                ckpt_file = ckpt_path / f"epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_file)
                print(f"Checkpoint saved: {ckpt_file}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving emergency checkpoint...")
        torch.save(model.state_dict(), ckpt_path / "interrupted.pt")
    finally:
        safety.stop()
        print("Training session ended.")


def main():
    parser = argparse.ArgumentParser(description="GFN Unified Training")
    parser.add_argument('--model', type=str, required=True, help="Path to model config YAML")
    parser.add_argument('--training', type=str, required=True, help="Path to training config YAML")
    parser.add_argument('--hardware', type=str, required=True, help="Path to hardware config YAML")
    parser.add_argument('--reset-optimizer', action='store_true', help="Reset optimizer state when resuming")
    args = parser.parse_args()
    
    model_cfg = load_config(args.model)
    train_cfg = load_config(args.training)
    hw_cfg = load_config(args.hardware)
    
    train(model_cfg, train_cfg, hw_cfg)


if __name__ == "__main__":
    main()
