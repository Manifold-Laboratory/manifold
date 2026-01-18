import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Models
from src.model import Manifold
from src.optim import RiemannianAdam
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import measure_peak_memory

class ParityTask:
    """
    Parity Check (Modulo 2).
    Input: "0 1 0 1 1"
    Target: "0 1 1 0 1" (Cumulative XOR / Sum Mod 2)
    
    A classic test for state tracking. 
    The parity flips state every time a '1' is encountered.
    """
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        # Input: Random digits 0 or 1
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        
        # Target: Cumulative Sum Mod 2
        y = torch.cumsum(x, dim=1) % self.mod
        
        return x, y

def train_until_convergence(model, task, max_steps=1500, lr=3e-4, device='cuda', threshold=0.01):
    # Use Riemannian Adam for Manifold model to prevent weight explosion
    if isinstance(model, Manifold):
        print(f"[*] Using RiemannianAdam Optimizer (max_norm=10.0)")
        optimizer = RiemannianAdam(model.parameters(), lr=lr, weight_decay=1e-4, retraction='normalize', max_norm=10.0)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Standard for GPT

    # Start from L=20 (Basic XOR) to ensure mastery before scaling
    current_length = 20
    
    # OneCycleLR is faster and more stable for logical tasks
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, pct_start=0.2)
    model.train()
    
    # Adaptive Loss Selection
    if hasattr(model, 'physics_config'):
        readout_type = model.physics_config.get('readout', {}).get('type', 'standard')
    else:
        readout_type = 'standard'
        
    is_binary = (readout_type == 'binary' or readout_type == 'implicit')
    
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
        print(f"[*] Training with BCE Loss (Binary Mode)")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"[*] Training with CrossEntropy Loss (Standard Mode)")
    
    loss_history = []
    normalized_loss = 1.0
    static_acc = 0.0
    
    # Configure readout temperature schedule
    if hasattr(model, 'readout') and hasattr(model.readout, 'set_max_steps'):
        model.readout.set_max_steps(max_steps)
    
    pbar = tqdm(range(max_steps), desc=f"Training {model.__class__.__name__} (L=20)")
    normalized_loss = 1.0
    
    # Fixed Length Training (OOD Benchmark Standard)
    current_length = 20
    
    for i in pbar:
        # Standard OOD Setup: Train on fixed SHORT length, Test on LONG
        # No curriculum, just pure generalization test.
             
        # Generate batch with fixed length
        temp_task = ParityTask(length=current_length)
        x, y = temp_task.generate_batch(128, device=device) 
        
        optimizer.zero_grad()
        
        if isinstance(model, Manifold):
            logits, _, _ = model(x)
        else:
            logits = model(x)
            
        if is_binary:
            # Map targets (IDs) to bits
            coord_dim = model.physics_config.get('embedding', {}).get('coord_dim', 16)
            mask = 2**torch.arange(coord_dim).to(device)
            target_bits = (y.unsqueeze(-1) & mask) > 0
            target_bits = target_bits.float() # Map {0, 1} to {0, 1} (Neutral 0)
            
            # TASK-FOCUSED LOSS
            v_bits = int(np.ceil(np.log2(task.vocab_size)))
            v_bits = max(1, v_bits) 
            
            # MSE on relevant spatial coordinates
            # Only train the bits that exist in the vocab
            v_bits = int(np.ceil(np.log2(task.vocab_size)))
            v_bits = max(1, v_bits) 
            loss = criterion(logits[:, :, :v_bits], target_bits[:, :, :v_bits])
            
            # Removed SOFT-BOUNDARY REGULARIZATION (Let the model reach the corners)
        else:
            loss = criterion(logits.view(-1, task.vocab_size), y.view(-1))
            
        loss.backward()
        # Tighten Gradient Clipping (0.1 -> 0.05) to prevent optimization spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        scheduler.step()
        
        # Update temperature schedule
        if hasattr(model, 'readout') and hasattr(model.readout, 'update_step'):
            model.readout.update_step()
        
        loss_val = loss.item()
        
        # Exponential moving average for reporting
        normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
        
        # Calculate Accuracy
        if is_binary:
            # Token Accuracy (full ID check from bits)
            # For parity (vocab 2), bit 0 is enough.
            preds = (logits[:, :, 0] > 0.0).long()
            acc = (preds == y).float().mean().item()
        else:
            # Standard Readout (Softmax)
            # logits: [batch, seq, vocab]
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()

        # EMA for Accuracy reporting
        static_acc = 0.95 * static_acc + 0.05 * acc
        
        pbar.set_postfix({
            'loss': f"{normalized_loss:.4f}", 
            'acc': f"{static_acc*100:.1f}%",
            'L': current_length,
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        loss_history.append(loss_val)
        
        # Strictly solved threshold
        # User requested: Stop ONLY when loss is low enough (don't stop just for accuracy)
        # Target: Loss < 0.005 (implies perfect mastery beyond just thresholding)
        if normalized_loss < 0.005 and i > 200:
            print(f"🚀 Converged at step {i} (Loss: {normalized_loss:.5f}, Acc: {static_acc*100:.1f}%)")
            # Save correct best model
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}_parity.pth")
            break
            
    # Load best weights if we converged (or just return current if ran out of steps)
    ckpt_path = f"best_{model.__class__.__name__}_parity.pth"
    if os.path.exists(ckpt_path):
         model.load_state_dict(torch.load(ckpt_path))
         print(f"Loaded best weights from {ckpt_path}")
         
    return loss_history

def evaluate_length_generalization(model, task_cls, train_len=20, lengths=[20, 40, 60, 80, 100], device='cuda'):
    model.eval()
    accuracies = []
    memories = []
    
    print(f"\nEvaluating {model.__class__.__name__} Generalization (Train L={train_len}):")
    
    for length in lengths:
        # Instantiate task with new length
        task = task_cls(length=length)
        x, y = task.generate_batch(200, device=device) # Larger test batch
        
        # Measure VRAM
        peak_mem = 0.0
        logits = None
        
        try:
            with torch.no_grad():
                # Define closure for memory measurement
                def forward_pass_streaming():
                    # STREAMING INFERENCE (O(1) Verification)
                    # We process the sequence token-by-token without storing the full history of logits
                    # This proves that the INTERNAL STATE is constant size.
                    batch_size, seq_len = x.shape
                    state = None
                    last_logits = None
                    
                    # Manual loop to avoid allocating [B, L, V] output tensor
                    for t in range(seq_len):
                        input_t = x[:, t:t+1]
                        step_logits, state, _ = model(input_t, state=state)
                        if t == seq_len - 1:
                            last_logits = step_logits
                    return last_logits

                def forward_pass_standard():
                    return model(x)

                if isinstance(model, Manifold):
                     # Measure O(1) memory
                     peak_mem = measure_peak_memory(model, forward_pass_streaming)
                     # Get result (just last token is enough for "did it track state to the end?")
                     logits = forward_pass_streaming()
                     if isinstance(logits, tuple): logits = logits[0]
                else:
                     # Measure baseline memory
                     peak_mem = measure_peak_memory(model, forward_pass_standard)
                     logits = forward_pass_standard()
                     if isinstance(logits, tuple): logits = logits[0]
                
        except RuntimeError: # OOM
            print(f"  Len {length}: Failed (OOM)")
            accuracies.append(0.0)
            memories.append(-1.0)
            continue
        except Exception as e:
            print(f"  Len {length}: Failed ({str(e)})")
            accuracies.append(0.0)
            memories.append(-1.0)
            continue
        
        # Calculate Accuracy
        # If streaming, we only have the last logit [B, 1, V]
        # We compare against the last target
        
        if logits.shape[1] != length:
             # We have a strict subset (likely just the last one)
             # Slice target to match
             y_target = y[:, -logits.shape[1]:]
        else:
             # We have full sequence
             y_target = y
            
        if logits.shape[-1] != task.vocab_size:
            # Binary/Implicit Readout Mode: Decode bits back to token IDs
            # We assume the first N bits represent the ID (up to vocab_size)
            # For parity (vocab 2), we only check the first bit.
            # In a general case, we would do a nearest-neighbor search.
            if task.vocab_size == 2:
                 # Boolean threshold at 0.0 (Bits are mapped -1, 1)
                 preds = (logits[:, :, 0] > 0.0).long()
            else:
                 # Nearest-neighbor bit decoding (Simplistic for benchmark)
                 # Map logits to bits
                 pred_bits = (logits > 0.0).float() # [B, L, coord_dim]
                 
                 # Compute all possible vocab bits
                 all_ids = torch.arange(task.vocab_size).to(device)
                 mask = 2**torch.arange(logits.shape[-1]).to(device)
                 # [V, D]
                 vocab_bits = (all_ids.unsqueeze(-1) & mask) > 0
                 vocab_bits = vocab_bits.float()
                 
                 # L2 distance: [B, L, 1, D] - [1, 1, V, D]
                 dists = torch.cdist(pred_bits.view(-1, logits.shape[-1]), vocab_bits).view(logits.shape[0], logits.shape[1], -1)
                 preds = torch.argmin(dists, dim=-1)
        else:
            preds = torch.argmax(logits, dim=-1)
        
        # Check accuracy of the LAST 10% of tokens (or just the available ones)
        # We use y_target which is already aligned
        valid_len = preds.shape[1]
        check_len = max(1, valid_len // 10) # check last 10% of what we have
        
        correct = (preds[:, -check_len:] == y_target[:, -check_len:]).float().mean()
        
        acc = correct.item()
        accuracies.append(acc)
        memories.append(peak_mem)
        print(f"  Len {length}: {acc*100:.1f}% | VRAM: {peak_mem:.1f} MB")
            
    return accuracies, memories

def run_benchmark_v3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🥊 GFN vs Transformer: The 'State Tracking' Rumble (V3 - Parity)")
    print(f"Goal: Demonstrate O(1) State Tracking vs Attention's weakness.")
    
    # 1. Models
    # Dimensions need to be sufficient but not huge for a benchmark.
    dim = 128 # Reverted to 128 (Capacity is not the issue, Dynamics are)
    depth = 6 # Deepen for XOR state chain tracking
    heads = 4 
    vocab = 2 # 0, 1
    
    # Manifold (The Challenger - Hybrid Mode for stability test)
    gfn = Manifold(
        vocab_size=vocab, dim=dim, depth=depth, heads=heads,
        use_scan=False, 
        integrator_type='leapfrog',
        physics_config={
            'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
            'readout': {'type': 'binary'},  # NATIVE BINARY MODE
            'active_inference': {'enabled': True, 'reactive_curvature': {'enabled': True, 'plasticity': 0.05}},
            'hyper_curvature': {'enabled': True},
            'stability': {'base_dt': 0.3, 'damping': 0.05, 'residual_scale': 0.5} 
        }
    ).to(device)
    
    # MicroGPT (The Champion)
    gpt = MicroGPT(
        vocab_size=vocab, dim=dim, depth=4, heads=heads, max_len=1000
    ).to(device) 
    
    # 2. Train on SHORT sequences (L=2)
    train_task = ParityTask(length=2)
    
    print(f"\n--- Training Phase (Target: Loss < 0.005) ---")
    print(f"Manifold Model: {dim} dim, {depth} layers, 16-bit binary readout")
    # Reduced LR to 1e-4 to prevent collapse near convergence (Handling "Aggressive Penalization")
    gfn_loss = train_until_convergence(gfn, train_task, max_steps=1500, lr=1e-4, device=device)
    gpt_loss = train_until_convergence(gpt, train_task, max_steps=4000, lr=1e-3, device=device) # GPT needs more care/steps sometimes
    
    # 3. Test on LONG sequences
    lengths = [20, 50, 100, 200, 400, 500, 1000, 10000]
    
    gfn_acc, gfn_mem = evaluate_length_generalization(gfn, ParityTask, train_len=2, lengths=lengths, device=device)
    gpt_acc, gpt_mem = evaluate_length_generalization(gpt, ParityTask, train_len=2, lengths=lengths, device=device)
    
    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    
    # Plot 1: Accuracy
    ax1.plot(lengths, gfn_acc, 'o-', linewidth=3, color='#E76F51', label='Manifold (Hyper-Cognitive)')
    ax1.plot(lengths, gpt_acc, 's--', linewidth=3, color='#2A9D8F', label='Transformer (Attention)')
    ax1.axvline(x=20, color='gray', linestyle=':', label='Training Length (L=20)')
    ax1.set_title("OOD Generalization (Accuracy)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Sequence Length", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(fontsize=10)
    
    # Plot 2: VRAM Scaling
    # Filter out failed runs (-1.0)
    gfn_mem_valid = [m if m >= 0 else np.nan for m in gfn_mem]
    gpt_mem_valid = [m if m >= 0 else np.nan for m in gpt_mem]
    
    ax2.plot(lengths, gfn_mem_valid, 'o-', linewidth=3, color='#E76F51', label='Manifold')
    ax2.plot(lengths, gpt_mem_valid, 's--', linewidth=3, color='#2A9D8F', label='Transformer')
    ax2.set_title("VRAM Scaling (O(1) vs O(N))", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Sequence Length", fontsize=12)
    ax2.set_ylabel("Peak VRAM (MB)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log') # Log scale to show O(N^2) explosion if present
    
    out_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "gfn_superiority"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parity_generalization.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    # Save Metrics to JSON
    import json
    results_data = {
        "task": "Parity Check (Modulo 2)",
        "train_length": 20,
        "test_lengths": lengths,
        "models": {
            "Manifold": {
                "training_loss": gfn_loss,
                "generalization_accuracy": gfn_acc,
                "peak_vram_mb": gfn_mem
            },
            "MicroGPT": {
                "training_loss": gpt_loss,
                "generalization_accuracy": gpt_acc,
                "peak_vram_mb": gpt_mem
            }
        }
    }
    
    json_path = out_dir / "parity_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\n✅ Benchmark Complete!")
    print(f"Chart saved to: {out_path}")
    print(f"Data saved to: {json_path}")

if __name__ == "__main__":
    run_benchmark_v3()
