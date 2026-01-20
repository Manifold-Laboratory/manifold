# MANIFOLD Training Guide

**Version:** 2.6.0 "Symplectic Forgetting"  
**Last Updated:** January 20, 2026

Complete guide for training MANIFOLD models, from quick start to advanced optimization.

---

## Quick Start

### Minimal Example

```python
import torch
from gfn.model import Manifold
from gfn.optim import RiemannianAdam

# Model
model = Manifold(vocab_size=1000, dim=256, depth=6, heads=4).cuda()

# CRITICAL: Use RiemannianAdam (standard Adam will fail)
optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,
    retraction='normalize',
    max_norm=10.0
)

# Training loop
criterion = torch.nn.CrossEntropyLoss()
for x, y in dataloader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()
    
    logits, _, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    
    # Gradient clipping (REQUIRED)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    
    optimizer.step()
```

---

## Configuration

### Basic Configuration (config.yaml)

```yaml
model:
  vocab_size: 50257
  dim: 512
  depth: 12
  heads: 8
  rank: 32
  integrator_type: leapfrog
  
optimizer:
  type: RiemannianAdam
  lr: 1e-4
  weight_decay: 1e-4
  retraction: normalize
  max_norm: 10.0
  
training:
  batch_size: 32
  max_epochs: 100
  gradient_clip: 0.05
  
scheduler:
  type: OneCycleLR
  max_lr: 1e-4
  pct_start: 0.2
```

### Advanced Physics Configuration

```yaml
model:
  physics_config:
    # Embedding
    embedding:
      type: functional    # O(1) w.r.t. vocab size
      mode: binary        # Binary coordinate encoding
      coord_dim: 16
    
    # Readout
    readout:
      type: binary        # Hard-threshold decoder
    
    # Active Inference
    active_inference:
      enabled: true
      reactive_curvature:
        enabled: true
        plasticity: 0.05
    
    # Hyper-Curvature
    hyper_curvature:
      enabled: true
    
    # Stability
    stability:
      base_dt: 0.3
      residual_scale: 0.5
      curvature_clamp: 5.0
```

---

## Critical Hyperparameters

### Learning Rate

**Recommended Values**:
- Binary tasks: `1e-4`
- Language modeling: `3e-5` to `1e-4`
- Fine-tuning: `1e-5`

**Sensitivity**: Manifold models are **highly sensitive** to learning rate.
- Too high (>3e-4): Divergence or oscillation
- Too low (<1e-5): Slow convergence (>10K steps)

**Finding Optimal LR**:
```python
from torch.optim.lr_scheduler import LRFinder

# Run LR range test
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1e-2, num_iter=100)
lr_finder.plot()  # Look for steepest descent
```

### Gradient Clipping

**Required**: `max_norm=0.05` (tighter than standard 0.1-1.0)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
```

**Why**: Geometric updates amplify gradient magnitudes near convergence.

### Integration Timestep (dt)

**Default**: `0.3` (recommended for most tasks)

**Tuning**:
- Increase (0.5-1.0): Faster training, risk of instability
- Decrease (0.1-0.2): More stable, slower convergence

Set via `physics_config['stability']['base_dt']`.

---

## Optimizer: RiemannianAdam

### Why Required

Standard Adam performs Euclidean updates that violate manifold constraints, causing "Euclidean drift":
- Symptom: Loss oscillates chaotically (1.0 → 0.2 → 1.0)
- Solution: Use RiemannianAdam with retraction

### Configuration

```python
from gfn.optim import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,                    # Lower than standard models
    betas=(0.9, 0.999),         # Standard Adam betas
    eps=1e-8,
    weight_decay=1e-4,          # Decoupled weight decay
    retraction='normalize',     # 'normalize', 'cayley', 'euclidean'
    max_norm=10.0               # Manifold bound
)
```

**Retraction Types**:
- `'normalize'`: Projects weights onto bounded sphere (recommended)
- `'cayley'`: Orthogonal retraction (experimental)
- `'euclidean'`: Disables retraction (not recommended)

---

## Learning Rate Schedules

### OneCycleLR (Recommended)

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,  # epochs * steps_per_epoch
    pct_start=0.2,            # 20% warm-up
    anneal_strategy='cos'
)

# Call after each step
for batch in dataloader:
    train_step(batch)
    scheduler.step()
```

**Benefits**:
- Fast warm-up prevents early instability
- Cosine annealing aids convergence

### Alternative: ReduceLROnPlateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Halve LR on plateau
    patience=10,      # Wait 10 epochs
    threshold=1e-4
)

# Call after each epoch
scheduler.step(val_loss)
```

---

## Loss Functions

### Binary Cross-Entropy (Binary Readout)

```python
criterion = torch.nn.BCEWithLogitsLoss()

# Compute loss
logits, _, _ = model(x)  # [batch, seq, coord_dim]
target_bits = get_binary_targets(y)  # [batch, seq, coord_dim]
loss = criterion(logits, target_bits)
```

**Use When**: `physics_config['readout']['type'] == 'binary'`

### Cross-Entropy (Standard Readout)

```python
criterion = torch.nn.CrossEntropyLoss()

# Compute loss
logits, _, _ = model(x)  # [batch, seq, vocab]
loss = criterion(logits.view(-1, vocab_size), y.view(-1))
```

**Use When**: `physics_config['readout']['type'] == 'standard'`

### Custom: Hamiltonian Regularization (Experimental)

```python
# Penalize energy drift
energy_loss = torch.mean((v_final.norm(dim=-1) - v_initial.norm(dim=-1))**2)
total_loss = task_loss + 0.01 * energy_loss
```

---

## Training Loop (Complete)

```python
import torch
from torch.utils.data import DataLoader
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from torch.optim.lr_scheduler import OneCycleLR

# Setup
model = Manifold(...).cuda()
optimizer = RiemannianAdam(model.parameters(), lr=1e-4, max_norm=10.0)
criterion = torch.nn.CrossEntropyLoss()

max_epochs = 100
total_steps = max_epochs * len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps)

# Training
for epoch in range(max_epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        
        # Forward
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            logits, _, _ = model(x)
            val_loss += criterion(logits.view(-1, vocab_size), y.view(-1))
    print(f"Epoch {epoch}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch}.pth')
```

---

## Convergence Monitoring

### Key Metrics

1. **Task Loss**: Should decrease monotonically
2. **Gradient Norm**: Should stabilize (not collapse to 0)
3. **Velocity Norm**: Should remain bounded (not explode)
4. **Accuracy**: Should increase steadily

### Diagnostic Plots

```python
import matplotlib.pyplot as plt

# Track during training
losses = []
grad_norms = []
velocity_norms = []

# In training loop:
grad_norm = sum(p.grad.norm() for p in model.parameters())
grad_norms.append(grad_norm.item())

_, (_, v), _ = model(x)
velocity_norms.append(v.norm(dim=-1).mean().item())

# Plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.plot(losses); plt.title('Loss')
plt.subplot(1, 3, 2); plt.plot(grad_norms); plt.title('Gradient Norm')
plt.subplot(1, 3, 3); plt.plot(velocity_norms); plt.title('Velocity Norm')
plt.show()
```

### Pathologies

**Symptom**: Loss oscillates wildly  
**Cause**: Using standard Adam instead of RiemannianAdam  
**Fix**: Switch optimizer

**Symptom**: Model diverges after ~100 steps  
**Cause**: Missing velocity normalization or excessive dt  
**Fix**: Ensure v2.5.0+ (has automatic normalization), or reduce dt

**Symptom**: Plateau at ~80% accuracy  
**Cause**: Learning rate too low or insufficient model capacity  
**Fix**: Increase LR or increase dim/depth

---

## Distributed Training

### Data Parallel (Single Node, Multi-GPU)

```python
model = Manifold(...).cuda()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Training proceeds as normal
```

### Distributed Data Parallel (Multi-Node)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Model
model = Manifold(...).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# DataLoader with DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, ...)

# Training
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Shuffle across ranks
    for batch in loader:
        train_step(batch)
```

---

## Memory Optimization

### Mixed Precision (FP16/BF16)

**Status**: Experimental (stability under testing)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        logits, _, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Accumulation

```python
accumulation_steps = 4

for i, (x, y) in enumerate(dataloader):
    logits, _, _ = model(x)
    loss = criterion(...) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        optimizer.zero_grad()
```

---

## Task-Specific Guides

### Binary Reasoning Tasks (Parity, XOR)

```yaml
model:
  dim: 128
  depth: 6
  physics_config:
    embedding:
      type: functional
      mode: binary
    readout:
      type: binary

optimizer:
  lr: 1e-4

training:
  batch_size: 128
  gradient_clip: 0.05
```

### Language Modeling

```yaml
model:
  dim: 512
  depth: 12
  physics_config:
    embedding:
      type: functional  # O(1) vocab scaling
    readout:
      type: standard

optimizer:
  lr: 3e-5  # Lower for language

training:
  batch_size: 16
  gradient_accumulation: 4
```

---

## Checkpointing

### Save Checkpoint

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'config': config,  # Save configuration
}, 'checkpoint.pth')
```

### Load Checkpoint

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
```

---

## Troubleshooting

| **Issue** | **Solution** |
|-----------|-------------|
| Loss oscillates | Use RiemannianAdam |
| Model diverges | Reduce LR or dt |
| Slow convergence | Increase LR or check data |
| OOM | Reduce batch size, enable gradient accumulation |
| Poor generalization | Train longer or increase depth |

---

## Best Practices Summary

✅ **Always use RiemannianAdam**  
✅ **Clip gradients at 0.05**  
✅ **Start with lr=1e-4, tune from there**  
✅ **Use OneCycleLR scheduler**  
✅ **Monitor velocity norms for stability**  
✅ **Save checkpoints frequently**  
✅ **Validate on held-out data**  

---

**Document Version**: 2.6.0  
**Last Updated**: January 20, 2026  
**Status**: Production-Ready
