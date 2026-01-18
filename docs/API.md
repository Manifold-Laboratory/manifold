# MANIFOLD API Reference

**Version:** 2.5.0 "Riemannian Stability"  
**Last Updated:** January 18, 2026

Complete Python API documentation for the MANIFOLD framework.

---

## Installation

```bash
git clone https://github.com/Manifold-Laboratory/manifold
cd manifold
pip install -e .
```

**Requirements**: PyTorch 2.3+, Python 3.10+

---

## Core Model

### `Manifold`

Main sequence modeling architecture with Riemannian dynamics.

```python
from src.model import Manifold

model = Manifold(
    vocab_size=50257,        # Vocabulary size
    dim=512,                 # Hidden dimension
    depth=12,                # Number of M-Layers
    heads=8,                 # Multi-head geodesic flow
    rank=32,                 # Christoffel rank (compression)
    integrator_type='leapfrog',  # 'leapfrog', 'heun', 'rk4'
    use_scan=False,          # Parallel scan (experimental)
    physics_config=None      # Dict for physics features
)
```

**Parameters**:
- `vocab_size` (int): Size of token vocabulary
- `dim` (int, default=256): Latent manifold dimension
- `depth` (int, default=4): Number of stacked M-Layers
- `heads` (int, default=4): Number of parallel geodesic heads
- `rank` (int, default=32): Low-rank Christoffel approximation
- `integrator_type` (str, default='heun'): Numerical integrator
  - `'leapfrog'`: Symplectic (recommended for speed)
  - `'heun'`: RK2 (good balance)
  - `'rk4'`: Runge-Kutta 4 (high accuracy, slow)
- `use_scan` (bool, default=False): Enable parallel scan (O(log N) training)
- `physics_config` (dict, optional): Advanced configuration (see below)

**Forward Pass**:
```python
# Standard forward
logits, state, christoffels = model(input_ids)

# With previous state (autoregressive)
logits, state, christoffels = model(input_ids, state=prev_state)

# Returns:
#   logits: [batch, seq_len, vocab_size]
#   state: (x, v) tuple - final position/velocity
#   christoffels: List of curvature tensors (for analysis)
```

**Generation**:
```python
prompt = torch.tensor([[1, 2, 3]])  # Token IDs
generated = model.generate(
    prompt_ids=prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=40,       # Optional: nucleus sampling
    top_p=0.9       # Optional: top-p sampling
)
```

### Physics Configuration

```python
physics_config = {
    'embedding': {
        'type': 'functional',  # 'standard', 'implicit', 'functional'
        'mode': 'binary',      # For functional: 'binary' or 'sinusoidal'
        'coord_dim': 16        # Coordinate dimension
    },
    'readout': {
        'type': 'binary'       # 'standard', 'binary', 'implicit'
    },
    'active_inference': {
        'enabled': True,
        'reactive_curvature': {
            'enabled': True,
            'plasticity': 0.05
        }
    },
    'hyper_curvature': {
        'enabled': True        # Dynamic curvature modulation
    },
    'stability': {
        'base_dt': 0.3,        # Integration timestep
        'damping': 0.05,       # (Deprecated - use velocity normalization)
        'residual_scale': 0.5,
        'curvature_clamp': 5.0 # Max curvature magnitude
    }
}

model = Manifold(..., physics_config=physics_config)
```

---

## Geometry Module

### `LowRankChristoffel`

Computes curvature (Christoffel symbols) with low-rank approximation.

```python
from src.geometry import LowRankChristoffel

christoffel = LowRankChristoffel(
    dim=512,
    rank=32,
    physics_config=None
)

# Compute curvature for velocity v at position x
gamma = christoffel(v, x)  # Returns: [batch, dim]
```

**Key Features**:
- Adaptive gating ("The Valve")
- Dynamic curvature modulation
- Saturation for numerical stability

### `SymplecticIntegrator`

Leapfrog (Velocity Verlet) integrator for geodesic equation.

```python
from src.geometry import Symplectic Integrator

integrator = SymplecticIntegrator(
    christoffel_net=christoffel,
    dt=0.1
)

# One integration step
x_next, v_next = integrator(x, v, force=F, dt_scale=1.0)
```

---

## Layers

### `MLayer`

Core Manifold layer (replaces Transformer attention).

```python
from src.layers import MLayer

layer = MLayer(
    dim=512,
    heads=8,
    rank=32,
    integrator_type='leapfrog',
    physics_config=None
)

# Forward pass
x_next, v_next, context, christoffels = layer(x, v, force, context)
```

**Multi-head Processing**:
- Splits (x, v) into `heads` independent geodesic flows
- Each head has independent Christoffel symbols & integrator
- Output heads are mixed via learned projection

### `ParallelMLayer`

Parallel scan variant (experimental, O(log N) training).

```python
from src.layers import ParallelMLayer

layer = ParallelMLayer(dim=512, heads=8)
x_out, v_out, ctx, christoffels = layer(None, None, force=force_sequence)
```

---

## Embeddings

### `FunctionalEmbedding`

Implicit neural field embedding (O(1) w.r.t. vocabulary).

```python
from src.embeddings import FunctionalEmbedding

embedding = FunctionalEmbedding(
    vocab_size=50257,
    emb_dim=512,
    coord_dim=16,
    mode='binary'  # 'binary' or 'sinusoidal'
)

# Forward
emb = embedding(input_ids)  # [batch, seq_len, emb_dim]
```

**Features**:
- SIREN architecture (sin activation)
- Zero-force bias (E(0) = 0)
- Independent of vocab size

---

## Optimization

### `RiemannianAdam`

Geometry-aware Adam optimizer with manifold retraction.

```python
from src.optim import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    retraction='normalize',  # 'euclidean', 'normalize', 'cayley'
    max_norm=10.0
)

# Standard PyTorch usage
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Critical**: Required for Manifold models (standard Adam causes Euclidean drift).

---

## Utilities

### Memory Measurement

```python
from tests.benchmarks.bench_utils import measure_peak_memory

def forward_fn():
    return model(x)

peak_mb = measure_peak_memory(model, forward_fn)
```

### Configuration Loading

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = Manifold(**config['model'])
```

---

## Example: Training Loop

```python
import torch
from src.model import Manifold
from src.optim import RiemannianAdam

# Model
model = Manifold(vocab_size=1000, dim=256, depth=6).cuda()

# Optimizer (MUST use Riemannian)
optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,
    retraction='normalize',
    max_norm=10.0
)

# Loss
criterion = torch.nn.CrossEntropyLoss()

# Training
for batch in dataloader:
    x, y = batch
    x, y = x.cuda(), y.cuda()
    
    optimizer.zero_grad()
    logits, _, _ = model(x)
    loss = criterion(logits.view(-1, 1000), y.view(-1))
    loss.backward()
    
    # Gradient clipping (recommended)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    
    optimizer.step()
```

---

## Example: Autoregressive Generation

```python
# Load trained model
model = Manifold.load('checkpoint.pth')
model.eval()

# Prompt
prompt = "The meaning of life is"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens]).cuda()

# Generate
with torch.no_grad():
    output_ids = model.generate(
        prompt_ids=input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9
    )

# Decode
text = tokenizer.decode(output_ids)
print(text)
```

---

## Advanced: Custom Physics

### Adding Custom Embedding

```python
class MyEmbedding(nn.Module):
    def forward(self, input_ids):
        # Custom logic
        return embeddings  # [batch, seq, dim]

model = Manifold(...)
model.embedding = MyEmbedding()
```

### Accessing Internal State

```python
logits, (x_final, v_final), christoffels = model(x)

# Analyze final state
position_norm = torch.norm(x_final, dim=-1)
velocity_norm = torch.norm(v_final, dim=-1)

# Analyze curvature (per layer)
for layer_idx, gamma in enumerate(christoffels):
    print(f"Layer {layer_idx} curvature: {gamma.abs().mean()}")
```

---

## Troubleshooting

**Issue**: Training loss oscillates chaotically (1.0 → 0.2 → 1.0)  
**Solution**: Use `RiemannianAdam` instead of standard Adam

**Issue**: Model diverges after ~100 steps  
**Solution**: Enable velocity normalization (default in v2.5.0)

**Issue**: Out of memory during training  
**Solution**: Reduce batch size, or use gradient accumulation

**Issue**: Slow training speed  
**Solution**: Enable CUDA kernels (in development), or use `integrator_type='leapfrog'`

---

## API Changelog

**v2.5.0 (Current)**:
- Added `RiemannianAdam` optimizer
- Implemented adaptive curvature gating
- Added zero-force embedding bias
- Velocity normalization (automatic)

**v2.0.0**:
- Functional embeddings
- Binary readout mode
- Multi-head architecture

**v1.0.0**:
- Initial release

---

**Documentation Version**: 2.5.0  
**API Stability**: Beta (breaking changes possible)  
**License**: Apache 2.0

For implementation details, see [SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md).
