# Manifold API Reference

> Complete API documentation for the Manifold framework.

---

## Core Models

### `Manifold`
The primary Geometric Intelligence model.

```python
from src import Manifold

model = Manifold(
    vocab_size: int,
    dim: int = 512,
    depth: int = 24,
    rank: int = 64,
    integrator_type: str = 'leapfrog',
    physics_config: dict = None
)

# Forward pass
logits, (x_final, v_final) = model(input_ids)
```

**Parameters:**
- `integrator_type`: Algorithm for geodesic evolution ('leapfrog', 'symplectic', 'heun').
- `physics_config`: Dictionary enabling Active Inference features (Plasticity, Singularities).

### `AdjointManifold`
Memory-constant ($O(1)$) variant for infinite context windows.

```python
from src import AdjointManifold

model = AdjointManifold(..., integration_time=1.0)
```

---

## Geometry Modules

### `ReactiveChristoffel`
Active metric tensor that adapts to latent energy and semantic potential.

```python
from src.geometry import ReactiveChristoffel

gamma = ReactiveChristoffel(dim, rank, physics_config)
# Returns curvature vector for current velocity
k = gamma(v) 
```

### `TimeDilationHead`
Predicts optimal integration time-step ($\Delta t$) per head.

```python
from src.geometry import TimeDilationHead

head = TimeDilationHead(dim)
dt_scale = head(x, v, force)
```

---

## Manifold Layers

### `MLayer`
A single Geodesic Evolution Layer. Replaces Multi-Head Attention.

```python
from src import MLayer

layer = MLayer(dim, heads=8, integrator_type='leapfrog')
x_next, v_next, context = layer(x, v, force, context)
```

### `RiemannianGating`
Classic curvature-based flow control mechanism.

---

## Optimization

### `RiemannianAdam`
Adam optimizer with manifold retraction steps to ensure weight validity.

```python
from src import RiemannianAdam

opt = RiemannianAdam(model.parameters(), lr=1e-3, retraction='normalize')
```

### `GFNLoss`
Entropic and Hamiltonian loss functions for physical regularization.

```python
from src import GFNLoss

loss_fn = GFNLoss(
    lambda_h=0.01, # Hamiltonian energy conservation
    lambda_c=0.05  # Curiosity temperature (Thermodynamics)
)
```

---
*Manifold Framework Technical Reference*
