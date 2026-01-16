# Component Reference

[!VERSION](https://img.shields.io/badge/version-1.0.0-blue.svg)

> Complete technical reference for all Manifold classes and modules.

---

## Table of Contents

1. [Model (`src/model.py`)](#model)
2. [Layers (`src/layers.py`)](#layers)
3. [Geometry (`src/geometry.py`)](#geometry)
4. [Losses (`src/losses.py`)](#losses)
5. [Optimization (`src/optim.py`)](#optimization)
6. [Adjoint (`src/adjoint.py`)](#adjoint)

---

## Model

### `class Manifold`

The main sequence model. Reformulates sequence processing as geodesic flow on a learned Riemannian manifold.

**Location**: `src/model.py`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | required | Size of the token vocabulary |
| `dim` | int | 256 | Hidden state dimension |
| `depth` | int | 4 | Number of M-Layers |
| `rank` | int | 32 | Rank for low-rank Christoffel decomposition |
| `heads` | int | 4 | Number of parallel geodesic heads per layer |
| `integrator_type` | str | 'heun' | Numerical integrator: 'heun', 'rk4', 'leapfrog', 'symplectic', 'rk45' |
| `use_scan` | bool | False | If True, use ParallelMLayer for O(log N) training |
| `physics_config` | dict | None | Configuration for cognitive physics features |

#### Forward Signature

```python
def forward(self, input_ids, attention_mask=None, state=None):
    """
    Args:
        input_ids: [batch, seq_len] Token indices
        attention_mask: [batch, seq_len] Optional mask (1=valid, 0=pad)
        state: Optional tuple (x, v) to continue from previous state

    Returns:
        logits: [batch, seq_len, vocab_size] Output predictions
        state: (x, v) Final state for continuation
        christoffels: List of Christoffel tensor outputs (for regularization)
    """
```

#### Internal Components

- `embedding`: `nn.Embedding(vocab_size, dim)` - Token to force conversion
- `layers`: `nn.ModuleList` of `MLayer` or `ParallelMLayer` or `FractalMLayer`
- `readout_norm`: `nn.LayerNorm(dim)` - Pre-output normalization
- `readout`: `nn.Linear(dim, vocab_size)` - State to logits projection
- `x0`, `v0`: `nn.Parameter` - Learnable initial state (position and velocity)

---

## Layers

### `class MLayer`

Manifold Layer. The core building block that evolves state (x, v) via geodesic flow.

**Location**: `src/layers.py`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | required | Hidden dimension (must be divisible by `heads`) |
| `heads` | int | 4 | Number of independent geodesic subspaces |
| `rank` | int | 16 | Rank per head for Christoffel decomposition |
| `base_dt` | float | 0.1 | Base integration timestep |
| `integrator_type` | str | 'heun' | Numerical integrator to use |
| `physics_config` | dict | None | Cognitive physics configuration |

#### Forward Signature

```python
def forward(self, x, v, force=None, context=None):
    """
    Args:
        x: [batch, dim] Position on manifold
        v: [batch, dim] Velocity (tangent vector)
        force: [batch, dim] External force from input token
        context: [batch, heads] Optional context from previous layer

    Returns:
        x_next: [batch, dim] Updated position
        v_next: [batch, dim] Updated velocity
        context_next: [batch, heads] Context for next layer
        christoffels: List of Γ(v) outputs per head
    """
```

#### Architecture

1. **Pre-LayerNorm**: Normalizes x and v before processing
2. **Head Splitting**: Divides dim into `heads` independent subspaces
3. **Per-Head Integration**: Each head evolves its state via its own integrator
4. **Concatenation**: Reassembles heads
5. **Output Projection**: Mixes information across heads

---

### `class ParallelMLayer`

Parallel Manifold Layer using Associative Scan for O(log N) training.

**Location**: `src/layers.py`

#### Key Difference from MLayer

Instead of sequential integration, this layer linearizes the geodesic flow:

```
dv/dt ≈ A * v + B * force
```

Where A and B are predicted per-timestep. This allows the entire sequence to be processed in parallel using Blelloch's scan algorithm.

#### Constructor Parameters

Same as `MLayer`, but ignores `integrator_type` (uses linearized dynamics).

---

### `class FractalMLayer`

Fractal Manifold Layer with recursive tunneling capability.

**Location**: `src/layers.py`

#### Behavior

When the local curvature exceeds a threshold, this layer activates a nested sub-layer to process the token at higher resolution.

#### Additional Parameters

| Parameter | Type | Source | Description |
|-----------|------|--------|-------------|
| `threshold` | float | `physics_config['fractal']['threshold']` | Curvature trigger |
| `alpha` | float | `physics_config['fractal']['alpha']` | Sub-manifold contribution weight |

#### Forward Logic

```python
curvature = compute_scalar_curvature(x, v)
if curvature > threshold:
    x_sub, v_sub = self.sub_layer(x, v, force)
    x = x + alpha * x_sub
    v = v + alpha * v_sub
return self.main_layer(x, v, force)
```

---

## Geometry

### `class LowRankChristoffel`

Computes Christoffel symbols using a low-rank decomposition.

**Location**: `src/geometry.py`

#### Mathematical Formulation

The Christoffel symbols Γ^k_{ij} define the curvature of the manifold. We use a symmetric low-rank decomposition:

```
Γ(v) = W * (U^T v)²
```

Where:
- `U`: [dim, rank] - Input projection
- `W`: [dim, rank] - Output projection
- The squaring ensures symmetry in lower indices (torsion-free)

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | required | Manifold dimension |
| `rank` | int | 16 | Decomposition rank |
| `physics_config` | dict | None | Configuration |

#### Dynamic Curvature Modulation

If `x` is provided to `forward()`, the curvature is modulated by position:

```
Γ_dynamic = Γ_static * (1 + sigmoid(V^T x))
```

This creates position-dependent "gravity wells".

---

### `class ReactiveChristoffel`

Extends `LowRankChristoffel` with Active Inference features.

**Location**: `src/geometry.py`

#### Additional Features

1. **Plasticity**: Curvature adapts based on velocity magnitude
2. **Singularities**: High-certainty regions create attractor basins
3. **Decay**: Curvature perturbations decay over time

---

### Integrators

All integrators implement the geodesic equation:

```
d²x/dt² + Γ(v, v) = F
```

#### `class HeunIntegrator` (Default)

Heun's Method (Improved Euler / RK2). 2nd order accuracy with 2 function evaluations per step.

**Recommended for**: General use, balanced accuracy/speed.

#### `class RK4Integrator`

Classical Runge-Kutta 4. 4th order accuracy with 4 function evaluations.

**Recommended for**: High accuracy requirements.

#### `class LeapfrogIntegrator`

Störmer-Verlet symplectic integrator. Preserves Hamiltonian structure.

**Recommended for**: Long-term energy conservation.

#### `class SymplecticIntegrator`

Velocity Verlet variant. Similar to Leapfrog with different staging.

#### `class DormandPrinceIntegrator`

Adaptive RK45 with error estimation. Automatically adjusts dt.

**Recommended for**: Variable complexity sequences.

---

## Losses

### `class GFNLoss`

Combined loss function for training Manifold.

**Location**: `src/losses.py`

#### Components

1. **Cross-Entropy Loss**: Standard next-token prediction
2. **Hamiltonian Regularization**: Penalizes energy drift
3. **Entropy Regularization**: Encourages diverse trajectories
4. **Noether Regularization**: Enforces symmetry constraints

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hamiltonian_weight` | float | 0.01 | Weight for energy conservation loss |
| `entropy_weight` | float | 0.001 | Weight for entropy maximization |
| `noether_weight` | float | 0.001 | Weight for symmetry enforcement |

---

## Optimization

### `class AdaHessian`

Second-order optimizer using Hessian diagonal approximation.

**Location**: `src/optim.py`

Provides curvature-aware learning rates without full Hessian computation.

---

## Adjoint

### `class AdjointManifold`

Wrapper for infinite-context training using the Adjoint Sensitivity Method.

**Location**: `src/adjoint.py`

Enables O(1) memory backpropagation through arbitrarily long sequences by recomputing forward states during backward pass.
