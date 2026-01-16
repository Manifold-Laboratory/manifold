# Configuration Reference

> Complete reference for all Manifold configuration options.

---

## Overview

Manifold uses a hierarchical configuration system. The main model parameters control architecture, while the `physics_config` dictionary controls the cognitive dynamics features.

---

## Model Parameters

These are passed directly to the `Manifold` constructor.

```python
model = Manifold(
    vocab_size=50257,    # Vocabulary size
    dim=512,             # Hidden dimension
    depth=12,            # Number of layers
    rank=64,             # Christoffel decomposition rank
    heads=8,             # Geodesic heads per layer
    integrator_type='heun',  # Numerical integrator
    use_scan=False,      # Parallel training mode
    physics_config={...} # Cognitive physics (see below)
)
```

### Parameter Details

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `vocab_size` | int | > 0 | Size of token vocabulary |
| `dim` | int | 64-2048 | Hidden state dimension. Must be divisible by `heads` |
| `depth` | int | 1-48 | Number of stacked M-Layers. Deeper = more reasoning capacity |
| `rank` | int | 4-128 | Rank for Christoffel decomposition. Higher = richer geometry |
| `heads` | int | 1-16 | Independent geodesic subspaces. Higher = more parallelism |
| `integrator_type` | str | see below | Numerical integration method |
| `use_scan` | bool | True/False | If True, uses ParallelMLayer for O(log N) training |

### Integrator Types

| Value | Method | Order | Evaluations | Use Case |
|-------|--------|-------|-------------|----------|
| `'heun'` | Heun's Method | 2 | 2 | **Default**. Good balance |
| `'rk4'` | Runge-Kutta 4 | 4 | 4 | High accuracy |
| `'leapfrog'` | Störmer-Verlet | 2 | 2 | Energy conservation |
| `'symplectic'` | Velocity Verlet | 2 | 2 | Alternative symplectic |
| `'rk45'` | Dormand-Prince | 4-5 | Variable | Adaptive step |

---

## Physics Configuration

The `physics_config` dictionary enables Manifold's cognitive dynamics features.

### Full Schema

```yaml
physics_config:
  # === Active Inference (v0.5) ===
  active_inference:
    enabled: bool           # Master switch for all active features
    
    # Reactive Curvature (Plasticity)
    reactive_curvature:
      enabled: bool         # Curvature adapts to velocity
      plasticity: float     # Adaptation rate [0.0, 1.0]
      decay: float          # Decay rate [0.9, 0.99]
    
    # Logical Singularities (Semantic Attractors)
    singularities:
      enabled: bool         # High-certainty gravity wells
      strength: float       # Attractor strength [1.0, 20.0]
      threshold: float      # Activation threshold [0.5, 0.9]
    
    # Dynamic Time (Auto-Wormholes)
    dynamic_time:
      enabled: bool         # Adaptive integration timestep
      range: [float, float] # [min_dt, max_dt], e.g. [0.1, 5.0]
    
    # Recursive Geodesics (Metacognition)
    recursive_geodesics:
      enabled: bool         # Layer-to-layer steering

  # === Semantic Symmetries (v0.7) ===
  symmetries:
    enabled: bool           # Noether invariance
    isomeric_groups: list   # Heads sharing geometry, e.g. [[0,1], [2,3]]

  # === Fractal Manifolds (v0.8) ===
  fractal:
    enabled: bool           # Recursive tunneling
    threshold: float        # Curvature trigger [0.3, 0.7]
    alpha: float            # Sub-manifold weight [0.1, 0.5]

  # === Thermodynamics (v0.6) ===
  thermodynamics:
    temperature: float      # Entropy weight in loss [0.0, 1.0]

  # === Stability ===
  stability:
    curvature_clamp: float  # Max |Γ| value [1.0, 20.0]
```

---

## Feature Details

### Active Inference

When `active_inference.enabled = True`, the manifold geometry becomes dynamic:

1. **Reactive Curvature**: High velocity (uncertainty) → increased curvature → slower flow
2. **Singularities**: Confident predictions create attractor basins
3. **Dynamic Time**: Model predicts optimal dt per head/step
4. **Recursive Geodesics**: Previous layer context steers current layer

### Semantic Symmetries

When `symmetries.enabled = True`:

- Heads in the same `isomeric_group` share identical Christoffel symbols
- Enforces that "the same logic applies to similar problems"
- Reduces parameters and improves generalization

### Fractal Manifolds

When `fractal.enabled = True`:

- Uses `FractalMLayer` instead of `MLayer`
- High-curvature tokens trigger recursive sub-layer processing
- Allows variable compute per token

### Thermodynamics

When `thermodynamics.temperature > 0`:

- Training loss includes entropy maximization term
- Prevents collapse to deterministic trajectories
- Encourages exploration of geometric capacity

---

## Example Configurations

### Minimal (Fast Training)

```python
physics_config = None  # All features disabled
```

### Standard (Balanced)

```python
physics_config = {
    'active_inference': {
        'enabled': True,
        'reactive_curvature': {'enabled': True, 'plasticity': 0.1},
        'singularities': {'enabled': True, 'strength': 5.0},
        'dynamic_time': {'enabled': True, 'range': [0.1, 3.0]}
    },
    'stability': {'curvature_clamp': 5.0}
}
```

### Full (Research)

```python
physics_config = {
    'active_inference': {
        'enabled': True,
        'reactive_curvature': {'enabled': True, 'plasticity': 0.15, 'decay': 0.95},
        'singularities': {'enabled': True, 'strength': 10.0, 'threshold': 0.8},
        'dynamic_time': {'enabled': True, 'range': [0.1, 5.0]},
        'recursive_geodesics': {'enabled': True}
    },
    'symmetries': {
        'enabled': True,
        'isomeric_groups': [[0, 1], [2, 3], [4, 5], [6, 7]]
    },
    'fractal': {
        'enabled': True,
        'threshold': 0.5,
        'alpha': 0.2
    },
    'thermodynamics': {'temperature': 0.01},
    'stability': {'curvature_clamp': 10.0}
}
```

---

## YAML Configuration Files

Manifold supports YAML configuration files in `configs/model/`.

### Example: `gfn_medium.yaml`

```yaml
model:
  vocab_size: 16
  dim: 512
  depth: 12
  rank: 64
  heads: 8
  integrator: "leapfrog"
  use_scan: false

  physics:
    active_inference:
      enabled: true
      reactive_curvature:
        enabled: true
        plasticity: 0.1
      singularities:
        enabled: true
        strength: 10.0
    symmetries:
      enabled: true
      isomeric_groups: [[0, 1], [2, 3]]
    fractal:
      enabled: true
      threshold: 0.5
    stability:
      curvature_clamp: 10.0
```
