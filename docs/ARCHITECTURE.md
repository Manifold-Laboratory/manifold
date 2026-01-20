# MANIFOLD Architecture

**Version:** 2.6.0 "Symplectic Forgetting"  
**Last Updated:** January 20, 2026

System architecture overview and design principles for the MANIFOLD framework.

---

## Overview

## Overview

MANIFOLD is a sequence modeling architecture based on **Second-Order State Space Models (SSMs)**. It replaces discrete attention mechanisms with a continuous dynamical system update.

**Key Properties**:
- **O(1) Memory**: Constant-size state (x, v) regardless of sequence length
- **Symplectic Dynamics**: Volume-preserving integration improves efficient information flow
- **Quadratic Interactions**: Learnable "curvature" layer mixes features multiplicatively
- **Compositional**: Multi-layer hierarchy

---

## System Diagram

```
Input Tokens
     ↓
[Functional Embedding] ← SIREN Neural Field (O(1))
     ↓
     F (External Force)
     ↓
┌────────────────────────┐
│   M-Layer 1            │
│  ┌──────────────────┐  │
│  │ Multi-Head Split │  │
│  └────┬─────────────┘  │
│       ↓                │
│  [Christoffel + Gate]  │ ← Learnable Curvature
│       ↓                │
│  [Symplectic Step]     │ ← Leapfrog Integrator
│       ↓                │
│  [Velocity Norm]       │ ← Stability Layer
│       ↓                │
│  [Head Mixing]         │
│  └────┬────────────────┘
│      (x₁, v₁)          │
└────────┼────────────────┘
         ↓
    ... (repeat L layers)
         ↓
      (x_L, v_L)
         ↓
  [Binary/Standard Readout]
         ↓
      Logits
```

---

## Core Components

### 1. Embedding Layer

**Purpose**: Convert discrete tokens → continuous force vectors

**Implementation**:
```python
class FunctionalEmbedding:
    # Token ID → Binary Coordinates → SIREN → Embedding
    coords = id_to_binary(token_id)  # [16-bit]
    embedding = SIREN_MLP(coords)    # [dim]
```

**Types**:
- `functional`: Neural field (O(1) w.r.t. vocab)
- `implicit`: Learnable coordinate table
- `standard`: Classic lookup table

**Innovation**: Zero-force bias (E(0) = 0) enables inertial coasting.

### 2. M-Layer (Manifold Layer)

**Replaces**: Transformer attention block

**Structure**:
```
Input: (x, v, F, context)
  ↓
Pre-LayerNorm (x only)
  ↓
Split into H heads
  ↓
For each head h:
  1. Compute Γ_h(v_h, x_h)     [Christoffel]
  2. Apply adaptive gate       [Gating]
  3. Integrate: (x', v') = LeapFrog(x, v, F-Γ, dt)
  4. Normalize velocity: v' ← v' / ||v'||
  ↓
Concatenate heads
  ↓
Mix via learned projection
  ↓
Output: (x_next, v_next, context_next)
```

**Parameters**:
- Position x: Semantic location in manifold
- Velocity v: Semantic momentum (memory)
- Force F: Input token embedding
- Context: Inter-layer communication

### 3. Quadratic Interaction Layer (Christoffel)

**Mathematical Form**:
```
Γ(v, x) = W · [(U^T v)² ⊙ sat(||U^T v||)]
```

**Components**:
- U, W ∈ ℝ^(dim×rank): Learnable basis matrices
- sat(·): Soft saturation (numerical stability)
- rank: Low-rank compression (16-64)

**Adaptive Features**:
- **Gating**: `gate(x) ⊙ Γ` (learn when to apply curvature)
- **Modulation**: `(1 + σ(V^T x)) · Γ` (position-dependent scaling)

**Initialization**: Near-zero (flat manifold) for stable training start.

### 4. Symplectic Integrator

**Algorithm**: Velocity Verlet (Leapfrog)

```python
# Acceleration at t (with Friction)
# F_friction = -sigmoid(W_forget·x) * v
a_t = F - Γ(v_t, x_t) + F_friction(v_t, x_t)

# Half-step velocity
v_half = v_t + 0.5 * dt * a_t

# Full-step position
x_{t+1} = x_t + dt * v_half

# Acceleration at t+1
a_{t+1} = F - Γ(v_half, x_{t+1}) + F_friction(v_half, x_{t+1})

# Half-step velocity finalization
v_{t+1} = v_half + 0.5 * dt * a_{t+1}

# Stabilization
v_{t+1} = v_{t+1} / (||v_{t+1}|| + ε)
```

**Properties**:
- Time-reversible
- Volume-preserving (det(J) = 1)
- O(dt³) local error, O(dt²) global error

**Alternatives**:
- RK4: Higher accuracy, slower
- Heun: Good balance

### 5. Multi-Head Architecture

**Parallelism**: H independent geodesic flows

Each head:
- Independent Christoffel symbols Γ_h
- Independent integration parameters dt_h
- Independent gating networks

**Mixing**:
```
x_out = Linear([x₁, x₂, ..., x_H])
v_out = Linear([v₁, v₂, ..., v_H])
```

**Interpretation**: Different heads explore different geometric "hypotheses".

### 6. Readout Layer

**Binary Mode** (for binary tasks):
```python
logits = MLP(x_final)  # → [batch, seq, coord_dim]
# Decode via hard threshold or nearest-neighbor
```

**Standard Mode** (for language):
```python
logits = Linear(x_final)  # → [batch, seq, vocab_size]
```

**Implicit Mode** (experimental):
```python
# Coordinate-to-token inverse mapping
logits = SIREN_inverse(x_final)
```

---

## Data Flow

### Training (Parallel, O(N) memory)

```
Tokens [B, L]
  ↓
Embeddings [B, L, D]
  ↓
Process all L tokens → Logits [B, L, V]
  ↓
Loss = CrossEntropy(Logits, Targets)
```

**Memory**: O(B·L·V) for output logits (unavoidable).

### Inference (Recurrent, O(1) memory)

```
Token_t [B, 1]
  ↓
Embedding_t [B, D]
  ↓
State (x, v) [B, 2D] ← From previous step
  ↓
M-Layers: (x', v') = f(x, v, Embedding_t)
  ↓
Logit_t [B, V]
  ↓
Sample next token
  ↓
Repeat with State = (x', v')
```

**Memory**: O(B·D) (constant).

---

## Optimization Strategy

### 1. Riemannian Manifold Constraints

**Problem**: Learned geometries define implicit constraints (e.g., bounded curvature).

**Solution**: RiemannianAdam with retraction:
```
W_new = Retract(W_old - lr · grad)
```

**Retraction**:
- Normalize: `W / max(||W||, max_norm)`
- Cayley: Orthogonal projection (for square matrices)

### 2. Stability Mechanisms

**Velocity Normalization** (critical):
```python
v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
```
Prevents explosion while preserving direction.

**Curvature Clamping**:
```python
Γ = torch.clamp(Γ, -5.0, 5.0)
```
Prevents singularities.

**Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(params, 0.05)
```
Tighter than standard (0.1-1.0).

---

## Computational Complexity

| **Operation** | **Training** | **Inference** |
|--------------|-------------|--------------|
| Embedding | O(L·d²) | O(d²) |
| Christoffel | O(L·d²·R) | O(d²·R) |
| Integration | O(L·d) | O(d) |
| Readout | O(L·d·V) | O(d·V) |
| **Total** | **O(L·d²·R)** | **O(d²·R)** |

**Comparison**:
- Transformer: O(L²·d) training, O(L·d) inference (+ O(L·d) KV cache)
- Manifold: O(L·d²) training, O(d²) inference (O(1) state)

**Asymptotic Win**: For L > d (long sequences), Manifold is more efficient.

---

## Design Principles

### 1. Physics-First

All operations grounded in differential geometry and Hamiltonian mechanics:
- Geodesic equation (GR)
- Symplectic integration (classical mechanics)
- Liouville's theorem (statistical mechanics)

### 2. Compositionality

Layers stack hierarchically:
- Layer 1: Processes raw tokens
- Layer L: Processes abstract semantic representations
- Each layer refines the manifold geometry

### 3. Memory Efficiency

State compression via geometric encoding:
- Traditional RNN: Stores history in fixed vector
- Manifold: Encodes history in curvature + momentum

### 4. Gradient Stability

Symplectic structure ensures:
- No vanishing (det(J) = 1)
- No exploding (energy conserved)
- Infinite horizon stability

---

## Comparison with Other Architectures

| **Architecture** | **Memory** | **Compute** | **Gradient Stability** | **Inductive Bias** |
|-----------------|-----------|------------|----------------------|-------------------|
| Transformer | O(N) | O(N²) | Good | Permutation invariance |
| LSTM/GRU | O(1) | O(N) | Poor | Sequential |
| Mamba (SSM) | O(1) | O(N) | Medium | State compression |
| **Manifold** | **O(1)** | **O(N·d²)** | **Excellent** | **Geometry** |

---

## Future Extensions

### Mixture of Dynamics (MoM)

Multiple dynamics experts:
- Euclidean (Linear/ResNet behavior)
- Hyperbolic (Tree/Hierarchy specialized)
- Spherical (Cyclic/Rotation specialized)
- Router selects dynamics per token

### CUDA Acceleration

Fused kernel implementation for the Leapfrog step:
```c
__global__ void symplectic_step_fused(
    float* x, float* v, float* F,
    float* U, float* W, float dt
);
```

Expected: 10-50× speedup.

### Neural ODE Integration

Replace discrete layers with continuous depth solvers:
```
d(x,v)/dt = f_θ(x, v, F, t)
```

Adaptive computation per token.

---

**Document Version**: 2.5.0  
**Last Updated**: January 18, 2026  
**Status**: Production Architecture
