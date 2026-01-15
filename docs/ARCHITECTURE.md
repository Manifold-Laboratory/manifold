# Manifold Architecture

> Deep dive into the Geometric Intelligence architecture.

---

## Core Concept

Manifold models sequences as **geodesic flows on a Riemannian manifold**.

```
Input:       Token → Force → Geodesic Flow → Position → Token
Training:    Input(Force) → Parallel Associative Scan → State Sequence (O(log N))
Inference:   Token → Sequential Geodesic Flow → State (O(1) Step)
```

---

## Mathematical Foundation

### State Variables
- **x**: Position on the manifold (hidden state)
- **v**: Velocity (tangent vector, rate of change)

### Geodesic Equation
$$\frac{d^2 x^k}{dt^2} + \Gamma^k_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt} = F^k$$

Where:
- $\Gamma^k_{ij}$: Christoffel symbols (curvature)
- $F^k$: External force (input token embedding)

---

## Architecture Design
 
 ```mermaid
 graph TD
     Force[Token Force] -->|Split| Heads
     
     subgraph Multi-Head Layer
         direction LR
         Head1[Head 1: Subspace Flow]
         Head2[Head 2: Subspace Flow]
         HeadN[Head N... ]
         
         Heads --> Head1
         Heads --> Head2
         Heads --> HeadN
         
         Head1 --> Concat
         Head2 --> Concat
         HeadN --> Concat
     end
     
     Concat --> Mix[Mixing Projection]
     Mix --> Norm[Pre-LayerNorm]
     Norm --> Next[Next Layer]
 ```
 
 ### Multi-Head Geodesic Flows
 
 Corresponding to attention subspaces in transformers, Manifold computes **parallel geodesic flows** on independent Riemannian sub-manifolds.
 
 - **Parallelism:** Splitting `dim` into `K` heads allows the model to learn `K` distinct geometries simultaneously (e.g., syntax vs. semantics).
 
 ### Pre-LayerNorm Design
 
 Consistent with modern large-scale practices, Manifold applies LayerNormalization **before** the geodesic evolution to ensure numerical stability and gradient health in deep networks.
 
 ```python
 x_norm, v_norm = ln(x), ln(v)
 x_heads = split(x_norm)
 # ... integrate ...
x_out = proj(concat(x_heads))
```

### Parallel Associative Scan

To enable massive parallel training on GPUs, Manifold utilizes a "Linearized Geodesic Flow" mode.
- **Linearization**: The network predicts $A_t$ (decay/rotation) and $B_t$ (input modulation) for all timesteps in parallel.
- **Scan**: A recursive doubling algorithm computes the prefix sum of states in $O(\log N)$ time.
 ```

---

## Complexity Analysis

| Model | Time | Memory | Context |
|-------|------|--------|---------|
| Transformer | O(N²) | O(N²) | Limited by attention |
| Mamba/SSM | O(N) | O(1) | Linear compression |
| **Manifold** | **O(log N)** | **O(1)** | Geodesic memory |

Manifold achieves O(1) memory by encoding information in the continuous trajectory of the state (x, v) rather than storing an explicit memory matrix.

---

## Cognitive Physics Features

### Dynamic Curvature Fields
Information transport is governed by position-dependent curvature:
$$\Gamma(v, x) = \Gamma_{static}(v, v) \cdot (1 + \sigma(V \cdot x))$$
This creates **attractors** where specific semantic concepts warp the geometry, modulating the flow of information through the network.

### Manifold Wormholes
To enable long-range transport, Manifold utilizes **Multi-Scale Heads**.
- **Fast Heads**: Handle local syntactic structures.
- **Slow Heads**: Transport semantic information across long temporal distances in a single effective integration step.

### Entropy-Driven Curiosity
The training action incorporates a thermodynamic regularization term:
$$ L = L_{task} - T \cdot S(\dot{x}) $$
By maximizing the differential entropy ($S$) of the flow, the model is physically encouraged to explore diverse cognitive trajectories, preventing representation collapse.

---

## Component Details

### 1. Embedding Layer
Standard token embedding that acts as "force" on the manifold.

```python
force = self.embedding(token)  # [batch, dim]
```

### 2. Christoffel Network (Low-Rank)
Computes curvature using efficient decomposition:

$$\Gamma(v, v) = W \cdot (U^T v)^2$$

Parameters:
- U: [dim, rank] - Projection basis
- W: [dim, rank] - Output weights

This reduces O(dim³) to O(dim × rank).

### 3. Integrators
Numerically solve the geodesic ODE:

| Integrator | Formula | Properties |
|------------|---------|------------|
| Heun | x' = x + dt/2 (v + v') | Fast, drifts |
| RK4 | 4th order Runge-Kutta | Accurate, slow |
| Leapfrog | v₁/₂ = v + dt/2 a, x' = x + dt v₁/₂ | **Symplectic** |

### 4. Gating Mechanism
Learned curvature-based flow control:

```python
gate = sigmoid(curvature_net(x))  # [0, 1]
x_out = x + gate * (x_new - x)
v_out = v + gate * (v_new - v)
```

High curvature → small steps (gate ≈ 0)
Low curvature → large steps (gate ≈ 1)

### 5. Readout
Project final position to vocabulary:

```python
logits = linear(layer_norm(x))
```

---

## Parameter Count

$$P = V \cdot D + L \cdot (3 \cdot D \cdot R + 2 \cdot R \cdot D) + D \cdot V$$

Where:
- V: vocab_size
- D: dim
- L: depth
- R: rank

Example (gfn_medium):
- V=16, D=512, L=12, R=128
- P ≈ 13M parameters

---

## Training Dynamics

1.  **Token arrives** → Force applied to manifold
2.  **State evolves** → Geodesic flow through layers
3.  **Readout** → Position decoded to prediction
4.  **Loss computed** → CE + Hamiltonian regularization
5.  **Gradients flow** → Through Riemannian optimizer

The key insight: **Reasoning = Trajectory on Manifold**

---

<div align="center">
  <b>Manifold Research Series</b><br>
  Reasoning = Trajectory. Exploration = Entropy.
</div>
