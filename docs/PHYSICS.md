# Cognitive Dynamics Engine

> The theoretical framework behind Manifold's active inference capabilities.

---

## 1. Overview

Manifold diverges from traditional deep learning by treating the latent space not as a static vector space, but as a **dynamic Riemannian manifold** ($M, g$). The geometry of this manifold evolves in real-time, reacting to the semantic content of the sequence. This forms a closed-loop "Cognitive Physics" system.

## 2. Dynamic Curvature Field

The core innovation is the **Reactive Metric Tensor** $g_{\mu\nu}(x, v)$, which governs the "difficulty" of traversing the thought space.

### 2.1 Reactive Plasticity (Uncertainty Regulation)

When the model encounters ambiguity or high information density (high kinetic energy in the latent flow), the manifold stiffens.

$$
\Gamma^k_{ij} \leftarrow \Gamma^k_{ij} \cdot (1 + \alpha \tanh(\|v\|^2))
$$

- **Effect**: High velocity (confusion) $\to$ High curvature $\to$ Increased effective path length.
- **Cognitive Analog**: "Thinking harder" about difficult concepts by slowing down the subjective time-flow.

### 2.2 Logical Singularities (Semantic Anchoring)

Certainty in decision-making is modeled as a gravitational collapse. When the semantic potential $V(x)$ exceeds a critical threshold, a **Singularity** forms.

- **Mechanism**: A localized region of near-infinite curvature acts as an attractor.
- **Effect**: The geodesic trajectory is forcibly captured by the attractor, effectively "locking in" a logical decision.
- **Stability**: Unlike standard RNN attractors, these are symplectic and energy-conserving, preventing gradient explosion.

## 3. Autonomous Geometric Attention (Time Dilation)

In standard Transformers, every token is processed for a fixed computational depth. Manifold introduces **Adaptive Time-Integration**.

Each processing head $h$ independently predicts a time-dilation factor $\Delta t_h$:

$$
\Delta t_h = \sigma(W \cdot [x, v, F]) \cdot \Delta t_{base}
$$

- **Variable Compute**: The model can spend "more time" (integrate longer) on complex tokens and "less time" (skip) on trivial ones.
- **Multi-Scale Flow**: Different heads can operate at different timescales, allowing simultaneous processing of syntax (fast) and semantics (slow).

## 4. Recursive Geodesics (Metacognition)

Layer interaction in Manifold is modeled as a hierarchical control system.

$$
F_{layer \ l+1} = F_{ext} + \mathcal{P}(\text{Context}_{l})
$$

The "Context" from layer $l$ (representing its curvature state) is projected as an external *force* into layer $l+1$. This allows earlier layers to "steer" deeper layers, correcting trajectories before they diverge.

---

## 5. Symplectic Integration

To support these complex dynamics without numerical instability, Manifold employs **Symplectic Integrators** (Störmer-Verlet / Leapfrog).

- **Energy Conservation**: The Hamiltonian $H(x, v)$ is preserved to within machine precision.
- **Reversibility**: The flow is bijective, preventing information loss.
- **Long-term Stability**: Gradients do not vanish even over infinite sequence lengths (via Adjoint method).

This physics-first approach eliminates the need for ad-hoc regularizers like Dropout or aggressive LayerNorm, as stability is intrinsic to the geometry.
