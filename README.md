# Manifold
> **Geometric Intelligence via Symplectic Geodesic Flows.**

[!VERSION](https://img.shields.io/badge/version-1.0.0-blue.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/physics-Symplectic-purple.svg)](docs/PHYSICS.md)
[![Documentation](https://img.shields.io/badge/docs-latest-orange.svg)](docs/API.md)

---

## Overview

**Manifold** is a fundamental reimagining of neural sequence modeling. Instead of relying on static attention matrices ($O(N^2)$) or unstable recurrent states, Manifold formulates intelligence as **Optimal Transport on a dynamic Riemannian Manifold**. It treats the hidden state as a physical particle traversing a learned geometry, governed by strictly energy-conserving **Symplectic Integrators**, enabling infinite context windows with constant memory.

---

## Core Idea

The central hypothesis of Manifold is that "reasoning" is geometric traversal. By learning a **Metric Tensor** $g_{\mu\nu}(x)$ that warps space in response to semantic density, the model naturally forms "gravity wells" around logical certainties and "expands space" (time dilation) around ambiguities. This allows the network to solve complex long-range dependencies not by "attending" to the past, but by evolving a state that physically conserves the information momentum required to solve the task.

---

## Loss Landscape Analysis

![Loss Landscape 3D](tests/benchmarks/results/loss_landscape/loss_landscape_3d_comparison.png)

The visualization above compares the optimization topology of Manifold (Left) versus a standard GRU/LSTM Baseline (Right). The Z-axis represents the loss value.

**Manifold** exhibits a remarkably **convex and smooth** landscape. Because the flow is Symplectic (volume-preserving), gradients flow through the system without exploding or vanishing, creating a global funnel that leads directly to the optimum.
**The Baseline**, in contrast, shows a "chaotic" landscape riddled with sharp local minima and high-frequency noise, explaining why standard RNNs struggle to converge on long-horizon tasks.

---

## Optimization Geometry

![Loss Landscape Contours](tests/benchmarks/results/loss_landscape/loss_landscape_contours.png)

This contour map view reveals the "stability basin" of the architecture.

- **Manifold (Left)**: The concentric rings indicate a well-conditioned Hamiltonian system. Perturbations in the weights result in proportional changes in loss, making training robust to hyperparameter variance.
- **Baseline (Right)**: The distorted, non-convex regions create optimization barriers. This requires ad-hoc fixes like gradient clipping or specific initialization to navigate, whereas Manifold is stable by design.

---

## Generalization Performance

![Parity Generalization](tests/benchmarks/results/gfn_superiority/parity_generalization.png)

We tested **Zero-Shot Generalization** on the Parity Task, a notoriously difficult problem for RNNs requiring infinite precision memory.

- **Blue Line (Manifold)**: Maintains near-100% accuracy even as sequence length extends far beyond the training distribution (Out-Of-Distribution). The symplectic state conservation means the "parity bit" information never decays.
- **Red Line (MicroGPT/Standard)**: Performance collapses immediately once the sequence length exceeds the training window. The model failed to learn the *algorithm* and merely memorized the *pattern*.

---

## Memory & Scaling

![VRAM Scaling](tests/benchmarks/results/long_context/vram_vs_context.png)

The "Log-Log" plot above demonstrates the "Infinite Context" breakthrough.

- **Manifold (Blue)**: The VRAM usage is a perfectly horizontal line. Whether processing 128 tokens or 1 million tokens, the memory state size ($dim \times 2$) remains constant.
- **Transfomer (Orange)**: The $O(N^2)$ Attention Matrix causes memory to explode exponentially. At ~32k tokens, even efficient Transformers run Out-Of-Memory (OOM) on consumer hardware.

$$ \text{Manifold Memory} \propto O(1) \quad \text{vs} \quad \text{Transformer Memory} \propto O(N^2) $$

---

## Architectural Properties

Manifold integrates five distinctive "Cognitive Physics" components:

1.  **Reactive Curvature ($\Gamma$)**: The manifold stiffens (high curvature) when uncertainty is high, effectively slowing down the "subjective time" of the token to allow for deeper processing.
2.  **Logical Singularities**: High-confidence predictions act as energetic attractors (Black Holes), locking the trajectory into a semantic decision.
3.  **Fractal Tunneling**: The state-space is recursive. Complex tokens trigger a "zoom" into a sub-manifold, allowing the model to allocate hierarchical compute density.
4.  **Noether Invariance**: The architecture enforces symmetry constraints, ensuring that logical rules learned in one context apply universally (Generalization).
5.  **Symplectic Integration**: The Hamiltonian (Energy) of the system is preserved, preventing the catastrophic forgetting common in long sequences.

---

## Comparison Summary

Compared to the current state-of-the-art:

*   **Vs Transformers**: Manifold offers **Infinite Context** and **Constant Memory**, whereas Transformers are limited by context window size and quadratic compute.
*   **Vs RWKV / Mamba**: While these are also efficient RNNs, Manifold is the only one based on **Symplectic Geometry**, offering superior numerical stability and a convex loss landscape for easier training.
*   **Vs LSTM/GRU**: Manifold eliminates the vanishing gradient problem entirely via the Adjoint Sensitivity method and provides strictly better generalization.

---

## Use Cases

-   **Long-Document Analysis**: Processing entire books or legal repositories in a single pass without "chunking".
-   **Robotics & Control**: The continuous-time physics engine makes it ideal for real-world continuous data streams.
-   **Scientific Modeling**: Predicting chaotic systems (weather, fluid dynamics) where conservation laws must be respected.
-   **Edge AI**: Running high-intelligence models on devices with extremely limited RAM (e.g., 4GB or less).

---

## Development Maturity

**Manifold v1.0.0** has reached a stable production milestone. The core **Symplectic Engine** and **Active Inference** modules have been rigorously verified against standard baselines, demonstrating the predicted **O(1) memory scaling** and **numerical stability** in reproducible benchmarks. The kernel backend (Fused CUDA) is fully optimized for NVIDIA Turing/Ampere architectures.

---

## Research Trajectory

The Manifold Laboratory is currently focused on scaling geometric intelligence to the billion-parameter regime.

1.  **Hyperscale Pre-training**: Validating the physics engine's loss convergence properties at 1B+ parameters on the Pile dataset.
2.  **Multi-Manifold MoE**: Developing a "Mixture of Geometries" architecture where different expert heads operate on topologically distinct manifolds (e.g., Hyperbolic for hierarchy, Euclidean for logic).
3.  **Native Multimodal Flows**: extending the geodesic formalism to continuous data streams (Audio/Video), treating them as unrolling surfaces rather than discrete tokens.
4.  **Hardware-Native Symplectic Logic**: Designing custom FPGA/ASIC kernels that enforce energy conservation at the circuit level.

---

## Repository Structure

```text
/
├── src/                # Core Manifold Source Code
│   ├── model.py        # The Main Architecture
│   ├── geometry.py     # Riemannian Metric & Curvature
│   └── physics.py      # Symplectic Integrators
├── docs/               # Deep Technical Documentation
│   ├── PHYSICS.md      # Mathematical Derivations
│   ├── BENCHMARKS.md   # Full Performance Reports
│   └── API.md          # Developer Verification
├── tests/              # Verification Suite
│   └── benchmarks/     # Reproducible Science Scripts
└── LICENSE             # Apache 2.0
```

---

<div align="center">
  <b>Manifold Laboratory</b><br>
  <i>Forging the physics of intelligence.</i>
</div>
