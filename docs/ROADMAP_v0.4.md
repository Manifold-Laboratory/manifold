# Roadmap v0.4.0: "The Scale & Stability Update"

This document outlines the proposed features for the next major version of MANIFOLD, focusing on numerical precision, architectural specialization, and computational efficiency.

## 1. Golden Integration (Adaptive Physics)
**Goal:** Eliminate numerical instability without using artificial "clamping".

### Concept
Currently, we use fixed `dt` (time steps) and `torch.clamp` to prevent exploding gradients. This is "Video Game Physics". Real physics uses **Adaptive Integrators**.
- **Implementation:** Implement **Dormand-Prince (RK45)** or similar adaptive schemes.
- **Mechanism:** The model calculates an error estimate for each step. If error > threshold, it rejects the step and retries with `dt/2`.
- **Benefit:** The model naturally "slows down" time when traversing complex semantic regions (high curvature), preventing "tunneling" or explosion, and speeds up in flat regions.

## 2. Manifold of Experts (MoM)
**Goal:** Specialization of geometry for different data types.

### Concept
Extend the "Multi-Head" concept to a **Mixture of Manifolds**. Instead of forcing all data onto a single type of geometry, we provide a dictionary of spaces.
- **Structure:** 
    - **Manifold A (Euclidean):** Good for linear/arithmetic data.
    - **Manifold B (Hyperbolic):** Good for hierarchical/tree-like data (syntax, code).
    - **Manifold C (Spherical):** Good for cyclical/rotational data.
- **Routing:** A "Router" network determines for each token (or sequence chunk) which Manifold handles the state evolution.
- **Integration:** This replaces the standard `MLayer` with a `MoMLayer` (Sparse Manifold Layer).

## 3. Fused CUDA Kernels (Production Speed)
**Goal:** Enable training on large-scale datasets.

### Concept
Transition from PyTorch-native implementation to custom CUDA kernels as the default backend.
- **Kernels:**
    - `fused_christoffel`: Compute $\Gamma$ and $a = F - \Gamma$ in one pass without realizing intermediates.
    - `fused_leapfrog`: Perform integration steps entirely in registers.
- **Benefit:** 5x-10x speedup by reducing HBM (memory) read/writes. Critical for scaling beyond "toy" problems.
