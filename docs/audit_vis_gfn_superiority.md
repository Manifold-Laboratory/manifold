# Audit: vis_gfn_superiority.py

## Overview
This script is the primary benchmarking dashboard for the GFN vs Transformer comparison. It specifically targets the **Parity Task** (sequential modulo-2 sum) to evaluate state-tracking capabilities.

## Data Collection & Telemetry

### 1. Training Logic (Manifold)
- **Optimizer**: `RiemannianAdam` with $LR=1e-4$ and $WD=1e-4$.
- **Scheduler**: `OneCycleLR`.
- **Loss Composition**:
  - **MSE Loss**: Penalizes prediction error on the primary coordinate ($L_{MSE} = ||logits[0] - target[0]||^2$).
  - **Physics Loss ($L_g$)**: `geodesic_regularization` ($\lambda=0.001$). Penalizes high curvature.
  - **Hamiltonian Loss ($L_h$ )**: `hamiltonian_loss` ($\lambda=0.01$). Penalizes changes in kinetic energy ($||v||^2$) over time.
- **Gradient Clipping**: Strict clipping at **0.05** to maintain stability in curved manifolds.

### 2. Implicit Readout Target Alignment
The dashboard uses a geometric mapping for binary targets:
- Targets are mapped from $\{0, 1\}$ to $\{-1, 1\}$ space.
- Mapping: `target_coords = target_bits.float() * 2 - 1`.
- This ensures that the manifold learning occurs in a zero-centered semantic space.

### 3. Inference & Scaling
- **O(1) Streaming**: The manifold model is evaluated token-by-token in evaluation mode (`L=20` to `L=2000`).
- **Memory Tracking**: Uses `PerformanceStats.measure_peak_memory` to prove O(1) memory complexity vs the Transformer's $O(N^2)$ or $O(N)$ with KV Cache.

## Observations
- The model requested `return_velocities=True` to compute the Hamiltonian loss.
- The `logits[:, :, 0]` selection indicates that the parity information is expected to be encoded in the **first dimension** of the embedding space.
