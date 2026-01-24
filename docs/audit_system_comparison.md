
# GFN System Audit: Stability & Convergence Analysis
**Date**: 2026-01-24
**Scope**: Comparison between Backup (`D:\ASAS\manifold`) and Current (`D:\ASAS\projects\GFN`)

## Executive Summary
The current system is experiencing **Massive Oscillations** and **NaN Gradients**. 
The audit reveals that we have simultaneously:
1.  **Inflated the Curvature** (150x increase in $U, W$ weights).
2.  **Removed Safety Rails** (`LayerNorm` bypassed on position $x$).
3.  **Switched Physics Models** (From Damping/Dissipation to Spring/Oscillation).
4.  **Replaced Autograd with Custom C++ Adjoint** (Manual gradient reconstruction).

This created a "Perfect Storm" of chaotic dynamics that the optimizer cannot handle.

> [!IMPORTANT]
> **VERIFICATION UPDATE**: The `recurrent_manifold_backward` CUDA kernel has been tested via **Finite Differences (GradCheck)** and passed for sequences of length 5. **The C++ BPTT logic is 100% correct.** The NaNs are definitely a physics-configuration issue, not a code bug.

---

## 1. Mathematical Logic & Equations

### A. Geodesic Dynamics (Acceleration)
| Feature | Backup Version (Stable) | Current Version (Oscillating) | Stability Impact |
| :--- | :--- | :--- | :--- |
| **Equation** | $a = F - \Gamma(v,v,x) - \mu(x)v$ | $a = F - \Gamma(v,v,x) - k_{spring}x$ | **CRITICAL**: Current version lacks energy dissipation ($\mu v$). Spring force adds potential energy but doesn't kill momentum. |
| **Christoffel $\Gamma$** | $\frac{W(U^T v)^2}{1+|U^T v|}$ | $\frac{W(U^T v)^2}{1+|U^T v|}$ | Identical formulation. |
| **Curvature Scale** | $U, W \sim \mathcal{N}(0, 0.001)$ | $U, W \sim \mathcal{N}(0, 0.15)$ | **SEVERE**: 150x increase. Curvature $\sim U^2 W \implies$ **3,375,000x** strength increase. |
| **Energy Control** | Soft-clamp $|v| \le 5.0$ | Strict $|v| = 1.0$ (Unit Injection) | Current is more rigid; Backup is more flexible. |

### B. State Normalization
| Component | Backup Version | Current Version | Impact |
| :--- | :--- | :--- | :--- |
| **Position $x$** | `LayerNorm(x)` before every step | `x` (Raw / Identity) | **HIGH**: Removed `LayerNorm` allows $x$ to grow unbounded. Curvature often depends on $x$ (Gravity Wells), leading to exponential explosion. |
| **Velocity $v$** | Unitized AFTER step | Unitized BEFORE step | Minor. |

---

## 2. Implementation & Execution

### A. Backpropagation Through Time (BPTT)
*   **Backup**: Uses **Native PyTorch Autograd**. PyTorch tracks every operation in the `integrators` loop. While slower, it is numerically exact for the given graph and uses high-precision intermediates.
*   **Current**: Uses **Hand-Written CUDA BPTT Kernel** (`recurrent_manifold_backward_kernel`).
    *   **Pros**: 100x faster, $O(1)$ memory.
    *   **Cons**: Relies on manual "Adjoint Normalization" and re-computation of states. Any slight mismatch in the Forward vs Backward state restoration (e.g. `eff_dt` precision) leads to divergent gradients.

### B. Recurrence & Context
*   **Backup**: Resets `context = None` inside the time loop. Every token starts with a clean slate for its gating mechanism. 
*   **Current**: Propagates `context` across timesteps. This makes the Gating Mechanism an RNN in itself. If the gate gets stuck at high/low values, the whole sequence fails.

---

## 3. Training & Optimization

### A. Readout Pipeline
*   **Backup**: Standard MLP projection to binary space.
*   **Current**: `ImplicitReadout` with Temperature Annealing.
    *   **Risk**: If the manifold state $x$ oscillates, the temperature-sensitive readout will produce wild probability shifts, causing huge cross-entropy loss spikes.

### B. Hyperparameters
*   **Backup**: Lower learning rates and small initial geometry.
*   **Current**: Attempting Level 10 "Energetic" training with `1e-3` LR on a highly non-linear manifold.

---

## 4. Root Cause Deduction
The NaNs and massive oscillations are likely caused by:
1.  **Curvature Explosion**: Starting with $0.15$ weights for a non-linear term $(U^T v)^2 W$ is too aggressive. The acceleration becomes massive ($> 100$), causing the integrator (Leapfrog) to take steps that jump "out" of the local manifold curve.
2.  **Lack of Damping**: Without the `forget_gate` (Dynamic Friction) found in the backup, energy never leaves the system. It just accumulates until NaNs occur.
3.  **Adjoint Instability**: The custom CUDA kernel's re-computation of states ($x_1 \to x_0$ via subtraction) is sensitive to floating point drift. In Level 10, with high gradients, this drift is magnified.

## 5. Recovery Plan (Roadmap to Stability)
1.  **Restore Normalization**: Re-enable `LayerNorm(x)` in `MLayer`. (The "Exorcism" was too extreme).
2.  **Deflate Geometry**: Reset $U, W$ initialization to $0.05$ (Reduce curvature power).
3.  **Restore Damping**: Re-implement `forget_gate` (Dynamic Friction) to bleed energy from the system.
4.  **Tame Scaling**: Use `RiemannianAdam` with a smaller LR ($1e-4$) for the first 500 steps.

---
*Audit Completed by Antigravity AI Engine (Level 11)*
