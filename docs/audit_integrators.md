# Audit: Numerical Integrators

## Overview
The `gfn/integrators` module contains the physics engine that evolves the particle/thought trajectory over time. It supports multiple integration schemes tailored for different precision and speed requirements.

## Category Analysis

### 1. Symplectic Integrators (Energy Preserving)
- **Leapfrog / Verlet**: 2nd order. Reversible and preserves the Hamiltonian phase space.
- **Yoshida**: 4th order. Uses specific coefficients to cancel lower-order errors, achieving high accuracy while remaining strictly symplectic.
- **Forest-Ruth / Omelyan**: 4th order variants optimized for distinct force/drift characteristics.
- **Consistency**: The `recurrent_manifold_fused.cu` kernel utilizes a 1st-order Euler-Symplectic step for speed, which matches the base behavior of the Leapfrog integrator when $dt$ is small.

### 2. Runge-Kutta Integrators (High Precision)
- **Heun**: 2nd order. Used as the default stable integrator in `MLayer`.
- **RK4**: 4th order. Standard high-fidelity integration.
- **Dormand-Prince (DP5)**: 5th order. Implemented for "Golden Validation" to ensure physical models don't drift during evaluation.

### 3. Advanced Flows
- **Neural Integrator**: Uses a `controller` MLP to predict optimal $dt$ based on state $[x, v, F]$. This allows for "Automatic Wormholes" where the system speeds up in flat regions and slows down in complex curvatures.
- **Coupling Flow Integrator**: Based on Normalizing Flows (NICE). Guarantees a Jacobian Determinant of exactly 1.0, ensuring perfect geometric volume preservation in the latent space.

## Mathematical Consistency
- **Force Eval**: All integrators correctly call `christoffel(v, x)` to retrieve the geometric resistance.
- **State Update**: Updates match the Hamilton equations of motion: $\dot{x} = v, \dot{v} = a$.
- **CUDA Fusion**: The high-order integrators (Yoshida, DP5, etc.) have mirrored CUDA implementations in `src/integrators/*.cu`, which are correctly hooked to the Python classes via `gfn.cuda.ops`.

## Conclusion
The integrators module is robust and physically compliant. The variety of integrators allows the GFN to balance the speed-accuracy trade-off during different phases of research and production.
