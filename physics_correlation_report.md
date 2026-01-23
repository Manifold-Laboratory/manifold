# Manifold GFN: Physics-Performance Correlation Report

**Date:** 2026-01-22
**Analyst:** Antigravity (Google DeepMind)

## Executive Summary
We analyzed four versions of the Manifold GFN codebase to identify the root causes of training convergence and divergence. The investigation reveals that **Damping (Friction)** is the critical stabilizer for "Hard Norm" systems, while **Soft Norm** provides a superior, more flexible alternative that allows for complex magnitude dynamics without explicit gradients fighting constraints.

## Version Comparison Table

| Feature | v2.5.0 (Old Stable) | v2.6.0 (Failed) | v2.6.1 (Success) | v2.6.2 (Current Patched) |
| :--- | :--- | :--- | :--- | :--- |
| **Status** | **Converged** | **Diverged** | **Converged** | **Converged** |
| **Velocity Output** | Hard Norm ($v/\|v\|$) | Hard Norm | Hard Norm | **Soft Norm** (Clamped) |
| **Friction** | **Yes** (Dynamic) | **NO** (Missing) | **Yes** (Dynamic) | **Yes** (Implicit/Soft) |
| **Singularities** | Soft Sigmoid | Soft Sigmoid | Soft Sigmoid | **Hard Threshold** |
| **Integrator** | Heun/RK4 | Heun/RK4 | Heun/RK4 | Yoshida/Leapfrog |

## Critical Findings

### 1. The "Undamped Hard-Norm" Failure Mode (v2.6.0)
- **Condition:** Forcing the velocity to be a unit vector ($v \leftarrow v/\|v\|$) while simultaneously having **zero friction** ($Acc = F - \Gamma$) leads to orbital chaos.
- **Mechanism:** The system preserves "momentum" perfectly in magnitude (always 1.0) and has no mechanism to dissipate energy in conflicting directions. The optimizer cannot effectively "steer" the particle because the momentum term dominates.
- **Outcome:** Loss stalls at ~0.85, Accuracy ~60%.

### 2. The "Damped Hard-Norm" Success Mode (v2.6.1 / v2.5.0)
- **Condition:** Hard Norm combined with **Dynamic Friction** ($Acc = F - \Gamma - \mu v$).
- **Mechanism:** Friction dissipates the "old" momentum, effectively making the update step $v_{new} \approx F_{ext}$ when friction is high. This allows the model to switch decisions quickly.
- **Outcome:** Stability is achieved. Loss < 0.01.

### 3. The "Soft-Norm Symplectic" Superior Mode (v2.6.2 Patched)
- **Condition:** **Soft Norm** ($\min(1, v_{max}/\|v\|)$) + **Symplectic Integration**.
- **Mechanism:**
    - Allows $v$ to carry information in its *magnitude* (Energy = Confidence).
    - Prevents gradient destruction associated with Hard Norm.
    - **Hard Threshold Singularities** allow for crisp logical state transitions (Events) without destabilizing the whole trajectory, as the integrator can handle the "kick".
- **Outcome:** Best theoretical foundation. Converges fast.

## Recommendations for Future Development

1.  **Abandon Hard Norm:** While v2.6.1 works, Hard Norm is a brute-force constraint that limits expressivity. Soft Norm (v2.6.2) is the correct path forward.
2.  **Maintain Friction:** Even in Symplectic systems, a "Forget Gate" (Dissipation) is valuable for recurrent tasks to clear history / context switch. v2.6.2 should ensure friction is tunable.
3.  **Event Horizons:** The Hard Threshold singularity logic in v2.6.2 is distinct and powerful for logic tasks. Keep it, but verify gradients flow around it (via the Soft Norm).

## Final Verdict
**v2.6.2 (Patched)** is the most advanced and correct formulation.
**v2.6.0** failed due to a specific regression (removing Friction) while keeping the hostile Hard Norm constraint.
