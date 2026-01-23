# Physics Analysis: Manifold GFN v2.6.2 (Current Patched)

**Date:** 2026-01-22
**Status:** Patched (Soft Norm)
**Observed Result:** Loss ~0.08, Acc 95.5% (Post-Patch) / Loss 1.0 (Pre-Patch)

## 1. Core Mathematical Formulation

### A. Velocity Control Strategy: **Soft Norm (Clamping)**
**Location:** `gfn/layers.py` (Lines 303-313)

The model implements a **Soft Norm** strategy to control velocity magnitude without destroying gradients.
$$v_{new} = v \cdot \min\left(1, \frac{v_{max}}{\|v\|}\right)$$
- **Max Velocity ($v_{max}$):** 5.0
- **Mechanism:** Differentiable clamping.
- **Gradient Behavior:**
    - If $\|v\| < v_{max}$: Gradients flow 100% (Identity operation).
    - If $\|v\| > v_{max}$: Gradients are dampened but *direction* gradients are preserved.
- **Contrast:** This replaces the "Hard Norm" ($v \leftarrow v/\|v\|$) which was identified as the cause of "Gradient Coma" (zero gradients in radial direction).

### B. Manifold Dynamics (MLayer)
- **Integration Scheme:** **Yoshida 4th-Order Symplectic** (or selected via config).
    - Code uses `dt_scale` for adaptive time stepping.
- **Energy Conservation:**
    - **Explicit Projection:** REMOVED. The code relies on the symplectic nature of the integrator and the Soft Norm to maintain stability.
    - **Velocity Normalization:** Pre-integration velocity normalization (direction control) was observed in `forward` (Line 237: `v_norm = v / (v_mag + 1e-6)`). *Note: This looks like a remnant or a specific design choice to normalize INPUT direction but allow OUTPUT magnitude dynamics up to a cap.*

### C. Geometry (Christoffel Symbols)
**Location:** `gfn/geometry.py`

- **Type:** `LowRankChristoffel` / `ReactiveChristoffel`.
- **Singularities:** **Hard Threshold (Event Horizon)**.
    - Logic: `mask = (potential > sing_thresh).float()`
    - Modulation: `modulation = 1.0 + (strength - 1.0) * mask`
    - **Significance:** Uses a step function (non-differentiable but stable) effectively creating "Event Horizons" where curvature jumps.
- **Curvature Gate:** `torch.sigmoid(self.gate_proj(x))` (The Valve).

## 2. Training Dynamics Implications

- **Stability:** The combination of **Yoshida Integrator** (High precision) and **Soft Norm** (Boundaries) creates a stable phase space.
- **Learning:**
    - **Magnitude Dynamics:** Unlike Hard Norm, Soft Norm allows the model to use velocity magnitude as a memory buffer (Energy = Information).
    - **Singularities:** The Hard Threshold prevents "Gravity Leak" (soft sigmoid affecting valid regions), allowing for crisp logical switching.

## 3. Conclusion
This version represents the **"Corrected"** state. The mathematical formulation supports stable oscillation (pendulum dynamics) and information storage in kinetic energy, bounded by $v_{max}$.
