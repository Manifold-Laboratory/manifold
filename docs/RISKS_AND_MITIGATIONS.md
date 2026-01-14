# Risks and Mitigations

This document outlines critical technical risks identified in the MANIFOLD architecture (v0.3/v0.4) and the specific safeguards implemented to correct them.

## 1. Gravitational Collapse (Gradient Stagnation)
**Risk:** When making curvature $\Gamma$ dependent on position $x$, the model might create "gravity wells" so deep (high positive curvature) that all geodesics converge to a single point.
- **Symptom:** Loss stagnation (e.g., stuck at ~2.26) and repetitive token generation.
- **Status:** **Mitigated.**
- **Defense Mechanism:**
    1.  **Metric Clamping:** We enforce `torch.clamp(curvature, -5.0, 5.0)` in `src/geometry.py`. This physically prevents infinite curvature.
    2.  **Bounded Modulation:** The position dependency is defined as $\Gamma_{new} = \Gamma_{old} \times (1 + \sigma(V \cdot x))$. Since `sigmoid` is bounded $[0, 1]$, the gravitational pull can at most double, never explode.
    3.  **Future Defense (v0.4):** Adaptive Time-Stepping (Golden Integration) adds a third layer of defense by slowing down time in dense regions effectively preventing numerical tunneling.

## 2. Wormhole Desynchronization
**Risk:** Multi-Scale heads run at different speeds ($dt, 1.5 dt, 2.25 dt \dots$). If the "slow" head (carrying context) and "fast" head (carrying syntax) are not properly aligned at the output, the representation becomes incoherent.
- **Symptom:** High training loss despite healthy gradient flow.
- **Status:** **Partially Mitigated.**
- **Defense Mechanism:**
    1.  **Linear Mixing:** The `ParallelMLayer` includes a final `out_proj` (Linear) layer that mixes the concatenated outputs of all heads. This allows the model to learn *how* to recombine the timelines.
    2.  **Pre-LayerNorm:** Normalizing before splitting heads ensures all timelines start from a common magnitude scale.
    3.  **Pending:** We rely on the model learning to synchronize. Explicit synchronization gates (Key-Query covariance) might be needed in v0.5 if this proves insufficient.

## 3. The Cost of Dynamic Scan
**Risk:** A true Dynamic Parallel Scan requires recomputing the operator $A_t(x_{t-1})$ at every step, which breaks parallelism ($O(N)$ dependency).
- **Status:** **Solved via Approximation.**
- **Defense Mechanism:**
    - **Linearized Approximation:** In `ParallelMLayer`, we calculate $A_t$ and $B_t$ based on the **Input Force Sequence** ($F_t$), not the recursive state $x_{t-1}$.
    - $A_t = Net(F_t)$
    - This allows us to compute all $A_t$ in parallel ($O(1)$ depth) and then run the scan ($O(\log N)$).
    - **Trade-off:** We sacrifice "perfect" dynamic feedback during the fast scan for massive training speed. The "true" dynamics are recovered during fine-tuning interactions or via the non-linear correction step.
    - **Performance Drop:** The drop from 2000 to 430 ex/s is due to the current **Python-based Scan implementation**. The logic itself is $O(\log N)$. Integrating the `scan_kernel.cu` (planned for v0.4) will restore the 2000+ ex/s speed.
