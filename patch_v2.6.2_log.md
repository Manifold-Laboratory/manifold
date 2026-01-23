# Patch v2.6.2 "The Super Commit" - Changelog

**Date:** 2026-01-22
**Status:** Applied & Verified

## 1. Stabilization (The Root Cause)
**File:** `gfn/geometry.py`
-   **Change:** Reverted Christoffel Symbols initialization from `0.01` to `0.001` (Lines 24-27).
-   **Reason:** v2.6.1 used 0.001. v2.6.2's 0.01 init injected 100x more "Curvature Energy" at start ($E \propto \Gamma^2 \propto U^4$), causing immediate chaotic heating before the optimizer could learn.

## 2. Energy Preservation (The Architecture)
**File:** `gfn/layers.py`
-   **Change (Input):** Replaced **Hard Norm** (`v/|v|`) with **Soft Input Clamp** (`max_v/(|v|+e)`) at Lines 230-238.
-   **Change (Output):** Verified **Soft Output Clamp** (Pendulum Logic) exists at Lines 305-315.
-   **Reason:** Allows "Confidence" (Velocity Magnitude) to flow between layers. Hard Norm was destroying the gradients of confident predictions, forcing the model to "shout" (High Curvature) to steer, rather than "speed up" (High Velocity).

## 3. High-Fidelity Integration
**File:** `gfn/model.py`
-   **Change:** Changed default `integrator_type` from `'heun'` to `'yoshida'` (Line 33).
-   **Reason:** Heun (2nd Order) is dissipative. Yoshida (4th Order Symplectic) conserves the Phase Space volume, ensuring that the "Soft Norm" system behaves like a true Hamiltonian system rather than a leaky ODE.

## 4. Verification Check
-   [x] **Friction:** Dynamic Friction (Forget Gate) verified in `geometry.py`.
-   [x] **Singularities:** Hard Threshold (Event Horizon) verified in `geometry.py`.
-   [x] **Logic:** Input/Output Soft Clamps match.

**Ready for Training.**
