# Changelog

All notable changes to the MANIFOLD project will be documented in this file.

## [v1.0.0] - The Conscious Engine (Unified Release)
**"The Singularity Milestone"** - Full integration of Thermodynamics, Semantic Symmetries, and Fractal Manifolds.

### Features
- **Unified Cognitive Suite**: Seamless integration of all three physical pillars (Curiosity, Noether Symmetries, Recursive Fractals).
- **Golden Config**: Introduced `configs/model/gfn_v1_0.yaml` as the reference standard for high-performance geometric intelligence.
- **Pristine Core**: Finalized technical audit and cleanup of `GFNLoss` and `FractalMLayer` for production stability.
- **Verified Integration**: Comprehensive suite of unified tests ensuring gradient flow across all modules.

### Pillars
- **Thermodynamics**: Entropy-Driven Curiosity for autonomous exploration.
- **Symmetries**: Isomeric Heads for semantic consistency and invariance.
- **Fractals**: Recursive Tunneling for adaptive multiscale resolution.

---

## [v0.8.0] - Fractal Manifolds (Recursive Intelligence)
**"The Infinite Resolution Update"** - Multiscale cognitive processing through recursive state-space nesting.

### Core Features
- **Fractal M-Layer**: Introduced a hierarchical manifold layer that can "tunnel" into high-resolution sub-manifolds.
- **Recursive Tunneling**: Implemented a dynamic gating mechanism that activates sub-manifold processing when local semantic curvature exceeds a learned threshold.
- **Multiscale Blending**: Perturbative state updates that blend macro-flow and micro-resolutions for superior semantic precision.

### Verified
- **Fractal Test Suite**: Passing `tests/unit/test_fractals.py` (Tunneling activation and multiscale blending).

---

## [v0.7.0] - Semantic Symmetries (Noether)
**"The Invariance Update"** - Enforcing geometric consistency across semantic contexts.

### Core Features
- **Isomeric Heads**: Implemented hard weight sharing in `MLayer`. Multiple manifold subspaces can now share identical geometric laws (Metric Invariance).
- **Noether Loss**: Added a new symmetry regularization term in `src/losses.py` that penalizes geometric divergence between supposedly isomeric subspaces.
- **Manifold Metadata Export**: Refactored the forward pass of `MLayer` and `Manifold` to export latent geometric outputs (Christoffel symbols) for global loss computation.

### Verified
- **Symmetry Test Suite**: Passing `tests/unit/test_symmetries.py` (Weight sharing and Noether loss consistency).

---

## [v0.6.0] - Thermodynamics (Entropy-Driven Curiosity)
**"The Life Force"** - Implementing the first pilar of v1.0 by introducing intrinsic cognitive motivation.

### Core Features
- **Entropy-Driven Curiosity**: Added a new thermodynamic loss term that maximizes the differential entropy of the manifold's velocity distribution.
    - Prevents "Cognitive Collapse" by forcing the model to explore diverse geodesics for the same task.
    - Implemented `curiosity_loss` in `src/losses.py` based on standard deviation logs (entropy proxy).
- **Curiosity Temperature ($T$ / $\lambda_c$)**: Parameter that controls the trade-off between task exploitation and cognitive exploration.

### Configuration
- **Active Thermodynamics**: Enabled `lambda_c: 0.05` in `configs/training/math_oracle.yaml`.

### Verified
- **Logic Validation**: Passing `tests/unit/test_curiosity.py` for entropy scaling and gradient existence.

---

## [v0.5.0] - Cognitive Dynamics (Active Inference)
**"The Awakening"** - Transforming Manifold from a static geometry into a reactive cognitive system.

### Core Features
- **Cognitive Physics Engine**: A completely new dynamical system in `src/geometry.py` that reacts to internal states.
    - **Reactive Curvature**: Manifold stiffens ($\Gamma \uparrow$) in response to high kinetic energy (uncertainty).
    - **Logical Singularities**: High semantic potential triggers "Event Horizons" (attractors) to stabilize decisions.
- **Autonomous Geometric Attention**: `TimeDilationHead` allows each processing thread to learn its own time-flow ($dt$), effectively creating wormholes to skip irrelevant data.
- **Recursive Geodesics**: Layers now project "curvature context" to subsequent layers, enabling hierarchical steering control.

### Breaking Changes
- **API Unification**: Renamed all `GLayer` instances to `MLayer`.
- **Configuration**: Introduced `physics_config` dict for fine-grained control of cognitive dynamics.

---

## [v0.4.0] - High-Performance Kernel & Stability
**"The Engine Update"** - Achieving production-grade inference speed and numerical stability.

### Added
- **Fused CUDA Kernels**: Custom C++ extensions for `christoffel_fused.cu` and `leapfrog_fused.cu`.
    - Robust Block-Reduction implementation for numerical precision.
- **Golden Integration**: Refined `src/geometry.py` integration logic.
    - Fixed `dt_scale` handling for hybrid Training/Inference compatibility.
    - Implemented secure fallback mechanisms for non-CUDA environments.

### Fixed
- **Training Instability**: Resolved `TypeError` in sequential mode during backpropagation.
- **Compilation Issues**: Fixed MSVC/PyTorch header conflicts (`ambiguous symbol std`) on Windows.

---

## [v0.3.0] - Geometric Enhancements

### Added
- **Dynamic Curvature Fields ($\Gamma(x, v)$)**: Curvature now depends on the *position* (state) of the manifold, not just velocity.
    - Creates "Gravity Wells" where specific semantic concepts naturally warp the geometry.
    - Implemented in `src/geometry.py` by modulating the Christoffel symbols with `1 + sigmoid(V \cdot x)`.
- **Manifold Wormholes (Multi-Scale Time)**: Implicit skip connections via temporal dilation.
    - Heads are now initialized with logarithmic time-scales ($dt, 1.5dt, 2.25dt \dots$).
    - Allows "Slow Manifolds" to transport information over long distances in fewer effective steps, preserving $O(1)$ memory.
- **Unit Tests**: Added `tests/unit/test_geometric_enhancements.py` to verify dynamic modulation and time-scaling.

### Changed
- **MLayer Initialization**: `vectorized` the learnable `dt_param` to support unique time-scales per head.

---

## [v0.2.0] - Parallel & Multi-Head
**"The Speed Update"** - Breaking the sequential bottleneck and expanding dimensionality.

### Added
- **Parallel Associative Scan**: Enables $O(\log N)$ training complexity.
    - Implemented Linearized Geodesic Flow approximation ($v_t = A_t v_{t-1} + B_t$) in `src/scan.py`.
    - Added `ParallelMLayer` in `src/layers.py` for scan-based training.
- **Multi-Head Geodesic Flows**: Splitting the manifold into independent subspaces.
    - Allows the model to learn distinct geometries (e.g., Syntax Manifold vs. Semantic Manifold) simultaneously.
- **CUDA Kernels**: Initial implementation of fused kernels for Christoffel and Leapfrog operations.

### Changed
- **Rebranding**: Renamed project from "GFN" to "MANIFOLD".
- **Refactoring**: Cleaned up `src/` structure, removing legacy V1 code.

---

## [v0.1.0] - Genesis
**"The Foundation"** - Initial proof of concept for Geodesic Flow Networks.

### Added
- **Core Architecture**: `Manifold` model with sequential `MLayer`.
- **Differential Geometry**: `LowRankChristoffel` symbols for efficient $O(D \cdot R)$ curvature computation.
- **Integrators**: Symplectic, RK4, and Heun integrators for solving the Geodesic ODE.
- **Riemannian Adam**: Custom optimizer for manifold-aware gradient updates.
- **Basic Training Loop**: Support for autoregressive training on math datasets.
