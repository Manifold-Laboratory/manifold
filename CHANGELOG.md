# Changelog

All notable changes to the MANIFOLD project will be documented in this file.

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
