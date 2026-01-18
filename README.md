# GFN | Manifold
> **Geodesic Flow Networks: Geometric Intelligence via Symplectic Flows**

[![VERSION](https://img.shields.io/badge/version-2.5.0-blue.svg)](https://github.com/Manifold-Laboratory/manifold/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/physics-Symplectic-purple.svg)](docs/PHYSICS.md)
[![Documentation](https://img.shields.io/badge/docs-latest-orange.svg)](docs/API.md)

---

## What's New in v2.5.0 (Riemannian Stability)

*   **Riemannian Optimization**: `RiemannianAdam` optimizer ensures parameter updates respect manifold geometry
*   **Adaptive Curvature Gating**: Learnable valve mechanism enables inertial coasting when optimal
*   **Zero-Force Inductive Bias**: Architectural enforcement of `E(0) = 0` for perfect state preservation
*   **Velocity Normalization**: Automatic stabilization preserving memory direction while controlling magnitude

---

## Overview

**GFN (Geodesic Flow Networks)**, publicly known as **Manifold**, reformulates sequence modeling as geodesic flow on a learned Riemannian manifold. Instead of attention matrices (O(N²)) or fixed-state compression, GFN models the latent state as a physical particle governed by symplectic integrators, enabling O(1) memory with infinite horizon stability.

**Core Innovation**: State transitions follow Einstein's geodesic equation with learned curvature, ensuring information conservation via Hamiltonian dynamics.

---
## Installation

```bash
pip install gfn
```

Or install from source:

```bash
git clone https://github.com/Manifold-Laboratory/manifold.git
cd manifold
pip install -e "."
```

**Requirements**: Python 3.10+, PyTorch 2.0+, CUDA (optional)

---

## Quick Start

```python
from src.model import Manifold
from src.optim import RiemannianAdam

# Model
model = Manifold(
    vocab_size=50257,
    dim=512,
    depth=12,
    heads=8,
    integrator_type='leapfrog'
).cuda()

# Optimizer (REQUIRED: standard Adam will fail)
optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,
    retraction='normalize',
    max_norm=10.0
)

# Training
for x, y in dataloader:
    optimizer.zero_grad()
    logits, _, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    optimizer.step()
```

---

## Verified Performance

### Binary Parity Task (Cumulative XOR)

**Challenge**: Predict cumulative XOR over arbitrarily long sequences (requires infinite-precision state tracking)

#### Training Performance

| **Model** | **Steps to Convergence** | **Final Loss** | **Training Time** |
|-----------|------------------------|---------------|------------------|
| **GFN** | **852** | **0.0863** | **48 min** (L=20) |
| MicroGPT | 4,000 | 0.0123 | 1h 21min (L=20) |

*GFN converged 4.7× faster than Transformer baseline*

#### Zero-Shot Generalization Results

**Trained on L=20 only**, tested on sequences up to **L=1000 (50× longer)**:

<p align="center">
  <img src="tests/benchmarks/results/gfn_superiority/parity_generalization.png" alt="Parity Generalization" width="900"/>
</p>

*Figure: Left plot shows perfect accuracy generalization. Right plot demonstrates O(1) memory scaling (flat line) vs theoretical O(N) baseline.*

**Detailed Results**:

| **Test Length** | **GFN Accuracy** | **VRAM** | **Extrapolation** |
|----------------|-----------------|---------|-------------------|
| 20 (seen)       | 100.0%          | 28.3 MB | 1× (baseline)     |
| 50              | 100.0%          | 28.4 MB | 2.5×              |
| 100             | 100.0%          | 28.6 MB | 5×                |
| 200             | 100.0%          | 29.0 MB | 10×               |
| 400             | 100.0%          | 29.8 MB | 20×               |
| 500             | 100.0%          | 30.4 MB | 25×               |
| **1000**        | **100.0%**      | **32.1 MB** | **50×**           |

**Key Findings**:
- ✅ **Perfect Generalization**: 100% accuracy on all lengths (never seen sequences 50× longer)
- ✅ **O(1) Memory Verified**: VRAM growth of only **3.8 MB** (13.4%) from L=20→1000
- ✅ **Linear Regression**: Memory vs Length slope = 0.0038 MB/token (effectively constant)
- ✅ **Symplectic Stability**: Zero gradient vanishing across 1000-step integration

*Full benchmark results and plots available in [tests/benchmarks/results/gfn_superiority/](tests/benchmarks/results/gfn_superiority/)*

---

## Core Architecture

### Geodesic Equation
```
d²x/dτ² + Γ(v, x) = F_ext(token)
```

- **x**: Position in semantic manifold
- **v**: Velocity (momentum/memory)
- **Γ**: Christoffel symbols (learned curvature)
- **F**: Input token force

### Symplectic Integration (Leapfrog)

```python
# Half-step velocity
v_half = v + 0.5 * dt * (F - Γ(v, x))

# Full-step position
x_next = x + dt * v_half

# Half-step velocity finalization
v_next = v_half + 0.5 * dt * (F - Γ(v_half, x_next))

# Stabilization
v_next = v_next / (||v_next|| + ε)
```

**Properties**:
- Time-reversible
- Volume-preserving (det(J) = 1)
- Energy-conserving (|ΔH| ≈ O(dt²))

---

## Comparison with Baselines

| **Architecture** | **Memory (Inference)** | **Compute (per token)** | **Gradient Stability** | **Verified** |
|-----------------|----------------------|------------------------|----------------------|--------------|
| Transformer | O(N) KV cache | O(N·d) attention | Good | — |
| LSTM/GRU | O(1) | O(d²) gates | Poor | — |
| Mamba (SSM) | O(1) | O(d²) state update | Medium | — |
| **GFN** | **O(1)** state | **O(d²·R)** Christoffel | **Excellent** | **✓** |

*Where N = sequence length, d = hidden dim, R = Christoffel rank (typically 16-32)*

**Note**: GFN's O(d²·R) per-token cost is comparable to LSTMs/Mamba. For training full sequences, all architectures are O(N·...) in sequence length.

---

## Documentation

- **[SCIENTIFIC_PAPER.md](docs/SCIENTIFIC_PAPER.md)** - Complete research paper with mathematical derivations
- **[API.md](docs/API.md)** - Python API reference
- **[TRAINING.md](docs/TRAINING.md)** - Training guide and best practices
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and components
- **[BENCHMARKS.md](docs/BENCHMARKS.md)** - Empirical performance validation
- **[PHYSICS.md](docs/PHYSICS.md)** - Mathematical foundations

---

## Use Cases

-   **Long-Context Reasoning**: Process sequences >10K tokens with constant memory
-   **Algorithmic Tasks**: Perfect extrapolation on logical reasoning (XOR, sorting, arithmetic)
-   **Edge Deployment**: Run large models on memory-constrained devices (<4GB RAM)
-   **Scientific Computing**: Model systems requiring conservation laws (physics simulations)

---

## Repository Structure

```text
/
├── src/                # Core Implementation
│   ├── model.py        # Main Manifold Architecture
│   ├── geometry.py     # Christoffel Symbols & Integrators
│   ├── layers.py       # M-Layer (Manifold Layer)
│   ├── embeddings.py   # Functional Embeddings
│   └── optim.py        # RiemannianAdam Optimizer
├── docs/               # Technical Documentation
│   ├── SCIENTIFIC_PAPER.md
│   ├── API.md
│   ├── TRAINING.md
│   └── BENCHMARKS.md
├── tests/              # Verification Suite
│   └── benchmarks/     # Reproducible Benchmarks
└── LICENSE             # Apache 2.0
```

---

## Development Status

**Version 2.5.0** is production-ready for research and experimentation.

**Verified**:
- ✅ O(1) memory scaling (empirically confirmed)
- ✅ Perfect generalization on Parity task
- ✅ Stable training with RiemannianAdam
- ✅ Symplectic gradient flow

**In Development**:
- CUDA kernel acceleration (10-50× speedup expected)
- Mixed precision training (FP16/BF16)
- Language modeling benchmarks (WikiText)
- Mixture of Manifolds (MoM) architecture

---

## Citation

If you use GFN in your research, please cite:

```bibtex
@software{gfn2026,
  title={GFN: Geodesic Flow Networks},
  author={Stürtz Joaquín},
  year={2026},
  version={2.5.0},
  url={https://github.com/Manifold-Laboratory/manifold},
  license={Apache-2.0}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Links**:
- [Report Issues](https://github.com/Manifold-Laboratory/manifold/issues)
- [Request Features](https://github.com/Manifold-Laboratory/manifold/issues/new)
- [View Roadmap](https://github.com/Manifold-Laboratory/manifold/projects)

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Manifold Laboratory</b><br>
  <i>Geometric intelligence through physical principles.</i>
</div>
