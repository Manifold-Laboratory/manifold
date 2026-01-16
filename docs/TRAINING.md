# Manifold Training Guide

[!VERSION](https://img.shields.io/badge/version-1.0.0-blue.svg)

This guide details how to train Manifold models, from configuration to monitoring and troubleshooting.

## 1. Quick Start

```bash
# Train a "Math Oracle" demo (learns arithmetic patterns)
python scripts/train.py \
    --model configs/model/gfn_medium.yaml \
    --training configs/training/math_oracle.yaml \
    --hardware configs/hardware/rtx_4090.yaml
```


## 2. Hardware Requirements (Verified)

We have empirically measured the memory usage of Manifold models. Thanks to $O(1)$ memory scaling, inference is extremely efficient.

| Model Size | Parameters | Training VRAM (Batch 1) | Inference VRAM (Context 128) | Recommended GPU |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | ~1.15 M | ~0.8 GB | ~16.6 MB | GTX 1650 / RTX 3050 |
| **Medium** | ~6.5 M | ~2.5 GB | ~100 MB | RTX 3060 / 4060 |
| **Large** | ~51 M | ~8.0 GB | ~450 MB | RTX 3090 / 4090 |
| **XL** | ~200 M | ~24 GB | ~1.5 GB | A100 / H100 |

> **Note**: Inference VRAM remains constant regardless of sequence length (tested up to 4096 tokens).

## 3. Configuration System

Manifold uses a hierarchical configuration system.

### A. Model Config (`configs/model/*.yaml`)
Defines the architecture topology.

```yaml
vocab_size: 64         # Input vocabulary
dim: 256              # Manifold dimension (D)
depth: 6              # Number of Layers (L)
heads: 4              # Parallel Geodesic Flows (H)
integrator: heun      # Symplectic Integrator (heun, rk4, leapfrog)
use_scan: false       # Parallel Training (requires CUDA)

# Cognitive Physics
physics:
  active_inference:
    enabled: true     # Enable Plasticity & Singularities
  fractal:
    enabled: true     # Enable Recursive Layers
```

### B. Training Config (`configs/training/*.yaml`)
Defines the task and optimization.

```yaml
batch_size: 32
lr: 1e-3
epochs: 100
dataset:
  name: math_oracle
  complexity: 2       # Number of digits
optimizer:
  name: riemannian_adam
  weight_decay: 0.1
  lambda_h: 0.01      # Hamiltonian regularization strength
```

## 4. Training Modes

### Sequential Mode (Default)
Standard token-by-token processing. Best for debugging and inference.
- **Complexity**: $O(N)$ time, $O(1)$ memory.
- **Usage**: Set `use_scan: false`.

### Parallel Mode (Scan)
Uses the Parallel Scan algorithm (prefix sum) to train on full sequences in logarithmic time.
- **Complexity**: $O(\log N)$ time, $O(N)$ memory.
- **Usage**: Set `use_scan: true`.
- **Note**: Requires working CUDA kernels.

## 5. Components & Loss

The loss function `GFNLoss` is a composite of three terms:

1.  **Task Loss ($L_{task}$)**: Standard Cross-Entropy prediction loss.
2.  **Hamiltonian Loss ($L_{H}$)**: Enforces energy conservation ($dH/dt \approx 0$).
3.  **Curiosity Loss ($L_{S}$)**: Maximizes state entropy to encourage exploration.

$$ L = L_{task} + \lambda_H L_H - T \cdot S $$

## 6. Troubleshooting

### NaN Loss Instability
Manifold dynamics can be chaotic. If loss goes to NaN:
1.  **Reduce Learning Rate**: Try 1e-4 or 5e-5.
2.  **Increase $\lambda_H$**: Stronger energy conservation helps stability.
3.  **Change Integrator**: Switch to `symplectic` or `rk4` for better stability.

### Checkpointing
Checkpoints are saved to `checkpoints/<run_name>/`.
To resume:
```bash
python scripts/train.py --resume checkpoints/run_1/epoch_10.pt
```
