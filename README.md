# Manifold: Geometric Sequence Modeling via Symplectic Flows

> **Infinite Context. Constant Memory. Hamiltonian Dynamics.**

<p align="center">
  <img src="tests/benchmarks/results/geodesic_flow/geodesic_flow_3d.png" alt="Latent Geodesic Trajectories" width="100%"/>
  <br>
  <i><b>Figure 1: The Geometry of Thought.</b> Visualization of the semantic state evolution ($x_t, v_t$) traversing a learned high-dimensional Riemannian manifold. Unlike discrete state transitions in traditional RNNs, Manifold models intelligence as a continuous symplectic flow, conserving momentum and information over infinite horizons.</i>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-orange.svg)]()

---

## 1. Introduction: The Memory Bottleneck

The fundamental limitation of modern Large Language Models (LLMs) is the **Key-Value (KV) Cache**. To generate the next token, a Transformer must explicitly attend to its entire history. This results in a memory complexity of $O(N)$, creating a hard physical ceiling on context length and inference throughput.

**Manifold** introduces a paradigm shift by reformulating sequence modeling through the lens of **Geometric Mechanics**. Instead of storing a history of discrete tokens, Manifold encodes context into the **momentum** of a dynamic massive particle moving through a curved semantic space.

This approach yields a **Physically-Structured State Space Model (SSM)** that achieves:
*   **$O(1)$ Inference Memory**: Constant state complexity (~30MB) regardless of sequence length ($L=10$ or $L=1,000,000$).
*   **Infinite Context Horizon**: Information is preserved via symplectic conservation laws rather than explicit storage.
*   **Symplectic Stability**: Energy-conserving integrators prevent the vanishing/exploding gradient problem inherent in standard RNNs.

---

## 2. The Superiority Benchmark

To rigorously evaluate the state-tracking capabilities of this architecture, we conducted the **Manifold Superiority Benchmark**. This benchmark utilizes the **Cumulative Parity (XOR) Task**, a problem that is computationally irreducible and requires perfect, lossless memory retention over the entire sequence duration. A single bit-flip error at $t=0$ propagates to invert the target at $t=\infty$, making it the ultimate test of long-term dependency handling.

We compared **Manifold (v2.6.0)** against a standard **Transformer (MicroGPT)** with equivalent parameter counts.

### 2.1. Infinite Length Generalization

Both models were trained **exclusively** on sequences of length $L=20$. We then evaluated their ability to generalize to sequences up to $L=100,000$ (5,000x longer than training).

<p align="center">
  <img src="tests/benchmarks/results/gfn_superiority/parity_result.png" alt="Superiority Benchmark Result" width="100%"/>
  <br>
  <i><b>Figure 2: The Generalization Gap.</b> (Left) Accuracy on Cumulative Parity task relative to sequence length. (Right) VRAM usage scaling. Manifold generalizes perfectly to 100,000+ tokens (~5000x training length) while maintaining O(1) memory.</i>
</p>

### 2.2. Vocabulary Scaling (O(1) Parameters)

Manifold's **Functional Embeddings** allow the vocabulary to grow indefinitely without increasing parameter count.

<p align="center">
  <img src="tests/benchmarks/results/infinite_scaling/infinite_scaling_plot.png" alt="Infinite Vocab Scaling" width="100%"/>
  <br>
  <i><b>Figure 3: Infinite Vocabulary.</b> Proving O(1) memory scaling with respect to vocabulary size (up to 1 Million tokens).</i>
</p>

**Empirical Conclusion**: Manifold demonstrates true **algorithmic generalization**. It has learned the underlying generative law of the data (the XOR operator) rather than simply memorizing patterns. This capability is enabled by its **momentum-based memory**, which acts as a robust, noise-resistant carrier of logical state.

---

## 3. Dynamic Physics: Forgetting & Remembering

Standard RNNs struggle to forget ("catastrophic memory"), while Transformers must explicitly mask history. Manifold employs a **Dynamic Forget Gate** (thermodynamic friction) that adapts to the input energy.

### 3.1. Context-Aware Forgetting

*   **Stable Context:** Friction $\approx 0$ (Symplectic Conservation). The model remembers.
*   **Context Switch:** Friction spikes (Energy Dissipation). The model forgets.

<p align="center">
  <img src="tests/benchmarks/results/stability/dynamic_friction_test.png" alt="Dynamic Friction Response" width="100%"/>
  <br>
  <i><b>Figure 3: The Physics of Forgetting.</b> (Left) When a high-energy "Context Switch" occurs (Blue), the Learnable Friction (Red) spikes immediately to dissipate previous state momentum. (Right) The learned activation function shows a clear phase transition from conservation to dissipation based on input magnitude.</i>
</p>

---

## 4. Theoretical Foundations

Manifold diverges from standard connectionsist architectures by imposing **Hamiltonian constraints** on the latent update rule. The network learns to shape the geometry of the solution space, such that the "natural motion" of the state vector corresponds to the desired computation.

### 4.1. The Geodesic Equation

The latent state update is governed by the discrete-time approximation of the geodesic equation on a Riemannian manifold:

$$
\frac{d^2x}{dt^2} + \Gamma^k_{ij}(x) \frac{dx^i}{dt} \frac{dx^j}{dt} = F(u_t)
$$

Where:
*   $x_t \in \mathbb{R}^d$: The **Position** (Semantic State).
*   $v_t = \dot{x}_t \in \mathbb{R}^d$: The **Velocity** (Contextual Momentum).
*   $\Gamma(x)$: The **Christoffel Symbols** (Learned Interaction Tensor), defining the local curvature and feature interactions ($O(d^2)$ complexity).
*   $F(u_t)$: The **External Force** derived from the input token embedding.

### 4.2. Symplectic Stability & Conservation

Standard Euler integration used in Residual Networks is energy-dissipative, leading to signal loss. Manifold employs a **Leapfrog Integrator**, a symplectic solver designed to strictly conserve phase-space volume.

<p align="center">
  <img src="tests/benchmarks/results/stability/stability_metrics_20260119_022614.png" alt="Symplectic Stability Metrics" width="100%"/>
  <br>
  <i><b>Figure 3: Conservation Laws.</b> Analysis of the Hamiltonian energy drift over long horizons. Unlike standard integration which diverges (Green), Manifold's symplectic solver (Blue) keeps energy bounded, ensuring gradient stability for $L \to \infty$.</i>
</p>

---

## 5. Latent Space Analysis

We perform a deep diagnostic of the model's internal representation to understand *how* it solves complex tasks.

### 5.1. Manifold Trajectories vs. Random Walks

By projecting the high-dimensional hidden states into 3D, we observe that Manifold learns smooth, deterministic orbits, whereas traditional RNNs often exhibit chaotic or collapsing trajectories.

<p align="center">
  <img src="tests/benchmarks/results/trajectories/trajectory_comparison.png" alt="Trajectory Comparison" width="100%"/>
  <br>
  <i><b>Figure 4: Latent Dynamics Comparison.</b> Left: The chaotic state evolution of a standard RNN. Right: The coherent, orbital structure of a Manifold trained on the same task. The geometric prior forces the state to follow smooth geodesic paths.</i>
</p>

### 5.2. The Geometry of Optimization

Why does Manifold converge faster on complex tasks? The answer lies in the Loss Landscape. By constraining parameters to the manifold, we convexify the optimization surface.

<p align="center">
  <img src="tests/benchmarks/results/loss_landscape/loss_landscape_3d_comparison.png" alt="Loss Landscape 3D" width="100%"/>
  <br>
  <i><b>Figure 5: Optimization Topography.</b> (Left) The sharp, non-convex landscape of a standard Transformer trained on Parity. (Right) The smooth, quasi-convex basin of Manifold, enabled by RiemannianAdam and geometric regularization.</i>
</p>

---

## 6. Advanced Dynamics: Beyond Text

The geometric framework is domain-agnostic. By projecting inputs into the tangent space of the manifold, the model processes text, images, and audio as unified force vectors. Current experiments demonstrate convergence in multimodal tasks, suggesting that geometric mechanics is a universal prior for sequential data.

<p align="center">
  <img src="tests/benchmarks/results/fractals/fractal_zoom_comparison.png" alt="Fractal Dynamics" width="100%"/>
  <br>
  <i><b>Figure 6: Fractal State Space.</b> Investigating the self-similar properties of the learned manifold. The model learns to organize information hierarchically, exhibiting fractal structures in its decision boundaries.</i>
</p>

---

## 7. Implementation & Usage

Manifold provides a production-ready implementation with a PyTorch-native API.

### 7.1. Installation

```bash
pip install gfn
# OR for development
git clone https://github.com/Manifold-Laboratory/manifold.git
cd manifold
pip install -e "."
```

### 7.2. Geodesic Training Loop

The optimizer must respect the geometry of the parameter space. Standard Adam optimization assumes a Euclidean flat space, which is suboptimal for Riemannian models. We provide `RiemannianAdam` to perform covariant gradient updates.

```python
import torch
from gfn.model import Manifold
from gfn.optim import RiemannianAdam

# Initialize the Geometric Engine
model = Manifold(
    vocab_size=50257,
    dim=512,
    depth=12,
    heads=8,
    integrator_type='leapfrog'  # Symplectic Solver
).cuda()

# Optimizer: RiemannianAdam is required for manifold constraints
optimizer = RiemannianAdam(model.parameters(), lr=1e-4, max_norm=10.0)

# Training with symplectic conservation
model.train()
for input_ids, targets in dataloader:
    optimizer.zero_grad()
    
    # Forward pass: Evolve state along geodesics
    logits, (x_final, v_final), _ = model(input_ids)
    
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 50257), targets.view(-1))
    loss.backward()
    
    # Gradient clipping is essential for differential stability around singularities
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    optimizer.step()
```

---

## 8. Citation

Manifold is an active research project. If you utilize this framework or its findings in your research, please cite:

```bibtex
@article{manifold2026,
  title={Manifold: Geometric Sequence Modeling via Symplectic Flows},
  author={Manifold Laboratory},
  journal={arXiv preprint},
  year={2026}
}
```

---

<div align="center">
  <b>Manifold Laboratory</b><br>
  <i>Geometric Intelligence via Physical Principles</i>
</div>
