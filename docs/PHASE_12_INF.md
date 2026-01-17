# Phase 12: Implicit Neural Fields (INFs)

**Status:** Implemented & Verified (v3.0)
**Date:** 2026-01-17

## 1. Concept
We transitioned from **Discrete Embeddings** (`nn.Embedding` table) to **Implicit Continuous Functions** (INFs).
Instead of looking up a static vector, the model:
1.  Looks up a **Low-Rank Coordinate** $c \in \mathbb{R}^{16}$.
2.  Passes it through a **SIREN (Sine ResNet)** MLP.
3.  Outputs the high-dimensional embedding vector.

$$ E(token) = \Phi_{\theta}(c_{token}) $$

### 2. Implementation
### `src/embeddings.py`
Two modes available:
1.  **`ImplicitEmbedding` (Hybrid)**: Learnable Coordinate Table `[Vocab, 16]` -> SIREN.
    *   Good for: Fixed vocabularies where we want topology but some learnability.
2.  **`FunctionalEmbedding` (Pure)**: Procedural Coordinate (Hash) -> SIREN.
    *   **No Param Table**. Coordinates are math functions of the ID.
    *   Good for: True "Infinite Vocabulary" inputs.

### 3. Benefits
*   **Topology:** Tokens now live in a continuous metric space.
*   **Memory Efficiency:**
    *   Standard (500k): **256M Params** (OOM danger)
    *   Implicit (500k): **136M Params** (~47% reduction)
    *   Functional (500k): **128M Params** (~50% reduction)

> **Note on O(1):** While the *Input Embedding* is now O(1) (Functional mode), the **Output Readout** interaction (`nn.Linear(dim, vocab)`) is still O(N). This restricts the "Infinite Vocab" goal until we implement **Implicit Readout**.

## 4. Verification
Ran `tests/benchmarks/benchmark_inf_vram.py`.
*   **10k - 100k:** Both INF modes save ~50% VRAM/Params.
*   **1 Million:** All options OOM'd due to the gigantic Output Readout Layer (which wasn't replaced).
*   **Conclusion:** INFs solve the Input bottleneck perfectly.

## 5. Usage
Set in `config.yaml`:
```yaml
physics:
  embedding:
    type: functional  # or 'implicit', 'standard'
    coord_dim: 16
```
