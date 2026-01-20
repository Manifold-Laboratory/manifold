# MANIFOLD Documentation

Welcome to the MANIFOLD documentation. This directory contains comprehensive technical reference and research materials for the MANIFOLD (Multi-scale Adaptive Neural Inference via Flow On Learned Dynamics) architecture.

## 📚 Core Documentation

### Technical Reference
- **[SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md)** - Complete research paper with mathematical derivations, experimental validation, and reproducibility details
- **[TECHNICAL_HANDBOOK.md](TECHNICAL_HANDBOOK.md)** - **The "Source of Truth"**: Fundamental equations, loss engine details, and optimization protocols.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview and design principles
- **[API.md](API.md)** - Python API reference for model usage
- **[TRAINING.md](TRAINING.md)** - Training procedures, hyperparameters, and optimization guides
- **[BENCHMARKS.md](BENCHMARKS.md)** - Empirical performance benchmarks and comparisons

### Theoretical Foundations
- **[PHYSICS.md](PHYSICS.md)** - Mathematical foundations: Riemannian geometry, symplectic integration, Hamiltonian mechanics
- **[CONCEPTS.md](CONCEPTS.md)** - Key concepts: geodesic flow, Christoffel symbols, manifold learning
- **[MODELS_AND_MECHANICS.md](MODELS_AND_MECHANICS.md)** - Detailed mechanical explanations of core components

### Specialized Topics
- **[COMPONENTS.md](COMPONENTS.md)** - Individual component specifications (embeddings, layers, integrators, optimizers)
- **[CONFIGURATION.md](CONFIGURATION.md)** - YAML configuration reference and best practices

## 🚀 Quick Start

1. **For Researchers**: Start with [SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md)
2. **For Developers**: Start with [API.md](API.md) and [TRAINING.md](TRAINING.md)
3. **For Theory**: Start with [PHYSICS.md](PHYSICS.md) and [CONCEPTS.md](CONCEPTS.md)

## 📊 Key Results

**Binary Parity Task (L=20 training)**:
- **100% accuracy** on sequences up to L=1000 (50× extrapolation)
- **O(1) memory**: 28-32MB VRAM regardless of sequence length
- **Verified symplectic stability**: No gradient vanishing/explosion

See [BENCHMARKS.md](BENCHMARKS.md) for complete results.

## 📖 Additional Resources

- [Main Project README](../README.md) - Project overview and installation
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Changelog](../CHANGELOG.md) - Version history

---

**Version**: 2.6.0 "Riemannian Stability"  
**Last Updated**: January 18, 2026  
**License**: Apache 2.0
