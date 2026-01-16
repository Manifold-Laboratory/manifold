"""
GFN: Geodesic Flow Networks
===========================

A novel neural architecture that models sequences as flows on Riemannian manifolds.

Available Integrators:
    - HeunIntegrator: 2nd order, fast (RECOMMENDED)
    - RK4Integrator: 4th order, accurate but slower
    - SymplecticIntegrator: Velocity Verlet, energy-preserving
    - LeapfrogIntegrator: Störmer-Verlet, best symplectic

Usage:
    from src import GFN
    model = GFN(vocab_size=16, dim=512, depth=12, rank=128, integrator_type='heun')
    
    # Physics-informed training:
    from src import GFNLoss, RiemannianAdam
    criterion = GFNLoss(lambda_h=0.01)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
"""

__version__ = "1.0.0"
__author__ = "Manifold Laboratory (Joaquín Stürtz)"

# Core Model
from .model import Manifold as GFN  # Alias for backward compatibility
from .model import Manifold
from .adjoint import AdjointManifold as AdjointGFN
from .adjoint import AdjointManifold

# Layers
from .layers import MLayer as GLayer  # Alias
from .layers import MLayer, ParallelMLayer, RiemannianGating

# Geometry - All integrators
from .geometry import (
    LowRankChristoffel,
    HeunIntegrator,
    RK4Integrator,
    SymplecticIntegrator,
    LeapfrogIntegrator,
)

# Loss Functions
from .losses import (
    hamiltonian_loss,
    geodesic_regularization,
    GFNLoss,
)

# Optimizers
from .optim import (
    RiemannianAdam,
    ManifoldSGD,
)

# Datasets
from .math_dataset import MathDataset
from .mixed_dataset import MixedHFDataset

# Safety
from .safety import GPUMonitor

# Registry
INTEGRATORS = {
    'heun': HeunIntegrator,
    'rk4': RK4Integrator,
    'symplectic': SymplecticIntegrator,
    'leapfrog': LeapfrogIntegrator,
}

__all__ = [
    "GFN",
    "GLayer", "RiemannianGating",
    "LowRankChristoffel", "HeunIntegrator", "RK4Integrator", "SymplecticIntegrator", "LeapfrogIntegrator",
    "INTEGRATORS",
    "hamiltonian_loss", "geodesic_regularization", "GFNLoss",
    "RiemannianAdam", "ManifoldSGD",
    "MathDataset", "MixedHFDataset",
    "GPUMonitor",
]
