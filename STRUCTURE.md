# Project Structure
.
MANIFOLD/
├── README.md                  # Main documentation
├── LICENSE                    # Apache License 2.0
├── BRAND.md                   # Brand guidelines
├── CONTRIBUTING.md            # Contribution guide
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
│
├── src/                       # Source code
│   ├── __init__.py           # Package init
│   ├── model.py              # MANIFOLD, AdjointMANIFOLD
│   ├── layers.py             # M-Layer implementation
│   ├── geometry.py           # Christoffel, Integrators
│   ├── losses.py             # GFNLoss
│   ├── optim.py              # RiemannianAdam
│   ├── math_dataset.py       # Math dataset
│   ├── mixed_dataset.py      # HuggingFace datasets
│   ├── safety.py             # GPU monitoring
│   ├── visualize.py          # Visualization utils
│   ├── adjoint.py            # Adjoint method
│   ├── cuda/                 # CUDA kernels (optional)
│   │   ├── README.md
│   │   ├── ops.py            # Python bindings
│   │   ├── cuda_kernels.cpp  # C++ bindings
│   │   ├── kernels/
│   │   │   ├── christoffel_fused.cu
│   │   │   └── leapfrog_fused.cu
│   │   └── tests/
│   │       ├── test_kernel_load.py
│   │       └── precompile_kernels.py
│   ├── data/                 # (Internal - unused)
│   └── utils/                # Utility functions
│       └── README.md
│
├── data/                      # Training data
│   ├── README.md
│   ├── math_10k.txt          # Math problems
│   └── math_examples_8digits.txt
│
├── configs/                   # Configuration files
│   ├── model/                # Model configs
│   │   ├── gfn_tiny.yaml
│   │   ├── gfn_small.yaml
│   │   ├── gfn_medium.yaml
│   │   └── gfn_large.yaml
│   ├── training/             # Training configs
│   │   └── experiment_medium.yaml
│   └── hardware/             # Hardware configs
│       └── gtx_1650.yaml
│
├── scripts/                   # Executable scripts
│   └── train.py              # Main training script
│
├── tests/                     # All tests
│   ├── unit/                 # Unit tests
│   │   ├── test_geometry.py
│   │   ├── test_layers.py
│   │   └── test_golden_integration.py
│   ├── professional/         # Benchmarks
│   │   ├── README.md
│   │   ├── baselines.py      # MicroGPT baseline
│   │   ├── benchmark_performance.py
│   │   ├── benchmark_ood.py
│   │   ├── benchmark_composition.py
│   │   ├── vis_manifold.py
│   │   └── results/          # Benchmark outputs
│   └── test_integration.py   # Integration tests
│
├── docs/                      # Documentation
│   ├── README.md             # Docs index
│   ├── ARCHITECTURE.md       # Architecture
│   ├── PHYSICS.md            # Math foundations
│   ├── API.md                # API reference
│   ├── TRAINING.md           # Training guide
│   ├── CONCEPTS.md           # Key concepts
│   ├── MODELS_AND_MECHANICS.md
│   ├── JUSTIFICATION.md      # Research justification
│   └── paper.md              # Draft paper
│
├── checkpoints/               # Model checkpoints
│   └── medium_fast/4_digits/
│       ├── *.pt              # Checkpoint files
│       ├── config.yaml       # Config snapshot
│       └── train.txt         # Training log
│
└── Compilation Scripts
    ├── compile_cuda_kernels.bat
    ├── precompile_kernels_once.bat
    └── SConstruct            # SCons build (alternative)

## Key Files

### Essential
- `src/model.py` - Main MANIFOLD class
- `src/geometry.py` - Core geometric operations
- `scripts/train.py` - Training entry point
- `README.md` - Main documentation

### Optional (Performance)
- `src/cuda/` - Custom CUDA kernels for 5-10x speedup
- `compile_cuda_kernels.bat` - Kernel compilation

### Configuration
- `configs/` - All YAML configs
- `requirements.txt` - Dependencies

### Documentation
- `docs/` - Technical documentation
- `BRAND.md` - Branding guidelines
- `CONTRIBUTING.md` - How to contribute
