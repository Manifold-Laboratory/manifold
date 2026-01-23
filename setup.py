import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Define CUDA Extension
def get_cuda_extension():
    cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gfn', 'cuda')
    
    # Windows flags
    is_windows = os.name == 'nt'
    cflags = ['/O2'] if is_windows else ['-O3', '-fPIC']
    nvcc_flags = ['-O3', '--use_fast_math']
    
    if is_windows:
        nvcc_flags.extend(['-allow-unsupported-compiler', '-D_ALLOW_COMPILER_SUBSTITUTIONS'])
    else:
        nvcc_flags.append('--compiler-options=-fPIC')

    return CUDAExtension(
        name='gfn_cuda', # Direct import name
        sources=[
            'gfn/cuda/cuda_kernels.cpp',
            'gfn/cuda/src/geometry/christoffel_fused.cu',
            'gfn/cuda/src/integrators/leapfrog_fused.cu',
            'gfn/cuda/src/integrators/yoshida_fused.cu',
            'gfn/cuda/src/layers/parallel_scan_fused.cu',
        ],
        extra_compile_args={
            'cxx': cflags,
            'nvcc': nvcc_flags
        }
    )

setup(
    name='manifold-gfn',
    version='2.7.0',
    description='Manifold GFN: Geometric Flow Networks with Symplectic Integration',
    author='Manifold Laboratory',
    packages=find_packages(),
    ext_modules=[get_cuda_extension()] if torch.cuda.is_available() else [],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'tqdm',
    ],
    extras_require={
        'full': [
            'transformers>=4.30.0',
            'datasets>=2.0.0',
            'huggingface-hub',
            'torchdiffeq>=0.2.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.12.0',
        ],
        'dev': [
            'pytest',
            'black',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
