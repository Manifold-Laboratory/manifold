from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure we are in the right directory
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gfn_cuda',
    ext_modules=[
        CUDAExtension(
            'gfn_cuda', 
            [
                'gfn/cuda/cuda_kernels.cpp',
                'gfn/cuda/src/geometry/christoffel_fused.cu',
                'gfn/cuda/src/integrators/euler_fused.cu',
                'gfn/cuda/src/integrators/leapfrog_fused.cu',
                'gfn/cuda/src/integrators/yoshida_fused.cu',
                'gfn/cuda/src/integrators/verlet_fused.cu',
                'gfn/cuda/src/integrators/forest_ruth_fused.cu',
                'gfn/cuda/src/integrators/omelyan_fused.cu',
                'gfn/cuda/src/integrators/heun_fused.cu',
                'gfn/cuda/src/integrators/rk4_fused.cu',
                'gfn/cuda/src/integrators/dormand_prince_fused.cu',
                'gfn/cuda/src/integrators/recurrent_manifold_fused.cu',
                'gfn/cuda/src/layers/parallel_scan_fused.cu',
            ],
            include_dirs=[os.path.join(os.getcwd(), 'gfn', 'cuda', 'include')],
            extra_compile_args={
                'cxx': ['/std:c++17', '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/Zm2000'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
