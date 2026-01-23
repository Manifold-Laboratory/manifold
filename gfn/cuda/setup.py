from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Build path
cuda_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gfn_cuda',
    ext_modules=[
        CUDAExtension(
            'gfn_cuda',
            [
                'cuda_kernels.cpp',
                'src/geometry/christoffel_fused.cu',
                'src/geometry/lowrank_christoffel.cu',
                'src/geometry/reactive_christoffel.cu',
                'src/integrators/leapfrog_fused.cu',
                'src/integrators/leapfrog_backward.cu',
                'src/integrators/yoshida_fused.cu',
                'src/integrators/euler_fused.cu',
                'src/integrators/verlet_fused.cu',
                'src/integrators/forest_ruth_fused.cu',
                'src/integrators/omelyan_fused.cu',
                'src/integrators/heun_fused.cu',
                'src/integrators/rk4_fused.cu',
                'src/integrators/dormand_prince_fused.cu',
                'src/integrators/recurrent_manifold_fused.cu',
                'src/integrators/recurrent_manifold_backward.cu',
                'src/integrators/manifold_step.cu',
                'src/layers/parallel_scan_fused.cu',
                'src/layers/head_mixing.cu',
            ],
            include_dirs=[os.path.join(cuda_dir, 'include')],
            extra_compile_args={
                'cxx': ['/std:c++17', '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/permissive-', '/Zc:__cplusplus', '/Zm2000', '/wd4996'],
                'nvcc': [
                    '-O3', '--use_fast_math', '-std=c++17',
                    '-Xcompiler', '/std:c++17', 
                    '-Xcompiler', '/DNOMINMAX',
                    '-Xcompiler', '/DWIN32_LEAN_AND_MEAN',
                    '-Xcompiler', '/permissive-',
                    '-Xcompiler', '/Zc:__cplusplus',
                    '-Xcompiler', '/wd4996'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
