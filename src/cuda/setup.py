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
                'kernels/christoffel_fused.cu',
                'kernels/leapfrog_fused.cu',
            ],
            extra_compile_args={
                'cxx': ['/std:c++17', '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/permissive-', '/Zc:__cplusplus'],
                'nvcc': [
                    '-O3', '--use_fast_math', '-std=c++17',
                    '-Xcompiler', '/std:c++17', 
                    '-Xcompiler', '/DNOMINMAX',
                    '-Xcompiler', '/DWIN32_LEAN_AND_MEAN',
                    '-Xcompiler', '/permissive-',
                    '-Xcompiler', '/Zc:__cplusplus'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
