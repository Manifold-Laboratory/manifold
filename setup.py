from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure we are in the right directory
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gfn_kernels',
    ext_modules=[
        CUDAExtension('gfn_kernels', [
            'src/csrc/bindings.cpp',
            'src/csrc/fused_ops.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
