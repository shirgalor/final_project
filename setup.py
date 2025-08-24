"""
Setup script for SymNMF package.

Builds the package with optional C extensions for performance acceleration.
The C extension provides faster multiplicative updates for large matrices.
"""

from setuptools import setup, Extension, find_packages
try:
    import numpy as np
    numpy_includes = [np.get_include()]
except Exception:
    numpy_includes = []

ext_modules = [
    Extension(
        name="symnmf._csymnmf",
        sources=["symnmf/symnmfmodule.c", "symnmf/symnmf.c"],
        include_dirs=numpy_includes,
        extra_compile_args=[],
        extra_link_args=[],
    )
]

setup(
    name="symnmf",
    version="0.1.0",
    description="Symmetric NMF for clustering with optional C acceleration",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["numpy", "scipy", "scikit-learn"],
)
