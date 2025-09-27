"""
Setup script for SymNMF package.

Builds the package with C extensions for SymNMF computations.
The C extension provides similarity matrix computation, degree matrix, 
normalized similarity matrix, and full SymNMF algorithm.
"""

from setuptools import setup, Extension, find_packages
import numpy as np

# Define the C extension module
ext_modules = [
    Extension(
        name="symnmf_c",  # This will create the symnmf module
        sources=[
            "./symnmfmodule.c",  # Python C API wrapper
            "./symnmf_tools.c",  # Core C functions
        ],
        include_dirs=[
            np.get_include(),  # NumPy headers
        ],
    )
]

setup(
    name="symnmf_c",
    version="1.0.0",
    description="Symmetric Non-negative Matrix Factorization with C acceleration",
    author="Sheer and Maya",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["numpy"],
    python_requires=">=3.6",
    zip_safe=False,
)
