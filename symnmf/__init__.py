"""
SymNMF: Symmetric Non-negative Matrix Factorization for Clustering

This package provides tools for clustering using Symmetric NMF, which factorizes
a similarity matrix W ≈ HH^T to discover cluster structure. The main functions are:

- build_similarity: Create Gaussian kernel similarity matrices
- normalize_symmetric: Apply symmetric normalization D^(-1/2) W D^(-1/2)  
- symnmf: Perform the factorization with optional C acceleration
- cluster_from_H: Extract final cluster labels via k-means on H
- degree: Compute degree vector from similarity matrix
"""

from .symnmf import build_similarity, degree, normalize_symmetric, symnmf, cluster_from_H
