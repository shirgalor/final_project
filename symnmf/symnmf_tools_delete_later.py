
import argparse
import math
import numpy as np

np.random.seed(1234)


def hard_clustering(H):
    return np.argmax(H, axis=1)

def train_H(H, W, beta=0.5, eps=1e-6, max_iter=500):
    for t in range(max_iter):
        H_old = H.copy()
        update_H(H, W, beta=beta)  # from before
        diff = np.linalg.norm(H - H_old, 'fro')**2
        if diff < eps:
            print(f"Converged at iteration {t}, diff={diff:.2e}")
            break
    return H


def update_H(H, W, beta=0.5, eps=1e-12):
    """
    One SymNMF-style multiplicative update step:
    H <- H * (1 - beta + beta * (W @ H) / (H @ (H.T @ H)))
    
    Args:
        H: (n, k) nonnegative matrix
        W: (n, n) symmetric nonnegative matrix (target ≈ H @ H.T)
        beta: relaxation parameter in [0, 1] (beta=1 gives the classic MU)
        eps: small number to avoid division by zero
    Returns:
        Updated H (in-place and also returned for convenience)
    """
    WH = W @ H                                 # (n, k)
    HHTH = H @ (H.T @ H)                       # (n, k)
    ratio = WH / (HHTH + eps)                  # elementwise
    H *= (1 - beta) + beta * ratio             # elementwise, keeps H ≥ 0
    return H

def initialize_H(W, k):
    """
    Initialize H for SymNMF.

    W : numpy.ndarray (n x n) - normalized similarity matrix
    k : int - number of clusters

    Returns:
        H : numpy.ndarray (n x k)
    """
    n = W.shape[0]
    m = np.mean(W)
    upper = 2 * math.sqrt(m / k)
    # use np.random.uniform() exactly as spec says
    H = np.random.uniform(0, upper, size=(n, k))
    return H

def build_normalized_similarity_matrix(A, D):
    """
    Compute the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2).
    
    A : numpy.ndarray (n x n) - similarity matrix
    D : numpy.ndarray (n x n) - degree matrix (diagonal)
    
    Returns:
        W : numpy.ndarray (n x n)
    """
    # Ensure numpy arrays
    A = np.array(A, dtype=float)
    D = np.array(D, dtype=float)

    # Inverse square root of D (only diagonal entries)
    D_inv_sqrt = np.diag([1.0 / np.sqrt(D[i, i]) if D[i, i] > 0 else 0.0 for i in range(len(D))])

    # Compute W
    W = D_inv_sqrt @ A @ D_inv_sqrt
    return W


def build_degree_matrix(A):
    """
    Build the diagonal degree matrix D from similarity matrix A.
    A is assumed to be an n x n 2D list or nested list.
    
    Formula:
        d_i = sum_j a_ij
        D = diag(d_1, ..., d_n)
    """
    n = len(A)
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        d_i = sum(A[i])   # row sum
        D[i][i] = d_i
    return D


def build_similarity_matrix(points):
    """
    Build the similarity matrix A from a list of points.
    Each row in `points` is a vector (list of floats).
    
    Formula:
        a_ij = exp(-||x_i - x_j||^2 / 2) if i != j
        a_ii = 0
    """
    n = len(points)            # number of points
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # squared Euclidean distance
                dist2 = sum((points[i][k] - points[j][k])**2 for k in range(len(points[i])))
                A[i][j] = math.exp(-dist2 / 2.0)
            else:
                A[i][j] = 0.0  # diagonal
    return A

def read_file_to_2d_array(filename: str):
    """
    Reads a text file into a 2D array (list of lists of floats).
    Assumes each line in the file is comma- or space-separated numbers.
    """
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            # split on comma or whitespace, ignore empty strings
            row = [float(x) for x in line.strip().replace(",", " ").split() if x]
            matrix.append(row)
    return matrix

def print_matrix(mat: np.ndarray):
    for r in mat:
        print(",".join(f"{v:.4f}" for v in r))