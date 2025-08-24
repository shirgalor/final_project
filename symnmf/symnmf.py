import numpy as np
from numpy.linalg import norm
from typing import Optional, Tuple, Dict

try:
    from . import _csymnmf as CEXT
    _HAS_C = True
except Exception:
    _HAS_C = False

def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance matrix efficiently.
    
    Uses the identity ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    to avoid explicit loops.
    
    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Squared distance matrix of shape (n_samples, n_samples)
    """
    G = X @ X.T
    sq = np.diag(G).reshape(-1, 1)
    D2 = sq + sq.T - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    return D2

def build_similarity(X: np.ndarray, sigma: Optional[float]=None, knn: Optional[int]=None,
                     self_connectivity: bool=False) -> np.ndarray:
    """
    Build a Gaussian-kernel similarity matrix from data points.
    
    Creates similarity matrix W[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2)).
    Optionally sparsifies using k-nearest neighbors and controls self-loops.
    
    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features)
        sigma (float, optional): Gaussian kernel bandwidth. Auto-estimated if None
        knn (int, optional): Keep only k nearest neighbors. Dense if None
        self_connectivity (bool): Include self-loops (diagonal elements)
        
    Returns:
        np.ndarray: Symmetric similarity matrix of shape (n_samples, n_samples)
    """
    X = np.asarray(X, dtype=float)
    D2 = _pairwise_sq_dists(X)
    if sigma is None:
        tri = D2[np.triu_indices_from(D2, k=1)]
        med = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
        sigma = np.sqrt(med) if med > 0 else 1.0
    W = np.exp(-D2 / (2.0 * sigma * sigma))
    if not self_connectivity:
        np.fill_diagonal(W, 0.0)
    if knn is not None and knn > 0:
        n = W.shape[0]
        mask = np.zeros_like(W, dtype=bool)
        idx = np.argsort(-W, axis=1)[:, :knn]
        rows = np.arange(n)[:, None]
        mask[rows, idx] = True
        W = np.where(mask | mask.T, W, 0.0)
    W = 0.5 * (W + W.T)
    return W

def degree(W: np.ndarray) -> np.ndarray:
    """
    Compute degree vector of similarity matrix.
    
    Args:
        W (np.ndarray): Symmetric similarity matrix
        
    Returns:
        np.ndarray: Degree vector where d[i] = sum of row i
    """
    return W.sum(axis=1)

def normalize_symmetric(W: np.ndarray) -> np.ndarray:
    """
    Apply symmetric normalization to similarity matrix.
    
    Computes D^(-1/2) * W * D^(-1/2) where D is diagonal degree matrix.
    Handles zero degrees by setting corresponding elements to 0.
    
    Args:
        W (np.ndarray): Similarity matrix
        
    Returns:
        np.ndarray: Symmetrically normalized matrix
    """
    d = degree(W)
    with np.errstate(divide='ignore'):
        inv_sqrt_d = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt_d)
    return D_inv_sqrt @ W @ D_inv_sqrt

def _init_h(n: int, k: int, W: np.ndarray, random_state: Optional[int]=None) -> np.ndarray:
    """
    Initialize factor matrix H for SymNMF.
    
    Args:
        n (int): Number of data points
        k (int): Number of factors/clusters
        W (np.ndarray): Similarity matrix for scaling
        random_state (int, optional): Random seed
        
    Returns:
        np.ndarray: Initial H matrix of shape (n, k)
    """
    rng = np.random.RandomState(random_state)
    scale = np.sqrt(np.maximum(W.mean(), 1e-8))
    H = np.abs(rng.randn(n, k)) * scale
    return H

def _objective(W: np.ndarray, H: np.ndarray) -> float:
    """
    Compute SymNMF objective function ||W - HH^T||_F^2.
    
    Args:
        W (np.ndarray): Target similarity matrix
        H (np.ndarray): Factor matrix
        
    Returns:
        float: Frobenius norm squared of reconstruction error
    """
    R = W - H @ H.T
    return float(np.sum(R*R))

def _update_py(W: np.ndarray, H: np.ndarray, max_iter: int, tol: float, verbose: bool) -> Dict[str, list]:
    """
    Pure Python implementation of SymNMF multiplicative updates.
    
    Iteratively updates H using: H *= (WH) / (HH^TH + eps)
    
    Args:
        W (np.ndarray): Similarity matrix
        H (np.ndarray): Factor matrix (modified in-place)
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance
        verbose (bool): Print progress
        
    Returns:
        Dict[str, list]: History with objective values
    """
    eps = 1e-12
    hist = {"obj": []}
    obj_prev = _objective(W, H)
    hist["obj"].append(obj_prev)
    for it in range(max_iter):
        WH = W @ H
        HHTH = H @ (H.T @ H)
        denom = HHTH + eps
        H *= WH / denom
        np.nan_to_num(H, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        obj = _objective(W, H)
        hist["obj"].append(obj)
        if verbose:
            print(f"[py] iter={it+1:03d}, obj={obj:.6e}, rel_change={(obj_prev-obj)/max(obj_prev,1e-12):.3e}")
        if abs(obj_prev - obj) / max(obj_prev, 1.0) < tol:
            break
        obj_prev = obj
    return hist

def symnmf(W: np.ndarray, k: int, init: str="rand", max_iter: int=300, tol: float=1e-4,
           use_c: bool=False, random_state: Optional[int]=None, verbose: bool=False):
    """
    Symmetric Non-negative Matrix Factorization.
    
    Factorizes W ≈ HH^T where H ≥ 0, minimizing ||W - HH^T||_F^2.
    
    Args:
        W (np.ndarray): Symmetric similarity matrix (n x n)
        k (int): Number of factors/clusters
        init (str): Initialization method (currently only "rand")
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance
        use_c (bool): Use C implementation if available
        random_state (int, optional): Random seed
        verbose (bool): Print progress
        
    Returns:
        tuple: (H, history) where H is factor matrix (n x k) and 
               history contains objective values
               
    Raises:
        ValueError: If W is not square or contains negative values
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square (n x n)")
    if np.any(W < 0):
        raise ValueError("W must be nonnegative")
    H = _init_h(n, k, W, random_state=random_state)
    if use_c and _HAS_C:
        H_out, obj_hist = CEXT.update(H, W, int(max_iter), float(tol), int(verbose))
        try:
            obj_hist_list = obj_hist.tolist()
        except Exception:
            obj_hist_list = [0.0]
        return H_out, {"obj": obj_hist_list}
    else:
        hist = _update_py(W, H, max_iter=max_iter, tol=tol, verbose=verbose)
        return H, hist

def cluster_from_H(H: np.ndarray, k: int, random_state: Optional[int]=None) -> np.ndarray:
    """
    Extract cluster labels from SymNMF factor matrix H.
    
    Applies k-means clustering to rows of H to obtain final cluster assignments.
    
    Args:
        H (np.ndarray): Factor matrix from SymNMF (n_samples x k)
        k (int): Number of clusters
        random_state (int, optional): Random seed for k-means
        
    Returns:
        np.ndarray: Cluster labels for each sample (n_samples,)
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return km.fit_predict(H)
