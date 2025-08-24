import argparse
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from symnmf.symnmf import build_similarity, normalize_symmetric, symnmf, cluster_from_H

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

def load_dataset(name: str, n: int=400, seed: int=0):
    """
    Load synthetic datasets for clustering analysis.
    
    Args:
        name (str): Dataset type - "moons" for two interleaving half circles,
                   "blobs" for Gaussian blobs
        n (int): Number of samples to generate (default: 400)
        seed (int): Random state for reproducibility (default: 0)
        
    Returns:
        tuple: (X, y) where X is feature matrix (n_samples, 2) and 
               y is true cluster labels (n_samples,)
               
    Raises:
        ValueError: If dataset name is not supported
    """
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=0.05, random_state=seed)
    elif name == "blobs":
        X, y = make_blobs(n_samples=n, centers=3, cluster_std=1.3, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y

def main():
    """
    Main function to perform Symmetric Non-negative Matrix Factorization clustering.
    
    Parses command line arguments, loads synthetic data, builds similarity matrix,
    performs SymNMF decomposition, extracts clusters, and optionally visualizes results.
    
    Command line options:
        --dataset: Dataset type ("moons" or "blobs")
        --k: Number of clusters 
        --sigma: Gaussian kernel width (auto if None)
        --knn: Use k-nearest neighbors sparsification (dense if None)
        --normalize: Apply symmetric normalization to similarity matrix
        --max-iter: Maximum SymNMF iterations
        --tol: Convergence tolerance
        --seed: Random seed for reproducibility
        --use-c: Use C implementation for faster computation
        --plot: Display clustering results (requires matplotlib)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "blobs"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--knn", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-c", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    X, y_true = load_dataset(args.dataset, n=400, seed=args.seed)
    W = build_similarity(X, sigma=args.sigma, knn=args.knn)

    if args.normalize:
        W = normalize_symmetric(W)

    H, hist = symnmf(W, k=args.k, max_iter=args.max_iter, tol=args.tol,
                     use_c=args.use_c, random_state=args.seed, verbose=True)

    labels = cluster_from_H(H, k=args.k, random_state=args.seed)

    print("Final objective:", hist["obj"][-1])
    print("Iterations:", len(hist["obj"]))

    if args.plot and HAS_PLT:
        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=labels, s=24)
        plt.title("Clusters from SymNMF + k-means on rows of H")
        plt.show()

if __name__ == "__main__":
    main()
