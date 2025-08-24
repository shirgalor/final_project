import numpy as np
from symnmf.symnmf import build_similarity, normalize_symmetric, symnmf, cluster_from_H

def test_small_run():
    """
    Basic integration test for SymNMF pipeline.
    
    Creates synthetic 2-cluster data, builds similarity matrix,
    applies symmetric normalization, runs SymNMF, and extracts clusters.
    Validates shapes and basic functionality.
    """
    np.random.seed(0)
    X = np.vstack([np.random.randn(3,2)*0.1 + [0,0], np.random.randn(3,2)*0.1 + [5,5]])
    W = build_similarity(X, sigma=None, knn=None)
    Wn = normalize_symmetric(W)
    H, hist = symnmf(Wn, k=2, max_iter=50, tol=1e-5, use_c=False, random_state=0, verbose=False)
    assert H.shape == (6,2)
    assert len(hist["obj"]) >= 1
    labels = cluster_from_H(H, 2, random_state=0)
    assert set(labels.tolist()) <= {0,1}
