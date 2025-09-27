import os
import argparse

import numpy as np

from sklearn.metrics import silhouette_score
from kmeans import kmeans, get_labels
from symnmf import handle_goal, read_data_points


def _parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run symnmf using C extension")
    parser.add_argument('k', type=int, help='Number of clusters')
    parser.add_argument('file_name', type=str, help='Input file path')
    return parser.parse_args()


def main():
    np.random.seed(1234)
    args = _parse_arguments()
    assert os.path.exists(args.file_name), "File does not exist"
    X = read_data_points(args.file_name)
    assert 1 < args.k < X.shape[0], "Invalid number of clusters"
    k = args.k

    H = handle_goal('symnmf', X, k)
    nmf_labels = np.argmax(H, axis=1)
    centroids = kmeans(k, X.tolist())
    kmeans_labels = get_labels(centroids, X)
    
    nmf_score = silhouette_score(X, nmf_labels)
    kmeans_score = silhouette_score(X, kmeans_labels)
    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")

  
if __name__ == "__main__":
    main()
    