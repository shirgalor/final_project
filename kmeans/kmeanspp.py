import numpy as np
import pandas as pd
import sys
import mykmeanspp


def error_exit(message):
    print(message)
    sys.exit(1)

def parse_args():
    args = sys.argv[1:]

    if len(args) not in [4, 5]:
        error_exit("An Error Has Occurred")

    try:
        K = int(float(args[0]))
        if K <= 1:
            raise ValueError
    except:
        error_exit("Invalid number of clusters!")

    eps_arg_index = 2 if len(args) == 5 else 1
    try:
        eps = float(args[eps_arg_index])
        if eps < 0:
            raise ValueError
    except:
        error_exit("Invalid epsilon!")

    if len(args) == 5:
        try:
            iter_ = int(float(args[1]))
            if not (1 < iter_ < 1000):
                raise ValueError
        except:
            error_exit("Invalid maximum iteration!")
        file1, file2 = args[3], args[4]
    else:
        iter_ = 300
        file1, file2 = args[2], args[3]

    # print(f"K: {K}, eps: {eps}, iter: {iter_}, file1: {file1}, file2: {file2}")
    return K, eps, iter_, file1, file2

def load_and_join(file1, file2):
    try:
        df1 = pd.read_csv(file1, header=None)
        df2 = pd.read_csv(file2, header=None)
    except Exception as e:
        error_exit("Error loading input files")

    merged = pd.merge(df1, df2, on=0, how='inner')
    merged.sort_values(by=0, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    keys = merged[0].tolist() 
    points = merged.drop(columns=[0]).to_numpy(dtype=float)

    return points, keys

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def kmeanspp(points, k):
    np.random.seed(1234)  # (b) Set seed for reproducibility
    n_points = points.shape[0]
    
    # (1) Choose first center uniformly at random
    centroids = []
    indices = []

    first_idx = np.random.choice(n_points)
    centroids.append(points[first_idx])
    indices.append(first_idx)

    for _ in range(1, k):
        distances = np.array([
            min(euclidean_distance(x, c) for c in centroids)
            for x in points
        ])
        
        probabilities = distances / distances.sum()
        next_idx = np.random.choice(n_points, p=probabilities)
        centroids.append(points[next_idx])
        indices.append(next_idx)

    return np.array(centroids), indices

def main():
    K, eps, iter_, file1, file2 = parse_args()
    points, keys = load_and_join(file1, file2)
    
    if K >= points.shape[0]:
        error_exit("Invalid number of clusters!")
    
    initial_centroids, chosen_indices = kmeanspp(points, K)

    print(",".join([str(int(keys[i])) for i in chosen_indices]))
    final_centroids = mykmeanspp.fit(
        points.tolist(),
        initial_centroids.tolist(),
        K,
        iter_,
        eps
    )

    for centroid in final_centroids:
        print(",".join(f"{val:.4f}" for val in centroid))
        
if __name__ == "__main__":
    main()