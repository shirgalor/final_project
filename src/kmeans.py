import numpy as np


def _euclidean(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def _get_closest_centroid(point, centroids):
    """Find the index of the closest centroid to a given point."""
    min_dis = float('inf')
    closest = 0
    for i, c in enumerate(centroids):
        dist = _euclidean(point, c)
        if dist < min_dis:
            min_dis = dist
            closest = i
    return closest


def kmeans(k, points, max_iter=300):
    """Perform k-means clustering."""
    N = len(points)
    centroids = points[:k]
    epsilon = 0.001
    delta = float('inf')
    iteration_number = 0
    while delta > epsilon and iteration_number < max_iter:
        iteration_number += 1
        clustering = {i: [] for i in range(k)}

        for p in points:
            closest = _get_closest_centroid(p, centroids)
            clustering[closest].append(p)

        delta = 0
        for i in range(k):
            if clustering[i]:
                new_centroid = [sum(dim) / len(clustering[i]) for dim in zip(*clustering[i])]
                delta = max(delta, _euclidean(centroids[i], new_centroid))
                centroids[i] = new_centroid

    return centroids


def get_labels(centroids, points):
    """Assign labels to points based on the closest centroid."""
    labels = []
    for p in points:
        min_c = _get_closest_centroid(p, centroids)
        labels.append(min_c)
    return np.array(labels)
