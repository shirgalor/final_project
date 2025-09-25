import sys

import numpy as np


def _euclidean(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def _read_input_data():
    lines = sys.stdin.read().strip().split('\n')
    points = [list(map(float, line.split(','))) for line in lines]
    return points, len(points)


def _get_closest_centroid(point, centroids):
    min_dis = float('inf')
    min_c = 0
    for i, c in enumerate(centroids):
        dist = _euclidean(point, c)
        if dist < min_dis:
            min_dis = dist
            min_c = i
    return min_c


def kmeans(k, points, max_iter=300):
    N = len(points)
    centroids = points[:k]
    epsilon = 0.001
    delta_val = float('inf')
    iteration_number = 0
    while delta_val > epsilon and iteration_number < max_iter:
        iteration_number += 1
        clustering = {i: [] for i in range(k)}

        for p in points:
            min_c = _get_closest_centroid(p, centroids)
            clustering[min_c].append(p)

        delta_val = 0
        for i in range(k):
            if clustering[i]:
                new_centroid = [sum(dim) / len(clustering[i]) for dim in zip(*clustering[i])]
                delta_val = max(delta_val, _euclidean(centroids[i], new_centroid))
                centroids[i] = new_centroid

    return centroids


def get_labels(centroids, points):
    labels = []
    for p in points:
        min_c = _get_closest_centroid(p, centroids)
        labels.append(min_c)
    return np.array(labels)
