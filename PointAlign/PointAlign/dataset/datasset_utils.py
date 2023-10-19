# -*- coding: utf-8 -*-
# @Time    : 2023-05-16
# @Author  : lab
# @desc    :
import numpy as np


def farthest_point_sample(point, n_point):
    """
    Farthest point sampling algorithm
    :param point: point cloud data, [N, D]
    :param n_point: number of samples
    :return: sampled point cloud index, [n_point, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((n_point,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    index = centroids.astype(np.int32)
    return index
