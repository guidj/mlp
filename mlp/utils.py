import numpy as np


def covariance(x, mean_x, y, mean_y):
    cov = np.sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])

    return cov


def variance(x, mean_x=None):

    if mean_x is None:
        mean_x = np.mean(x)

    return np.sum([(x[i] - mean_x)**2 for i in range(len(x))])
