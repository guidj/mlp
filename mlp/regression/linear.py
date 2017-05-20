"""
Linear regression models
"""

from mlp import interfaces
from mlp import utils

import numpy as np


class BasicLinearRegression(interfaces.Model):
    def __init__(self):
        self._coefficients = None
        self._means = None

    def fit(self, X, y):
        assert isinstance(y, (np.ndarray, np.generic))
        assert len(X.shape) == 2

        self._coefficients = [0.0 for _ in range(X.shape[1] + 1)]
        self._means = [0.0 for _ in range(X.shape[1])]

        mean_y = np.mean(y)

        for c in range(X.shape[1]):
            mean_x = np.mean(X[:, c])
            self._means[c] = mean_x
            self._coefficients[c + 1] = utils.covariance(X[:, c], mean_x, y, mean_y) / utils.variance(X[:, c], mean_x)

        b0 = mean_y - np.sum([self._coefficients[c + 1] * self._means[c] for c in range(X.shape[1])])
        self._coefficients[0] = b0

        return self

    def transform(self, X):

        y = [None for _ in range(X.shape[0])]

        for r in range(X.shape[0]):
            y[r] = self._coefficients[0] + np.sum([self._coefficients[c + 1] * X[r, c] for c in range(X.shape[1])])

        return y

    @property
    def coefficients(self):
        return self._coefficients
