import numpy as np
from scipy.spatial.distance import cdist

class RBF(object):
    def __init__(self, lengthscale=1.0, variance=1.0):
        """
        RBF Kernel with k(x,z) = var * np.exp(-0.5 * ||x-z||^2/length^2)
        """
        self.length = lengthscale
        self.var = variance

    def compute(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        dist = cdist(X1, X2)
        return self.var * np.exp(-0.5 * np.square(dist) / self.length**2)

    def compute_diag(self, X1):
        return np.identity(X1.shape[0]) * self.var