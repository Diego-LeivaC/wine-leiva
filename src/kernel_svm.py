import numpy as np
from sklearn.metrics import accuracy_score

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma
        self.kernel = self._rbf_kernel if kernel == 'rbf' else self._linear_kernel

    def _rbf_kernel(self, X1, X2):
        if X1.ndim == 1:
            X1 = X1[np.newaxis, :]
        if X2.ndim == 1:
            X2 = X2[np.newaxis, :]
        sq_dists = -2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1) + np.sum(X1**2, axis=1)[:, np.newaxis]
        return np.exp(-self.gamma * sq_dists)

    def _linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def fit(self, X, y, epochs=100):
        n_samples = X.shape[0]
        self.X = X
        self.y = np.where(y == 0, -1, 1)  # Convert to {-1, 1}
        self.alpha = np.zeros(n_samples)
        self.b = 0

        K = self.kernel(X, X)

        for epoch in range(epochs):
            for i in range(n_samples):
                margin = np.sum(self.alpha * self.y * K[:, i]) + self.b
                if self.y[i] * margin < 1:
                    self.alpha[i] += self.C
                    self.b += self.C * self.y[i]

    def project(self, X):
        K = self.kernel(X, self.X)
        return np.dot(K, self.alpha * self.y) + self.b

    def predict(self, X):
        return (self.project(X) > 0).astype(int)
