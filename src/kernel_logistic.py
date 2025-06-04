import numpy as np
from sklearn.metrics import accuracy_score

class KernelLogisticRegression:
    def __init__(self, kernel='rbf', gamma=0.1, lr=0.01, epochs=100):
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
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
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]

        self.alpha = np.zeros(n_samples)
        K = self.kernel(X, X)

        for epoch in range(self.epochs):
            z = K.dot(self.alpha)
            y_pred = self._sigmoid(z)
            gradient = K.T.dot(y_pred - y)
            self.alpha -= self.lr * gradient

    def predict_proba(self, X):
        K = self.kernel(X, self.X_train)
        z = K.dot(self.alpha)
        return self._sigmoid(z)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
