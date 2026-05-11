"""
Neural Network model with multiple initialization strategies.
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = list(layer_sizes)
        self.weights = []
        self.biases = []
        self.activations = []

    def randomize(self, method="he"):
        self.weights, self.biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            if method == "xavier":
                W = np.random.randn(fi, fo) * np.sqrt(2.0 / (fi + fo))
            elif method == "he":
                W = np.random.randn(fi, fo) * np.sqrt(2.0 / fi)
            elif method == "lecun":
                W = np.random.randn(fi, fo) * np.sqrt(1.0 / fi)
            elif method == "uniform":
                lim = np.sqrt(6.0 / (fi + fo))
                W = np.random.uniform(-lim, lim, (fi, fo))
            elif method == "normal_01":
                W = np.random.randn(fi, fo) * 0.1
            elif method == "normal_1":
                W = np.random.randn(fi, fo) * 1.0
            elif method == "large":
                W = np.random.randn(fi, fo) * 5.0
            elif method == "sparse":
                W = np.random.randn(fi, fo) * 0.01
                W[np.random.rand(fi, fo) < 0.8] = 0.0
            else:
                W = np.random.randn(fi, fo) * 0.5
            self.weights.append(W)
            self.biases.append(np.random.randn(1, fo) * 0.01)
        self.activations = []

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.activations = [X.copy()]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = self._sigmoid(z) if i == len(self.weights) - 1 else np.maximum(0, z)
            self.activations.append(a.copy())
        return a

    def evaluate(self, X, y):
        out = self.forward(X)
        acc = float(np.mean((out > 0.5).astype(float) == y))
        eps = 1e-8
        loss = float(-np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps)))
        return acc, loss

    def copy_from(self, other):
        self.layer_sizes = list(other.layer_sizes)
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]
        self.activations = [a.copy() for a in other.activations] if other.activations else []