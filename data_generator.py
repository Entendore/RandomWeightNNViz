"""
Dataset generators for 2D patterns, 4×4 digit patterns, and hand-sign patterns.
"""

import numpy as np


class DataGenerator:
    @staticmethod
    def get_shape(name):
        if name == "digits_4x4":
            return 16, 3
        if name == "hand_signs":
            return 5, 4
        return 2, 1

    @staticmethod
    def generate(name, n=400):
        if name == "xor":
            X = np.random.randn(n, 2) * 0.5
            y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)

        elif name == "circle":
            X = np.random.randn(n, 2) * 1.5
            y = (np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(float).reshape(-1, 1)

        elif name == "spiral":
            n2 = n // 2
            X, y = np.zeros((n, 2)), np.zeros((n, 1))
            for c in range(2):
                for i in range(n2):
                    r = i / n2 * 3
                    t = i / n2 * 4 * np.pi + c * np.pi
                    X[c * n2 + i] = [r * np.cos(t) + np.random.randn() * 0.15,
                                     r * np.sin(t) + np.random.randn() * 0.15]
                    y[c * n2 + i] = c

        elif name == "moons":
            n2 = n // 2
            th = np.linspace(0, np.pi, n2)
            X0 = np.column_stack([np.cos(th) + np.random.randn(n2) * 0.1,
                                   np.sin(th) + np.random.randn(n2) * 0.1])
            X1 = np.column_stack([1 - np.cos(th) + np.random.randn(n2) * 0.1,
                                   -np.sin(th) + 0.5 + np.random.randn(n2) * 0.1])
            X, y = np.vstack([X0, X1]), np.vstack([np.zeros((n2, 1)), np.ones((n2, 1))])

        elif name == "digits_4x4":
            h_line = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            v_line = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
            d_line = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
            classes = [h_line, v_line, d_line]
            X, y = [], []
            for _ in range(n):
                c = np.random.randint(0, 3)
                X.append(np.clip(classes[c] + np.random.randn(16) * 0.3, 0, 1))
                lab = np.zeros(3)
                lab[c] = 1
                y.append(lab)
            X, y = np.array(X), np.array(y)

        elif name == "hand_signs":
            fist = np.array([0, 0, 0, 0, 0])
            peace = np.array([0, 1, 1, 0, 0])
            point = np.array([0, 1, 0, 0, 0])
            open_h = np.array([1, 1, 1, 1, 1])
            classes = [fist, peace, point, open_h]
            X, y = [], []
            for _ in range(n):
                c = np.random.randint(0, 4)
                X.append(np.clip(classes[c] + np.random.randn(5) * 0.2, 0, 1))
                lab = np.zeros(4)
                lab[c] = 1
                y.append(lab)
            X, y = np.array(X), np.array(y)

        else:  # gaussian
            n2 = n // 2
            X = np.vstack([np.random.randn(n2, 2) * 0.5 + [-1, -1],
                           np.random.randn(n2, 2) * 0.5 + [1, 1]])
            y = np.vstack([np.zeros((n2, 1)), np.ones((n2, 1))])

        return X, y