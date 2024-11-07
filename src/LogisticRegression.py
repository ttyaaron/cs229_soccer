import numpy as np


class LogisticRegression:
    def __init__(self):
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        return (1 / (2 * m)) * np.sum((h - y) ** 2)  # MSE

    def gradient(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        return (1 / m) * X.T @ (h - y)

    def hessian(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        R = np.diag(h * (1 - h))  # Diagonal matrix of the sigmoid derivative
        return (1 / m) * X.T @ R @ X

    def gradient_descent(self, X, y, alpha, num_iters):
        for i in range(num_iters):
            grad = self.gradient(X, y)
            self.theta -= alpha * grad
            if i % 10 == 0:  # Track error every 10 iterations
                print(f"Iteration {i}, Error: {self.compute_cost(X, y)}")

    def newtons_method(self, X, y, num_iters):
        for i in range(num_iters):
            grad = self.gradient(X, y)
            H = self.hessian(X, y)
            self.theta -= np.linalg.inv(H) @ grad
            if i % 10 == 0:  # Track error every 10 iterations
                print(f"Iteration {i}, Error: {self.compute_cost(X, y)}")

    def train(self, X, y, method="gradient_descent", alpha=0.01, num_iters=1000):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        self.theta = np.zeros(X.shape[1])

        if method == "gradient_descent":
            self.gradient_descent(X, y, alpha, num_iters)
        elif method == "newtons_method":
            self.newtons_method(X, y, num_iters)
        else:
            raise ValueError("Invalid method specified. Choose 'gradient_descent' or 'newtons_method'.")

        return self.compute_cost(X, y)  # Final training error

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        return self.sigmoid(X @ self.theta) >= 0.5

    def test(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)  # MSE as error metric
        return mse
