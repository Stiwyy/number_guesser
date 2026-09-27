import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros((1, 128))

        self.W2 = np.random.randn(128, 64) * 0.01
        self.b2 = np.zeros((1, 64))

        self.W3 = np.random.randn(64, 10) * 0.01
        self.b3 = np.zeros((1, 10))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)

        z3 = a2 @ self.W3 + self.b3

        return softmax(z3)
