import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(predictions, labels):
    batch_size = predictions.shape[0]

    probabilities = predictions[np.arange(batch_size), labels]

    probabilities = np.clip(probabilities, 1e-12, 1.0)

    loss = -np.log(probabilities)

    return np.mean(loss)


def accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)

    return np.mean(predicted_labels == labels)


class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros((1, 128))

        self.W2 = np.random.randn(128, 64) * 0.01
        self.b2 = np.zeros((1, 64))

        self.W3 = np.random.randn(64, 10) * 0.01
        self.b3 = np.zeros((1, 10))

    def forward(self, X):
        self.X = X

        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.output = softmax(self.z3)

        return self.output

    def backward(self, labels):
        batch_size = self.X.shape[0]

        dz3 = self.output.copy()
        dz3[np.arange(batch_size), labels] -= 1
        dz3 /= batch_size

        self.dW3 = self.a2.T @ dz3
        self.db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)

        self.dW2 = self.a1.T @ dz2
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        self.dW1 = self.X.T @ dz1
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

        self.W3 -= learning_rate * self.dW3
        self.b3 -= learning_rate * self.db3
