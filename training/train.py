import numpy as np
from tensorflow.keras.datasets import mnist

from neural_network.network import NeuralNetwork, cross_entropy_loss

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("Flattened training:", x_train.shape)
print("Flattened test:", x_test.shape)

network = NeuralNetwork()

epochs = 5
batch_size = 64
learning_rate = 0.1

for epoch in range(epochs):
    indices = np.random.permutation(len(x_train))

    x_train = x_train[indices]
    y_train = y_train[indices]

    total_loss = 0

    for start in range(0, len(x_train), batch_size):
        end = start + batch_size

        X_batch = x_train[start:end]
        y_batch = y_train[start:end]

        predictions = network.forward(X_batch)

        loss = cross_entropy_loss(predictions, y_batch)

        total_loss += loss

        network.backward(y_batch)

        network.update(learning_rate)

    average_loss = total_loss / (len(x_train) / batch_size)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")
