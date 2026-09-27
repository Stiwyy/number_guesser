import numpy as np
from tensorflow.keras.datasets import mnist

from neural_network.network import NeuralNetwork


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype(np.float32) / 255.0

x_test = x_test.reshape(-1, 784)

network = NeuralNetwork()
network.load("models/digit_model.npz")


for i in range(20):
    image = x_test[i : i + 1]

    probabilities = network.forward(image)

    prediction = np.argmax(probabilities[0])

    confidence = probabilities[0][prediction]

    print(f"Actual: {y_test[i]} Prediction: {prediction} Confidence: {confidence:.2%}")
