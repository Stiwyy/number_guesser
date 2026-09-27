import numpy as np

from neural_network.network import NeuralNetwork


network = NeuralNetwork()

network.load("models/digit_model.npz")

print("Model loaded!")

print("W1:", network.W1.shape)
print("b1:", network.b1.shape)

print("W2:", network.W2.shape)
print("b2:", network.b2.shape)

print("W3:", network.W3.shape)
print("b3:", network.b3.shape)
