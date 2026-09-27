import numpy as np

from neural_network.network import NeuralNetwork


network = NeuralNetwork()

x = np.random.rand(1, 784)

output = network.forward(x)

print(output)
print(output.shape)
print("Sum:", np.sum(output))
print("Prediction:", np.argmax(output))
