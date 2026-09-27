import numpy as np

from neural_network.network import NeuralNetwork, cross_entropy_loss


network = NeuralNetwork()

x = np.random.rand(1, 784)

output = network.forward(x)

label = np.array([7])

loss = cross_entropy_loss(output, label)

print("Output:")
print(output)

print()

print("Output shape:")
print(output.shape)

print()

print("Probability sum:")
print(np.sum(output))

print()

print("Prediction:")
print(np.argmax(output))

print()

print("Actual:")
print(label[0])

print()

print("Loss:")
print(loss)
