import numpy as np

from neural_network.network import NeuralNetwork, cross_entropy_loss


network = NeuralNetwork()

x = np.random.rand(1, 784)

label = np.array([7])


output = network.forward(x)

loss_before = cross_entropy_loss(output, label)

print("Before:")
print("Prediction:", np.argmax(output))
print("Loss:", loss_before)


network.backward(label)
network.update(0.1)


output = network.forward(x)

loss_after = cross_entropy_loss(output, label)

print()
print("After:")
print("Prediction:", np.argmax(output))
print("Loss:", loss_after)
