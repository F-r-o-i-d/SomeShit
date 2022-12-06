import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

X = [[1  , 2  , 3   , 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]
        
X, y = spiral_data(samples=100, classes=3)
class Layer_Dense:
    def __init__(self, n_input, n_neurons) -> None:
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLu:
    def __init__(self) -> None:
        pass

    def forward(self, inputs):
        self.output = np.maximum(0,inputs)


if __name__ == "__main__":
    L1 = Layer_Dense(2, 5)
    A1 = Activation_ReLu()

    L1.forward(X)
    print(L1.output)
    A1.forward(L1.output)
    print(A1.output)