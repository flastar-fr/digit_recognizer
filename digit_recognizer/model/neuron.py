import numpy as np

from digit_recognizer.model.utils import sigmoid


class Neuron:
    weights: np.ndarray
    bias: float | int

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs) -> np.ndarray:
        return sigmoid(np.dot(inputs, self.weights) + self.bias)
