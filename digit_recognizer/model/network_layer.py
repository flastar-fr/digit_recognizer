from typing import Iterable

import numpy as np

from digit_recognizer.model.neuron import Neuron
from digit_recognizer.model.utils import deriv_sigmoid, LEARN_RATE, sigmoid


class NetworkLayer:
    weights: float | np.ndarray
    biases: float | np.ndarray

    def __init__(self, amount_neurons: int, amount_weights: int):
        self.weights = np.random.normal(size=(amount_neurons, amount_weights))
        self.biases = np.random.normal(size=(amount_neurons,))

    def feedforward(self, inputs: Iterable) -> np.ndarray:
        return sigmoid(np.dot(self.weights, inputs) + self.biases)

    def back_propagate(self, hidden_outputs, next_layer_deltas, next_layer_old_weights, inputs):
        hidden_deriv = deriv_sigmoid(hidden_outputs)
        error_sum = np.dot(next_layer_deltas, next_layer_old_weights)

        my_deltas = error_sum * hidden_deriv

        old_weights = self.weights.copy()

        self.weights -= LEARN_RATE * np.outer(my_deltas, inputs)
        self.biases -= LEARN_RATE * my_deltas

        return my_deltas, old_weights
