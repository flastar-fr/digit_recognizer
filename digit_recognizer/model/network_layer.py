import numpy as np

from digit_recognizer.model.neuron import Neuron
from digit_recognizer.model.utils import deriv_sigmoid, LEARN_RATE


class NetworkLayer:
    def __init__(self, amount_neurons: int, amount_weights: int):
        self.neurons = [
            Neuron(weights=np.random.normal(size=(amount_weights,)), bias=np.random.normal())
            for _ in range(amount_neurons)
        ]

    def feedforward(self, inputs: list[float | int]) -> list[float | int]:
        return [neuron.feedforward(inputs) for neuron in self.neurons]

    def back_propagate(self, hidden_outputs, next_layer_deltas, next_layer_old_weights, inputs):
        my_deltas = []
        my_old_weights = []

        for j, neuron in enumerate(self.neurons):
            neuron_out = hidden_outputs[j]
            hidden_deriv = deriv_sigmoid(neuron_out)

            error_sum = 0
            for next_neuron_idx in range(len(next_layer_deltas)):
                error_sum += next_layer_deltas[next_neuron_idx] * next_layer_old_weights[next_neuron_idx][j]

            my_delta = error_sum * hidden_deriv
            my_deltas.append(my_delta)

            my_old_weights.append(list(neuron.weights))

            for k, weight in enumerate(neuron.weights):
                corresponding_input = inputs[k]
                neuron.weights[k] -= LEARN_RATE * (my_delta * corresponding_input)

            neuron.bias -= LEARN_RATE * my_delta

        return my_deltas, my_old_weights
