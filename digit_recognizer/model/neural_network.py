import numpy as np

from digit_recognizer.model.network_layer import NetworkLayer
from digit_recognizer.model.utils import deriv_sigmoid, mse_loss


LEARN_RATE = .1
EPOCHS = 100


class NeuralNetwork:
    hidden: list[NetworkLayer]
    output: NetworkLayer

    def __init__(self, amount_inputs: int, amount_hidden_layers: int, amount_neurons_in_hidden_layer: int, amount_outputs: int):
        self.hidden = []

        last_layer_size = amount_inputs

        for _ in range(amount_hidden_layers):
            self.hidden.append(NetworkLayer(amount_neurons_in_hidden_layer, last_layer_size))
            last_layer_size = amount_neurons_in_hidden_layer

        self.output = NetworkLayer(amount_outputs, last_layer_size)

    def feedforward(self, x):
        inputs = self.hidden[0].feedforward(x)
        for layer in self.hidden[1:]:
            inputs = layer.feedforward(inputs)

        return self.output.feedforward(inputs)

    def train(self, data, all_y_trues):
        for epoch in range(EPOCHS):
            for x, y_true in zip(data, all_y_trues):
                self.back_propagate(x, y_true)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

    def back_propagate(self, x, y_true):
        activations = [x]
        for layer in self.hidden:
            activations.append(layer.feedforward(activations[-1]))
        activations.append(self.output.feedforward(activations[-1]))

        out_neuron_outputs = activations[-1]

        output_errors = -2 * (y_true - out_neuron_outputs)
        out_deriv = deriv_sigmoid(out_neuron_outputs)

        delta_outs = output_errors * out_deriv
        old_output_weights = self.output.weights.copy()

        self.output.weights -= LEARN_RATE * np.outer(delta_outs, activations[-2])
        self.output.biases -= LEARN_RATE * delta_outs

        next_layer_deltas = delta_outs
        next_layer_weights = old_output_weights

        for i in reversed(range(len(self.hidden))):
            layer = self.hidden[i]
            layer_outputs = activations[i + 1]
            layer_inputs = activations[i]

            next_layer_deltas, next_layer_weights = layer.back_propagate(
                layer_outputs, next_layer_deltas, next_layer_weights, layer_inputs
            )
