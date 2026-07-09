import json

import numpy as np

from digit_recognizer.model.neural_network import NeuralNetwork


data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])
all_y_trues = np.array([
    [1],
    [0],
    [0],
    [1],
])


with open('database/dataset_multioutput.json') as f:
    dataset = json.load(f)


data = np.array([
    [
        d["vibration_level"]/12,
        d["temperature"]/110,
        d["rotation_speed"]/3500,
        d["operating_hours"]/10_000,
    ]
    for d in dataset
])

all_y_trues = np.array([
    [
        d["remaining_useful_life"]/500,
        d["failure_risk"]
    ]
    for d in dataset
])

print(data)
print(all_y_trues)


network = NeuralNetwork(
    4,
    2,
    2,
    2
)

network.train(data, all_y_trues)
