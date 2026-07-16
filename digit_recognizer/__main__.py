from pathlib import Path

import numpy as np

from digit_recognizer.model.neural_network import NeuralNetwork


def based_tests_datas():
    return np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ]), np.array([
        [1],
        [0],
        [0],
        [1],
    ])


def load_mnist_ubyte(images_path: Path, labels_path: Path):
    with open(labels_path, "rb") as f:
        magic, num_labels = np.fromfile(f, dtype=">i4", count=2)
        labels = np.fromfile(f, dtype="u1")

    with open(images_path, "rb") as f:
        magic, num_images, rows, cols = np.fromfile(f, dtype=">i4", count=4)
        imagesbrutes = np.fromfile(f, dtype="u1")

        images = imagesbrutes.reshape(num_images, rows * cols)

    data = images.astype(np.float32) / 255.0

    all_y_trues = np.zeros((num_labels, 10))
    all_y_trues[np.arange(num_labels), labels] = 1.0

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    all_y_trues = all_y_trues[indices]

    return data, all_y_trues



print("Start data preparation")
training_path_images = Path("database/train-images.idx3-ubyte")
training_path_labels = Path("database/train-labels.idx1-ubyte")
training_data, training_all_y_trues = load_mnist_ubyte(training_path_images, training_path_labels)

print("End of data preparation")
network = NeuralNetwork(
    len(training_data[0]),
    1,
    16,
    len(training_all_y_trues[0])
)

print("Start training")
network.train(training_data, training_all_y_trues)
print("End training")

print("Start testing")
testing_path_images = Path("database/t10k-images.idx3-ubyte")
testing_path_labels = Path("database/t10k-labels.idx1-ubyte")
testing_data, testing_all_y_trues = load_mnist_ubyte(testing_path_images, testing_path_labels)
y_preds = np.apply_along_axis(network.feedforward, 1, testing_data)

predictions = np.argmax(y_preds, axis=1)
real_values = np.argmax(testing_all_y_trues, axis=1)

accuracy = np.mean(predictions == real_values) * 100
print(f"Précision sur le dataset d'entraînement : {accuracy:.2f}%")
