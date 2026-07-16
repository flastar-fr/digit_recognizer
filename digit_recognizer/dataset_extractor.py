import os
import struct
import numpy as np
from PIL import Image


def extract_mnist_ubyte_to_png(images_filepath, labels_filepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(labels_filepath, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(images_filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

    print(f"Extraction de {num_images} images en cours...")

    for i in range(num_images):
        img = Image.fromarray(images[i])
        label = labels[i]

        digit_dir = os.path.join(output_dir, str(label))
        os.makedirs(digit_dir, exist_ok=True)

        img_path = os.path.join(digit_dir, f"img_{i}.png")
        img.save(img_path)

    print(f"Terminé ! Les images sont dans le dossier : '{output_dir}'")


extract_mnist_ubyte_to_png(
    images_filepath='database/train-images.idx3-ubyte',
    labels_filepath='database/train-labels.idx1-ubyte',
    output_dir='database/mnist_png/train'
)

extract_mnist_ubyte_to_png(
    images_filepath='database/t10k-images.idx3-ubyte',
    labels_filepath='database/t10k-labels.idx1-ubyte',
    output_dir='database/mnist_png/exec'
)