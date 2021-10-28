from os import listdir
import tensorflow as tf
from tensorflow import keras
import numpy as np

IMAGE_HIGHT = 64
IMAGE_WIDTH = 64


def main():
    art_dataset = load_dataset("data/art")
    cityscape_dataset = load_dataset("data/cityscape")

    art_dataset = process_dataset(art_dataset)
    cityscape_dataset = process_dataset(cityscape_dataset)

    print(art_dataset.shape)


def normalize(dataset):
    return (dataset / 127.5) - 1


def process_dataset(dataset):
    return normalize(dataset)


def load_dataset(path):
    images = list()
    for filename in listdir(path):

        image = tf.keras.preprocessing.image.load_img(
            path + '/' + filename, target_size=(IMAGE_HIGHT, IMAGE_WIDTH))
        array = tf.keras.preprocessing.image.img_to_array(image)
        images.append(array)
    return np.stack(images)


if __name__ == "__main__":
    main()
