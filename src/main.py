from os import listdir
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

from tensorflow.python.data.ops.dataset_ops import Dataset

IMAGE_HIGHT = 64
IMAGE_WIDTH = 64


def main():
    (
        art_dataset_training,
        art_dataset_validation,
        cityscape_dataset_training,
        cityscape_dataset_validation,
    ) = _get_datasets()


def load_dataset(path):
    images = list()
    for filename in listdir(path):

        image = tf.keras.preprocessing.image.load_img(
            path + "/" + filename, target_size=(IMAGE_HIGHT, IMAGE_WIDTH)
        )
        array = tf.keras.preprocessing.image.img_to_array(image)
        images.append(array)
    return np.stack(images)


def _validation_split(dataset):
    validation_split = math.floor(len(dataset) * 0.8)

    dataset_training = dataset[:validation_split]
    dataset_validation = dataset[validation_split:]

    return dataset_training, dataset_validation


def _convert_to_dataset_object(dataset):
    return Dataset.from_sparse_tensor_slices(dataset)


def _get_datasets():
    art_dataset = load_dataset("./training_data/art")
    cityscape_dataset = load_dataset("./training_data/cityscapes")

    art_dataset_training, art_dataset_validation = _validation_split(
        art_dataset
    )

    (
        cityscape_dataset_training,
        cityscape_dataset_validation,
    ) = _validation_split(cityscape_dataset)

    art_dataset_training = _convert_to_dataset_object(art_dataset_training)
    cityscape_dataset_training = _convert_to_dataset_object(
        cityscape_dataset_training
    )
    art_dataset_validation = _convert_to_dataset_object(art_dataset_validation)
    cityscape_dataset_validation = _convert_to_dataset_object(
        cityscape_dataset_validation
    )

    return (
        art_dataset_training,
        art_dataset_validation,
        cityscape_dataset_training,
        cityscape_dataset_validation,
    )


if __name__ == "__main__":
    main()
