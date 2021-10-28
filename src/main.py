from os import listdir
import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np
import math

from tensorflow.python.data.ops.dataset_ops import Dataset


def main():
    (
        art_dataset_training,
        art_dataset_validation,
        cityscape_dataset_training,
        cityscape_dataset_validation,
    ) = _get_datasets()

    # Preprocess training data
    art_dataset_training.map(preprocess_train_image).cache().shuffle(
        232
    ).batch(1)
    cityscape_dataset_training.map(preprocess_train_image).cache().shuffle(
        232
    ).batch(1)

    # Preprocess validation data
    art_dataset_validation.map(normalize_pixel_values).cache().shuffle(
        232
    ).batch(1)
    cityscape_dataset_validation.map(normalize_pixel_values).cache().shuffle(
        232
    ).batch(1)


def load_dataset(path):
    images = list()
    for filename in listdir(path):

        image = load_img(path + "/" + filename)
        array = tf.keras.preprocessing.image.img_to_array(image)
        images.append(array)
    return np.stack(images)


def _validation_split(dataset):
    validation_split = math.floor(len(dataset) * 0.8)

    dataset_training = dataset[:validation_split]
    dataset_validation = dataset[validation_split:]

    return dataset_training, dataset_validation


def _convert_to_dataset_object(dataset):
    return Dataset.from_tensor_slices(dataset)


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


def normalize_pixel_values(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img):
    img = tf.image.random_flip_left_right(img)
    img = normalize_pixel_values(img)
    return img


if __name__ == "__main__":
    main()
