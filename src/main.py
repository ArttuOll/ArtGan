import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory, get_file
from pathlib import Path

autotune = tf.data.AUTOTUNE

training_dataset = image_dataset_from_directory(
    "./training_data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=1,
)

validation_dataset = image_dataset_from_directory(
    "./training_data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=1,
)

print(training_dataset)
print(validation_dataset)
