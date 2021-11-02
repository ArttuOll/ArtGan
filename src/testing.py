import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
from keras.preprocessing.image import load_img


# This script reads images from `test_images` directory, then uses the trained model to
# convert them to art styles and save them on the same directory, with `_art` added  to
# the name.

def normalize_pixel_values(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def load_test_images(path):
    images = list()
    files_list = listdir(path)
    for filename in files_list:
        image = load_img(path + "/" + filename,
                         target_size=(256, 256), interpolation='bilinear')
        array = tf.keras.preprocessing.image.img_to_array(image)
        images.append((filename.split(".")[-2], array))
    return images


# directory that contains pictures from Joensuu/Kupio.
test_images = load_test_images("./test_images")
model = keras.models.load_model("./trained_model")

for i, named_image in enumerate(test_images):
    name, img = named_image
    img = normalize_pixel_values(img)
    art = model.predict(img[None, :, :, :])*0.5 + 0.5
    plt.figure()
    plt.imshow(art, interpolation='bilinear')
    plt.savefig("./test_images/" + name + "_art.jpg")
