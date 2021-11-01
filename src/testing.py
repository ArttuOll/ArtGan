import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
from keras.preprocessing.image import load_img


def normalize_pixel_values(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def load_test_images(path):
    images = list()
    for filename in listdir(path):
        image = load_img(path + "/" + filename, target_size=(256, 256))
        array = tf.keras.preprocessing.image.img_to_array(image)
        images.append(array)
    return np.stack(images)


# directory that contains pictures from Joensuu/Kupio.
test_images = load_test_images("./test_images")
test_images = normalize_pixel_values(test_images)

model = keras.models.load_model("./trained_model")

fig = plt.figure(figsize=(8, 8))
for i in range(5):
    original = test_images[i, :, :, :]*0.5 + 0.5
    art = model.predict(test_images[i:i+1, :, :, :])*0.5 + 0.5

    fig.add_subplot(5, 2, i * 2 + 1)
    plt.imshow(original)

    fig.add_subplot(5, 2, i * 2 + 2)
    plt.imshow(art)

plt.show()
