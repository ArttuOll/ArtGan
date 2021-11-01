from os import listdir
import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np
import math
from CycleGan import CycleGan
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.python.data.ops.dataset_ops import Dataset
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Size of the random crops to be used during training.
input_img_size = (256, 256, 3)


def main():
    (
        art_dataset_training,
        art_dataset_validation,
        cityscape_dataset_training,
        cityscape_dataset_validation,
    ) = get_datasets("./training_data")

    # Preprocess training data
    art_dataset_training = art_dataset_training.map(preprocess_train_image).cache().shuffle(
        232
    ).batch(1)
    cityscape_dataset_training = cityscape_dataset_training.map(preprocess_train_image).cache().shuffle(
        232
    ).batch(1)

    # Preprocess validation data
    art_dataset_validation = art_dataset_validation.map(normalize_pixel_values).cache().shuffle(
        232
    ).batch(1)
    cityscape_dataset_validation = cityscape_dataset_validation.map(normalize_pixel_values).cache().shuffle(
        232
    ).batch(1)

    model = build_model()

    # Train the model
    model.fit(
        tf.data.Dataset.zip(
            (cityscape_dataset_training, art_dataset_training)
        ),
        batch_size=1,
        epochs=10,
    )

    fig = plt.figure(figsize=(8, 8))
    for i, img in enumerate(cityscape_dataset_training.take(5)):
        original_image = img[0, :, :, :].numpy() * 0.5 + 0.5
        art_image = model.predict(img) * 0.5 + 0.5

        fig.add_subplot(5, 2, i*2 + 1)
        plt.imshow(original_image)

        fig.add_subplot(5, 2, i*2 + 2)
        plt.imshow(art_image)

    plt.show()

    # Save the model
    model.save("./trained_model")


def load_dataset(path):
    images = list()
    for filename in listdir(path):
        image = load_img(path + "/" + filename, target_size=(256, 256))
        array = tf.keras.preprocessing.image.img_to_array(image)
        images.append(array)
    return np.stack(images)


def validation_split(dataset):
    validation_split = math.floor(len(dataset) * 0.8)

    dataset_training = dataset[:validation_split]
    dataset_validation = dataset[validation_split:]

    return dataset_training, dataset_validation


def convert_to_dataset_object(dataset):
    return Dataset.from_tensor_slices(dataset)


def get_datasets(path):
    art_dataset = load_dataset(path + "/art")
    cityscape_dataset = load_dataset(path + "/cityscapes")

    art_dataset_training, art_dataset_validation = validation_split(
        art_dataset
    )

    (
        cityscape_dataset_training,
        cityscape_dataset_validation,
    ) = validation_split(cityscape_dataset)

    art_dataset_training = convert_to_dataset_object(art_dataset_training)
    cityscape_dataset_training = convert_to_dataset_object(
        cityscape_dataset_training
    )
    art_dataset_validation = convert_to_dataset_object(art_dataset_validation)
    cityscape_dataset_validation = convert_to_dataset_object(
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


# Train the end-to-end model
# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredLogarithmicError()


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def build_model():
    OUTPUT_CHANNELS = 3
    generator_g = pix2pix.unet_generator(
        OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(
        OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(
        norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(
        norm_type='instancenorm', target=False)

    cycle_gan_model = CycleGan(
        generator_G=generator_g,
        generator_F=generator_f,
        discriminator_X=discriminator_x,
        discriminator_Y=discriminator_y,
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    return cycle_gan_model


if __name__ == "__main__":
    main()
