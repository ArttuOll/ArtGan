from os import listdir
import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np
import math
from GANMonitor import GANMonitor
from CycleGan import CycleGan
from ReflectionPadding2D import ReflectionPadding2D
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
    ) = _get_datasets("./training_data")

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

    # Get the generators
    gen_G = get_resnet_generator(name="generator_G")
    gen_F = get_resnet_generator(name="generator_F")

    # Get the discriminators
    disc_X = get_discriminator(name="discriminator_X")
    disc_Y = get_discriminator(name="discriminator_Y")
    model = build_model(gen_G, gen_F, disc_X, disc_Y)

    # Callback for periodically saving generated images
    plotter = GANMonitor()
    checkpoint_filepath = (
        "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
    )
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath
    )

    # Train the model

    model.fit(
        tf.data.Dataset.zip(
            (cityscape_dataset_training, art_dataset_training)
        ),
        # callbacks=[plotter, model_checkpoint_callback],
        batch_size=1,
        epochs=1,
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


def _validation_split(dataset):
    validation_split = math.floor(len(dataset) * 0.8)

    dataset_training = dataset[:validation_split]
    dataset_validation = dataset[validation_split:]

    return dataset_training, dataset_validation


def _convert_to_dataset_object(dataset):
    return Dataset.from_tensor_slices(dataset)


def _get_datasets(path):
    art_dataset = load_dataset(path + "/art")
    cityscape_dataset = load_dataset(path + "/cityscapes")

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


# Building blocks used in the CycleGan generators and discriminators
def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,


):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(
        x
    )
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(
        x
    )
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(
        x
    )
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(
        x
    )
    if activation:
        x = activation(x)
    return x


# Build the generators
def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(
        filters, (7, 7), kernel_initializer=kernel_init, use_bias=False
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(
        x
    )
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(
            x, filters=filters, activation=layers.Activation("relu")
        )

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

# Build the discriminators


def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1,
        (4, 4),
        strides=(1, 1),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


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


def build_model(gen_G, gen_F, disc_X, disc_Y):
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
