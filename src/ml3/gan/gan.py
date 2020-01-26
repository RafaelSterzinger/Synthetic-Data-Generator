from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from ml3.config import *
from PIL import Image

import tensorflow as tf
import time
import numpy as np
import os


# from https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb

class Generator(object):

    def __init__(self) -> None:
        super().__init__()
        self.model = self.__build()

    def __build(self):
        model = Sequential()

        model.add(Dense(4 * 4 * 256, activation="relu", input_dim=SEED_SIZE))
        model.add(Reshape((4, 4, 256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # Output resolution, additional upsampling
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        if GENERATE_RES > 1:
            model.add(UpSampling2D(size=(GENERATE_RES, GENERATE_RES)))
            model.add(Conv2D(128, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        # Final CNN layer
        model.add(Conv2D(IMAGE_CHANNELS, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        return model

    def plot_model(self):
        plot_model(self.model, "plots/generator.png")


class Discriminator(object):

    def __init__(self) -> None:
        super().__init__()
        self.model = self.__build()

    def __build(self):
        image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


class GAN(object):

    def __init__(self) -> None:
        super().__init__()
        self.generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
        self.generator = Generator()
        self.discriminator = Discriminator()

    # This method returns a helper function to compute cross entropy loss
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, real_images):
        seed = tf.random.normal([EVAL_BATCH_SIZE, SEED_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator.model(seed, training=True)

            real_output = self.discriminator.model(real_images, training=True)
            fake_output = self.discriminator.model(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.model.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.model.trainable_variables))
        return gen_loss, disc_loss

    def save_images(self, cnt, noise):
        image_array = np.full((
            PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE + PREVIEW_MARGIN)),
            PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE + PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        generated_images = self.generator.model.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        image_count = 0
        for row in range(PREVIEW_ROWS):
            for col in range(PREVIEW_COLS):
                r = row * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
                c = col * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
                image_array[r:r + GENERATE_SQUARE, c:c + GENERATE_SQUARE] = generated_images[image_count] * 255
                image_count += 1

        output_path = os.path.join(DATA_PATH, 'output')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = os.path.join(output_path, f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)

    def train(self, dataset):
        fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
        start = time.time()

        for epoch in range(EPOCHS):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for image_batch in dataset:
                t = self.train_step(image_batch)
                gen_loss_list.append(t[0])
                disc_loss_list.append(t[1])

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            epoch_elapsed = time.time() - epoch_start
            print(f'Epoch {epoch + 1}, gen loss={g_loss},disc loss={d_loss}, {epoch_elapsed}')
            self.save_images(epoch, fixed_seed)

        elapsed = time.time() - start
        print(f'Training time: {elapsed}')
