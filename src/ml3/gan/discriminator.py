from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, ZeroPadding2D, BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

import ml3.config as cfg


def build_discriminator():
    activation = LeakyReLU(alpha=0.2)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=cfg.IMAGE_SHAPE, padding="same"))
    model.add(activation)
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(activation)
    model.add(Dropout(0.25))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(activation)
    model.add(Dropout(0.25))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(activation)
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
   # plot_model(model, "plots/discriminator.png")

    return model
