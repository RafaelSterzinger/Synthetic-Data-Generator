from keras.layers import Dense, UpSampling2D, Conv2D, Reshape, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

import ml3.config as cfg

# Make generator dynamic
def build_generator():
    noise_shape = (cfg.SEED_SIZE,)
    activation = LeakyReLU(alpha=0.2)

    model = Sequential()

    model.add(Dense(128 * 16 * 16, activation=activation,
                    input_shape=noise_shape))

    model.add(Reshape((16, 16, 128)))

    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(activation)

    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(activation)

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=3, padding="same", activation='tanh'))

    model.summary()

    return model
