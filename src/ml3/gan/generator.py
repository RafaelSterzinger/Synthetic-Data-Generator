from tensorflow.keras.layers import Dense, UpSampling2D, Conv2D, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

import ml3.config as cfg


# TODO: Make generator dynamic e.g. images 128x128
def build_generator():
    noise_shape = (cfg.SEED_SIZE,)

    model = Sequential()

    model.add(Dense(128 * 16 * 16, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((16, 16, 128)))

    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(cfg.SIZE, kernel_size=3, padding="same"))  # 64 instead of cfg.size
    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=3, padding="same", activation='tanh'))

    return model


def load_generator(path: str, epoch: int):
    return load_model(f'{path}/model_epoch_{epoch}.h5')
