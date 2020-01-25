from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import plot_model

from ml3 import config as cfg


def discriminator():
    model = Sequential()
    model.add(Dense(cfg.latent_dim * 8, input_dim=cfg.feature_dim))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(cfg.latent_dim * 4))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(cfg.latent_dim * 2))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    img = Input(shape=cfg.image_shape)
    model = model(img)
    model = Model(img, model)
    model.compile(optimizer=Adam(beta_1=0.0, beta_2=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    plot_model(model, "plots/discriminator.png")

    return model
