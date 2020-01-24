import numpy as np
from keras import Sequential, Model
from keras.layers import Dense, Reshape, Input, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import plot_model

import config as cfg


def generator():
    model = Sequential()
    model.add(Dense(cfg.latent_dim * 2, input_shape=(cfg.latent_dim,)))
    model.add(LeakyReLU())
    model.add(Dense(cfg.latent_dim * 4))
    model.add(LeakyReLU())
    model.add(Dense(cfg.latent_dim * 8))
    model.add(LeakyReLU())
    model.add(Dense(cfg.feature_dim, activation='tanh'))
    model.add(Reshape((cfg.size, cfg.size)))
   # noise = Input(shape=(cfg.latent_dim,))
   # model = model(noise)
   # model = Model(noise, model)
    model.compile(optimizer=Adam(beta_1=0.0, beta_2=0.9), loss='binary_crossentropy')

    model.summary()
    plot_model(model, "plots/generator.png")

    return model
