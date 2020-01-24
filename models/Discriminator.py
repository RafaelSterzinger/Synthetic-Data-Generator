from keras import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

import config as cfg


class Discriminator:
    def __init__(self):
        model = Sequential()
        model.add(Dense(cfg.latent_dim * 8, input_dim=cfg.feature_dim))
        model.add(LeakyReLU())
        model.add(Dropout())
        model.add(Dense(cfg.latent_dim*4))
        model.add(LeakyReLU())
        model.add(Dropout)
        model.add(cfg.latent_dim*2)
        model.add(LeakyReLU())
        model.add(Dropout())
        model.add(Dense(1))
        self.model = model
