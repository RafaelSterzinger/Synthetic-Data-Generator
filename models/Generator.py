from keras import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
import config as cfg


class Generator:
    def __init__(self):
        model = Sequential()
        model.add(Dense(cfg.latent_dim * 2, input_shape=cfg.latent_dim))
        model.add(LeakyReLU())
        model.add(Dense(cfg.latent_dim * 4))
        model.add(LeakyReLU())
        model.add(cfg.latent_dim * 8)
        model.add(LeakyReLU())
        model.add(Dense(cfg.feature_dim, activation='sigmoid'))
        self.model = model