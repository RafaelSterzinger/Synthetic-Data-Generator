import numpy as np
import ml3.config as cfg
import matplotlib.pyplot as plt
from ml3.eval import *
from ml3.gan import generator, discriminator
from PIL import Image

# https://pathmind.com/wiki/generative-adversarial-network-gan
from ml3.gan.gan import GAN


def train(epoch: int, batch_size=128):
    pass

    # generator = Generator.generator()

    # for epoch in range(epoch):
    #     # create fake images
    #     noise = np.random.normal(0, 1, (1,cfg.latent_dim))
    #     fake_img = generator.predict(noise)
    #     img = Image.fromarray(np.squeeze(fake_img))
    #     img.show()

    # train discriminator
    # dis_loss_real = discriminator.predict_on_batch()


def train_gan():
    gan = GAN()
    gan.train()


if __name__ == '__main__':
    train(100)
