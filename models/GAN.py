import numpy as np
import config as cfg
from models import Discriminator, Generator
import matplotlib.pyplot as plt


def train(epoch: int, batch_size=128):
    generator = Generator.generator()

    for epoch in range(epoch):
        # create fake images
        noise = np.random.normal(0, 1, (1,cfg.latent_dim))
        type(noise)
        fake_img = generator.predict(noise)
        plt.imshow(np.squeeze(fake_img))

        # train discriminator
        # dis_loss_real = discriminator.predict_on_batch()


if __name__ == '__main__':
    train(100)
