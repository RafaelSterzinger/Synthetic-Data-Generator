import numpy as np
from ml3 import config as cfg
import matplotlib.pyplot as plt
from ml3.gan.generator import generator


def train(epoch: int, batch_size=128):
    gen = generator()
    for epoch in range(epoch):
        # create fake images
        noise = np.random.normal(0, 1, (1, cfg.latent_dim))
        type(noise)
        fake_img = gen.predict(noise)
        plt.imshow(np.squeeze(fake_img))

        # train discriminator
        # dis_loss_real = discriminator.predict_on_batch()


if __name__ == '__main__':
    train(100)
